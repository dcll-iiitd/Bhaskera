"""
bhaskera.launcher.monitoring
============================
Single source of truth for spinning up Bhaskera's observability stack.

Responsibilities
----------------
1. Translate ``cfg.monitoring`` into the env vars Ray actually reads
   (``RAY_GRAFANA_HOST``, ``RAY_PROMETHEUS_HOST``,
   ``RAY_GRAFANA_IFRAME_HOST``, ``RAY_GRAFANA_ORG_ID``,
   ``RAY_PROMETHEUS_NAME``, ``RAY_PROMETHEUS_HEADERS``).
2. Compute the kwargs to pass to ``ray.init``: ``include_dashboard``,
   ``dashboard_host``, ``dashboard_port``, ``_metrics_export_port``.
3. Optionally write a Prometheus scrape-config snippet to disk so a
   sidecar Prometheus can be pointed at it.
4. **Auto-pick up the URLs that ``bhaskera-dashboard`` provisioned.**
   If ``~/.bhaskera/dashboard.json`` exists, its ``grafana_url`` and
   ``prometheus_port`` are loaded as fallbacks — so the in-training
   banner shows the same dashboard URL the user already configured,
   without making them duplicate it in their training YAML.
5. Print the dashboard URL prominently after init so the user always
   knows where to look.

Importantly, this does **not** start Prometheus or Grafana for you —
those are stateful services with their own lifecycle.  But you also
don't need to write configs by hand: either Ray itself writes a
working ``prometheus.yml`` and ``grafana.ini`` to
``/tmp/ray/session_latest/metrics/`` after ``ray.init``, or you can
use ``bhaskera-dashboard`` to drive a long-lived Prometheus + Grafana
stack across runs.

The returned ``MonitoringContext`` is consumed by ``train.py`` and
``infer.py``.
"""
from __future__ import annotations

import json
import logging
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Standard ports — match Ray's defaults so the docs actually apply.
_DEFAULT_DASHBOARD_PORT      = 8265
_DEFAULT_METRICS_EXPORT_PORT = 8080
_DEFAULT_PROMETHEUS_PORT     = 9090
_DEFAULT_GRAFANA_PORT        = 3000

# Path written by ``bhaskera-dashboard`` on first run.
_DASHBOARD_CONFIG_PATH = Path.home() / ".bhaskera" / "dashboard.json"


@dataclass
class MonitoringContext:
    """
    Snapshot of the monitoring config after env-var wiring.

    Returned by ``setup_monitoring`` so the caller can pass the
    relevant kwargs into ``ray.init`` and emit a final summary line
    pointing the user at the dashboard.
    """
    dashboard:           bool                 = True
    dashboard_host:      str                  = "0.0.0.0"
    dashboard_port:      int                  = _DEFAULT_DASHBOARD_PORT
    metrics_export_port: int                  = _DEFAULT_METRICS_EXPORT_PORT
    prometheus_host:     Optional[str]        = None
    grafana_host:        Optional[str]        = None
    grafana_iframe_host: Optional[str]        = None
    grafana_dashboard_url: Optional[str]      = None   # full URL to bhaskera_finetuning dash
    extra_init_kwargs:   dict[str, Any]       = field(default_factory=dict)
    scrape_config_path:  Optional[str]        = None
    source:              str                  = "config"  # "config" | "bhaskera-dashboard" | "default"

    def ray_init_kwargs(self) -> dict[str, Any]:
        """Subset of init kwargs derived from this context."""
        kw: dict[str, Any] = {
            "include_dashboard":     self.dashboard,
            "dashboard_host":        self.dashboard_host,
            "dashboard_port":        self.dashboard_port,
            "_metrics_export_port":  self.metrics_export_port,
        }
        kw.update(self.extra_init_kwargs)
        return kw

    def banner(self, head_ip: Optional[str] = None) -> str:
        """Multi-line, eye-catching summary line for the launcher."""
        ip = head_ip or _local_ip()
        lines = [
            "",
            "═══════════════════════════════════════════════════════════════",
            "  Bhaskera Observability",
            "───────────────────────────────────────────────────────────────",
            f"  Ray Dashboard       : http://{ip}:{self.dashboard_port}"
            if self.dashboard else "  Ray Dashboard       : (disabled)",
            f"  Prometheus scrape   : http://{ip}:{self.metrics_export_port}/metrics",
        ]
        if self.source == "bhaskera-dashboard":
            lines.append(f"  Source              : ~/.bhaskera/dashboard.json")
        else:
            lines += [
                "  Ray-generated cfgs  : /tmp/ray/session_latest/metrics/",
                "    └─ prometheus --config.file=.../prometheus/prometheus.yml",
                "    └─ grafana-server --config .../grafana/grafana.ini web",
            ]
        if self.prometheus_host:
            lines.append(f"  Prometheus server   : {self.prometheus_host}")
        if self.grafana_host:
            lines.append(f"  Grafana             : {self.grafana_host}")
        if self.grafana_dashboard_url:
            lines.append(f"  Dashboard URL       : {self.grafana_dashboard_url}")
        if self.grafana_iframe_host:
            lines.append(f"  Grafana iframe host : {self.grafana_iframe_host}")
        if self.scrape_config_path:
            lines.append(f"  Scrape config       : {self.scrape_config_path}")
        lines.append(
            "═══════════════════════════════════════════════════════════════"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_monitoring(cfg) -> MonitoringContext:
    """
    Translate ``cfg.monitoring`` into Ray init kwargs and env vars.

    Must be called *before* ``ray.init``.  Idempotent: safe to call
    repeatedly (only the first call's values stick into env vars).

    If ``cfg.monitoring`` is missing entirely, defaults are used and
    Ray Dashboard is enabled — that's the explicit "if it is none it
    should by default create ray dashboard" requirement.

    If ``~/.bhaskera/dashboard.json`` exists (written by
    ``bhaskera-dashboard``), its values are used as fallbacks for any
    Prometheus/Grafana settings the YAML config didn't specify.  YAML
    always wins; the saved dashboard config only fills gaps.
    """
    saved = _load_saved_dashboard_config()

    mon = getattr(cfg, "monitoring", None)
    if mon is None:
        # Safe defaults, so users with old configs still get the dashboard.
        ctx = MonitoringContext(source="default" if not saved else "bhaskera-dashboard")
        _apply_saved_dashboard(ctx, saved)
        _emit_scrape_config(ctx, cfg)
        return ctx

    ctx = MonitoringContext(
        dashboard           = bool(getattr(mon, "dashboard",           True)),
        dashboard_host      = str(getattr(mon, "dashboard_host",       "0.0.0.0")),
        dashboard_port      = int(getattr(mon, "dashboard_port",       _DEFAULT_DASHBOARD_PORT)),
        metrics_export_port = int(getattr(mon, "metrics_export_port",  _DEFAULT_METRICS_EXPORT_PORT)),
    )

    # ── Prometheus ─────────────────────────────────────────────────
    prom = getattr(mon, "prometheus", None)
    if prom is not None and getattr(prom, "enabled", False):
        host = str(getattr(prom, "host", "")).strip()
        if host:
            ctx.prometheus_host = host
            os.environ["RAY_PROMETHEUS_HOST"] = host
        name = str(getattr(prom, "name", "Prometheus") or "Prometheus")
        os.environ["RAY_PROMETHEUS_NAME"] = name
        headers = getattr(prom, "headers", None)
        if headers:
            try:
                os.environ["RAY_PROMETHEUS_HEADERS"] = json.dumps(headers)
            except Exception as e:
                logger.warning(f"Could not serialise prometheus.headers: {e}")

    # ── Grafana ────────────────────────────────────────────────────
    graf = getattr(mon, "grafana", None)
    if graf is not None and getattr(graf, "enabled", False):
        host = str(getattr(graf, "host", "")).strip()
        if host:
            ctx.grafana_host = host
            os.environ["RAY_GRAFANA_HOST"] = host
        iframe = str(getattr(graf, "iframe_host", "") or "").strip()
        if iframe:
            ctx.grafana_iframe_host = iframe
            os.environ["RAY_GRAFANA_IFRAME_HOST"] = iframe
        org = str(getattr(graf, "org_id", "1") or "1")
        os.environ["RAY_GRAFANA_ORG_ID"] = org

    # Fill any gaps from the saved bhaskera-dashboard config.
    _apply_saved_dashboard(ctx, saved)

    _emit_scrape_config(ctx, cfg)
    return ctx


# ---------------------------------------------------------------------------
# bhaskera-dashboard fallback wiring
# ---------------------------------------------------------------------------

def _load_saved_dashboard_config() -> Optional[dict]:
    """Read ``~/.bhaskera/dashboard.json`` if it exists. Never raises."""
    if not _DASHBOARD_CONFIG_PATH.exists():
        return None
    try:
        return json.loads(_DASHBOARD_CONFIG_PATH.read_text())
    except Exception as e:
        logger.debug(f"Could not read {_DASHBOARD_CONFIG_PATH}: {e}")
        return None


def _apply_saved_dashboard(ctx: MonitoringContext, saved: Optional[dict]) -> None:
    """Use saved bhaskera-dashboard values to fill gaps in ``ctx``.

    YAML config wins — this only fills slots the YAML left empty.
    Mirrors the values into the env vars Ray reads, again only when
    they aren't already set.
    """
    if not saved:
        return

    # Grafana host: prefer YAML, fall back to saved.
    grafana_url = str(saved.get("grafana_url", "")).strip()
    if grafana_url and not ctx.grafana_host:
        ctx.grafana_host = grafana_url
        os.environ.setdefault("RAY_GRAFANA_HOST", grafana_url)
        ctx.source = "bhaskera-dashboard" if ctx.source == "config" else ctx.source

    # Prometheus host: bhaskera-dashboard runs Prometheus locally on the
    # head node, so localhost:<port> is the right URL from inside the
    # cluster. Users wanting cross-node access can override in YAML.
    prom_port = int(saved.get("prometheus_port", _DEFAULT_PROMETHEUS_PORT))
    prom_url  = f"http://localhost:{prom_port}"
    if not ctx.prometheus_host:
        ctx.prometheus_host = prom_url
        os.environ.setdefault("RAY_PROMETHEUS_HOST", prom_url)
        os.environ.setdefault("RAY_PROMETHEUS_NAME", "Prometheus")
        ctx.source = "bhaskera-dashboard" if ctx.source == "config" else ctx.source

    # Best-effort dashboard URL — assumes the same slug Grafana returned
    # on import. We can't recompute the exact slug here without another
    # REST call, so we just point at the Grafana root, which always works.
    if grafana_url and not ctx.grafana_dashboard_url:
        ctx.grafana_dashboard_url = grafana_url

    # Override metrics port if the user pinned one via bhaskera-dashboard
    # AND the YAML didn't override it. (We can detect "didn't override"
    # by checking it's still the default.)
    saved_metrics = saved.get("metrics_port")
    if (
        saved_metrics
        and ctx.metrics_export_port == _DEFAULT_METRICS_EXPORT_PORT
    ):
        try:
            ctx.metrics_export_port = int(saved_metrics)
        except (TypeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Optional scrape-config emission
# ---------------------------------------------------------------------------

def _emit_scrape_config(ctx: MonitoringContext, cfg) -> None:
    """
    Drop a Prometheus scrape-config snippet next to the run logs so a
    sidecar Prometheus can pick up the cluster without the user
    hand-editing ``prometheus.yml``.

    Ray itself writes one to ``/tmp/ray/session_latest/metrics/...`` —
    we don't compete with that.  We write a Bhaskera-flavoured one
    that adds a ``job: bhaskera`` label so users running multiple
    frameworks against one Prometheus can filter cleanly.
    """
    save_dir = getattr(getattr(cfg, "checkpoint", None), "save_dir", "./checkpoints")
    out_dir = os.path.join(os.path.abspath(save_dir), "_monitoring")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        logger.debug(f"Could not create monitoring dir {out_dir}: {e}")
        return

    run_name = getattr(getattr(cfg, "logging", None), "run_name", "run")
    project  = getattr(getattr(cfg, "logging", None), "project",  "bhaskera")

    snippet = f"""# Bhaskera — Prometheus scrape config (generated)
# Add this under `scrape_configs:` in your prometheus.yml.

- job_name: bhaskera
  metrics_path: /metrics
  static_configs:
    - targets:
        - {_local_ip()}:{ctx.metrics_export_port}
      labels:
        framework: bhaskera
        run: {run_name}
        project: {project}
"""
    path = os.path.join(out_dir, "prometheus_scrape.yml")
    try:
        with open(path, "w") as f:
            f.write(snippet)
        ctx.scrape_config_path = path
    except Exception as e:
        logger.debug(f"Could not write scrape config to {path}: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    """Best-effort local IP, falls back to ``127.0.0.1``."""
    try:
        # Connecting to a public IP doesn't actually send a packet; it
        # just makes the kernel pick the outgoing interface.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.2)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
