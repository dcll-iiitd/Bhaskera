"""
bhaskera.launcher.dashboard
===========================
``bhaskera-dashboard`` — one-shot CLI that wires up Prometheus + Grafana for
a Bhaskera run. Hides every step you'd normally do by hand on an HPC cluster:

    1. Discover the worker nodes (auto from SLURM, or ``--nodes``).
    2. Render a fresh ``prometheus.yml`` pointing at every node:metrics_port.
    3. Start (or SIGHUP-reload) Prometheus as a detached background process.
    4. Register the Prometheus datasource in Grafana via REST.
    5. Import ``configs/grafana/bhaskera_finetuning.json`` via REST.
    6. Print the dashboard URL — and, on HPC where the compute node is
       firewalled, the exact ``ssh -L`` command (or open the tunnel for you).

Design
------
* **First run** takes paths once; everything is persisted to
  ``~/.bhaskera/dashboard.json``.
* **Every run after** is just ``bhaskera-dashboard`` — config is loaded from
  disk, idempotent ops re-apply.
* **Subcommands**: ``start`` (default), ``stop``, ``status``, ``tunnel``.
* **No Docker, no root.** Works with any prometheus / grafana binary the
  user already extracted somewhere on ``$SCRATCH``.

Typical first-run on an HPC head node
-------------------------------------
::

    bhaskera-dashboard \\
        --prometheus-bin   /scratch/me/prometheus-2.54/prometheus  \\
        --prometheus-data  /scratch/me/monitoring/prometheus-data  \\
        --log-dir          /scratch/me/monitoring/logs             \\
        --grafana-url      http://gpunode01:3000                   \\
        --grafana-user     admin                                   \\
        --grafana-password admin                                   \\
        --login-node       login.cluster.example.edu               \\
        --metrics-port     8080                                    \\
        --dashboard-json   configs/grafana/bhaskera_finetuning.json

Subsequent runs from inside any SLURM job:
::

    bhaskera-dashboard          # start / reload + print SSH tunnel command
    bhaskera-dashboard status   # is prometheus alive? is grafana reachable?
    bhaskera-dashboard tunnel   # just print / open the ssh -L command
    bhaskera-dashboard stop     # kill prometheus
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("bhaskera.dashboard")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PROM_PORT     = 9090
_DEFAULT_GRAFANA_PORT  = 3000
_DEFAULT_METRICS_PORT  = 8080      # Ray ``_metrics_export_port`` default
_DEFAULT_RETENTION     = "90d"
_DEFAULT_SCRAPE_EVERY  = "15s"
_DEFAULT_DASHBOARD_REL = "configs/grafana/bhaskera_finetuning.json"

_CONFIG_DIR  = Path.home() / ".bhaskera"
_CONFIG_PATH = _CONFIG_DIR / "dashboard.json"
_PID_FILE    = _CONFIG_DIR / "prometheus.pid"
_TUNNEL_PID  = _CONFIG_DIR / "ssh_tunnel.pid"


# ---------------------------------------------------------------------------
# Persisted config
# ---------------------------------------------------------------------------

@dataclass
class DashboardConfig:
    """Everything ``bhaskera-dashboard`` needs to start the stack.

    Persisted to ``~/.bhaskera/dashboard.json`` after the first run so
    subsequent invocations need zero arguments.
    """
    # Prometheus
    prometheus_bin:    str = ""
    prometheus_data:   str = ""
    prometheus_port:   int = _DEFAULT_PROM_PORT
    retention:         str = _DEFAULT_RETENTION
    scrape_interval:   str = _DEFAULT_SCRAPE_EVERY

    # Logs / scratch
    log_dir:           str = ""

    # Targets
    metrics_port:      int = _DEFAULT_METRICS_PORT
    nodes:             list[str] = field(default_factory=list)   # explicit override

    # Grafana
    grafana_url:       str = f"http://localhost:{_DEFAULT_GRAFANA_PORT}"
    grafana_user:      str = "admin"
    grafana_password:  str = "admin"
    dashboard_json:    str = ""    # path to bhaskera_finetuning.json

    # SSH port forwarding (HPC)
    login_node:        str = ""    # e.g. "login.cluster.example.edu"
    ssh_user:          str = ""    # defaults to $USER if empty
    ssh_port:          int = 22    # SSH port on the login node (e.g. 4422)
    ssh_identity:      str = ""    # path to private key, e.g. ~/.sdh/id_rsa
    ssh_opts:          str = ""    # extra raw ssh opts, e.g. "-A"
    services_on_login: bool = True # True: grafana/prom live on the login node;
                                   # False: they live on a compute node and we
                                   # need a two-hop ProxyJump to reach them.
    local_grafana_port: int = _DEFAULT_GRAFANA_PORT
    local_prom_port:    int = _DEFAULT_PROM_PORT

    # ── persistence ────────────────────────────────────────────────
    @classmethod
    def load(cls) -> "DashboardConfig":
        if not _CONFIG_PATH.exists():
            return cls()
        try:
            data = json.loads(_CONFIG_PATH.read_text())
        except Exception as e:
            logger.warning(f"Could not read {_CONFIG_PATH}: {e} — starting fresh.")
            return cls()
        # Tolerate older configs missing newer fields.
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def save(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))
        # Friendly perms — this file may have a Grafana password in it.
        try:
            os.chmod(_CONFIG_PATH, 0o600)
        except OSError:
            pass

    # ── validation ─────────────────────────────────────────────────
    def require_prometheus(self) -> None:
        missing = []
        if not self.prometheus_bin:  missing.append("--prometheus-bin")
        if not self.prometheus_data: missing.append("--prometheus-data")
        if not self.log_dir:         missing.append("--log-dir")
        if missing:
            sys.exit(
                "Missing required arguments on first run: "
                + ", ".join(missing)
                + f"\nRun once with these flags; they will be saved to {_CONFIG_PATH}."
            )

    def require_grafana(self) -> None:
        if not self.grafana_url:
            sys.exit("--grafana-url is required (e.g. http://localhost:3000)")


# ---------------------------------------------------------------------------
# Node discovery
# ---------------------------------------------------------------------------

def discover_nodes(explicit: list[str] | None) -> list[str]:
    """Return the list of hostnames Prometheus should scrape.

    Preference order:
        1. ``explicit`` (the ``--nodes`` flag).
        2. ``SLURM_JOB_NODELIST`` expanded via ``scontrol show hostnames``.
        3. Fall back to localhost.
    """
    if explicit:
        return explicit

    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if nodelist:
        scontrol = shutil.which("scontrol")
        if scontrol:
            try:
                out = subprocess.check_output(
                    [scontrol, "show", "hostnames", nodelist],
                    text=True, timeout=10,
                )
                hosts = [h.strip() for h in out.splitlines() if h.strip()]
                if hosts:
                    logger.info(f"Discovered {len(hosts)} node(s) from SLURM: {hosts}")
                    return hosts
            except Exception as e:
                logger.warning(f"scontrol expansion failed: {e}")
        # No scontrol — try to use the raw value if it looks like a single host.
        if "[" not in nodelist:
            return [nodelist]

    logger.warning(
        "No --nodes given and no SLURM_JOB_NODELIST in env — "
        "falling back to localhost. Pass --nodes to override."
    )
    return [socket.gethostname()]


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------

def render_prometheus_yml(cfg: DashboardConfig, nodes: list[str]) -> Path:
    """Write a fresh ``prometheus.yml`` and return its path.

    Lives next to the data dir so ``--config.file`` and ``--storage.tsdb.path``
    are co-located and easy to clean up.
    """
    data_dir = Path(cfg.prometheus_data)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = ",\n          ".join(f"'{h}:{cfg.metrics_port}'" for h in nodes)
    yml = f"""# Auto-generated by bhaskera-dashboard — DO NOT EDIT BY HAND
# Re-run `bhaskera-dashboard` to regenerate.

global:
  scrape_interval:     {cfg.scrape_interval}
  evaluation_interval: {cfg.scrape_interval}
  external_labels:
    framework: bhaskera

scrape_configs:
  - job_name: bhaskera
    metrics_path: /metrics
    static_configs:
      - targets: [
          {targets}
        ]
        labels:
          framework: bhaskera

  # Prometheus scraping itself — handy for sanity checks.
  - job_name: prometheus
    static_configs:
      - targets: ['localhost:{cfg.prometheus_port}']
"""
    out = data_dir / "prometheus.yml"
    out.write_text(yml)
    logger.info(f"Wrote {out}  ({len(nodes)} target(s))")
    return out


def _read_pid() -> Optional[int]:
    if not _PID_FILE.exists():
        return None
    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    # Is the process still alive?
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        # Stale pidfile.
        try:
            _PID_FILE.unlink()
        except OSError:
            pass
        return None


def start_prometheus(cfg: DashboardConfig, config_path: Path) -> int:
    """Start Prometheus (or SIGHUP it if already running). Returns the PID."""
    existing = _read_pid()
    if existing is not None:
        logger.info(f"Prometheus already running (pid={existing}) — sending SIGHUP for config reload.")
        try:
            os.kill(existing, signal.SIGHUP)
        except OSError as e:
            logger.warning(f"SIGHUP failed: {e} — restarting.")
            stop_prometheus()
            return start_prometheus(cfg, config_path)
        return existing

    if not Path(cfg.prometheus_bin).exists():
        sys.exit(f"prometheus binary not found at {cfg.prometheus_bin}")

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / "prometheus.stdout.log"
    stderr_log = log_dir / "prometheus.stderr.log"

    cmd = [
        cfg.prometheus_bin,
        f"--config.file={config_path}",
        f"--storage.tsdb.path={Path(cfg.prometheus_data) / 'tsdb'}",
        f"--storage.tsdb.retention.time={cfg.retention}",
        f"--web.listen-address=0.0.0.0:{cfg.prometheus_port}",
        # ``--web.enable-lifecycle`` lets the user POST /-/reload too.
        "--web.enable-lifecycle",
    ]
    logger.info(f"Starting prometheus: {' '.join(shlex.quote(c) for c in cmd)}")

    # Detached: survives the parent shell. Output -> log files for debugging.
    with open(stdout_log, "ab") as so, open(stderr_log, "ab") as se:
        proc = subprocess.Popen(
            cmd,
            stdout=so,
            stderr=se,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=cfg.prometheus_data,
        )

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(proc.pid))

    # Give it a beat so we can fail loudly if --config.file is bad.
    time.sleep(1.5)
    if proc.poll() is not None:
        tail = stderr_log.read_text().splitlines()[-20:] if stderr_log.exists() else []
        sys.exit(
            f"Prometheus exited immediately (code={proc.returncode}). "
            f"Last stderr lines:\n  " + "\n  ".join(tail)
        )

    logger.info(f"Prometheus started (pid={proc.pid}). Logs: {stdout_log}")
    return proc.pid


def stop_prometheus() -> bool:
    pid = _read_pid()
    if pid is None:
        logger.info("No Prometheus pidfile — nothing to stop.")
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        logger.warning(f"SIGTERM failed: {e}")
    # Wait briefly; escalate to SIGKILL if needed.
    for _ in range(20):
        try:
            os.kill(pid, 0)
            time.sleep(0.25)
        except OSError:
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        _PID_FILE.unlink()
    except OSError:
        pass
    logger.info(f"Prometheus stopped (was pid={pid}).")
    return True


def prometheus_status(cfg: DashboardConfig) -> dict[str, Any]:
    pid = _read_pid()
    reachable = False
    try:
        import urllib.request
        with urllib.request.urlopen(
            f"http://localhost:{cfg.prometheus_port}/-/ready", timeout=2,
        ) as r:
            reachable = r.status == 200
    except Exception:
        reachable = False
    return {"pid": pid, "reachable": reachable}


# ---------------------------------------------------------------------------
# Grafana (REST API; idempotent)
# ---------------------------------------------------------------------------

def _grafana_request(
    cfg: DashboardConfig,
    method: str,
    path: str,
    body: Optional[dict] = None,
) -> tuple[int, dict]:
    """Tiny REST helper. Uses ``requests`` if available, ``urllib`` otherwise."""
    url = cfg.grafana_url.rstrip("/") + path
    payload = json.dumps(body).encode() if body is not None else None

    try:
        import requests  # type: ignore
        r = requests.request(
            method, url,
            data=payload,
            headers={"Content-Type": "application/json"},
            auth=(cfg.grafana_user, cfg.grafana_password),
            timeout=10,
        )
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text}
    except ImportError:
        pass

    # Fallback: stdlib only.
    import base64
    import urllib.error
    import urllib.request as urlreq

    auth = base64.b64encode(
        f"{cfg.grafana_user}:{cfg.grafana_password}".encode()
    ).decode()
    req = urlreq.Request(
        url,
        data=payload,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth}",
        },
    )
    try:
        with urlreq.urlopen(req, timeout=10) as r:
            text = r.read().decode("utf-8", errors="replace")
            try:
                return r.status, json.loads(text)
            except Exception:
                return r.status, {"raw": text}
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(text)
        except Exception:
            return e.code, {"raw": text}


def ensure_prom_datasource(cfg: DashboardConfig) -> str:
    """Create or update the Prometheus datasource in Grafana. Returns the UID."""
    name = "Prometheus"
    prom_url = f"http://localhost:{cfg.prometheus_port}"

    code, resp = _grafana_request(cfg, "GET", f"/api/datasources/name/{name}")
    if code == 200 and resp.get("uid"):
        uid = resp["uid"]
        # Refresh URL/access in case the user moved the prometheus port.
        body = {**resp, "url": prom_url, "access": "proxy", "isDefault": True}
        _grafana_request(cfg, "PUT", f"/api/datasources/uid/{uid}", body)
        logger.info(f"Grafana datasource '{name}' already exists — updated (uid={uid}).")
        return uid

    body = {
        "name":      name,
        "type":      "prometheus",
        "url":       prom_url,
        "access":    "proxy",
        "isDefault": True,
    }
    code, resp = _grafana_request(cfg, "POST", "/api/datasources", body)
    if code not in (200, 201):
        sys.exit(f"Grafana datasource creation failed (HTTP {code}): {resp}")
    uid = resp.get("datasource", {}).get("uid") or resp.get("uid", "")
    logger.info(f"Created Grafana datasource '{name}' (uid={uid}).")
    return uid


def import_dashboard(cfg: DashboardConfig, datasource_uid: str) -> str:
    """Import the bhaskera dashboard JSON into Grafana. Returns its URL."""
    if not cfg.dashboard_json or not Path(cfg.dashboard_json).exists():
        logger.warning(
            f"dashboard_json not set or missing ({cfg.dashboard_json!r}) — "
            "skipping dashboard import. Pass --dashboard-json to enable."
        )
        return cfg.grafana_url

    raw = json.loads(Path(cfg.dashboard_json).read_text())

    # Strip any pre-existing id/uid so re-imports don't 409. Grafana will
    # reuse the existing dashboard if the *title* matches.
    raw.pop("id", None)

    body = {
        "dashboard": raw,
        "overwrite": True,
        "inputs": [
            {
                "name":     "DS_PROMETHEUS",
                "type":     "datasource",
                "pluginId": "prometheus",
                "value":    "Prometheus",
            }
        ],
    }
    code, resp = _grafana_request(cfg, "POST", "/api/dashboards/import", body)
    # Some Grafana versions only accept /api/dashboards/db.
    if code == 404:
        code, resp = _grafana_request(cfg, "POST", "/api/dashboards/db", body)

    if code not in (200, 201):
        logger.warning(f"Dashboard import returned HTTP {code}: {resp}")
        return cfg.grafana_url

    slug = resp.get("url") or resp.get("importedUrl") or ""
    full = cfg.grafana_url.rstrip("/") + slug if slug else cfg.grafana_url
    logger.info(f"Dashboard imported: {full}")
    return full


# ---------------------------------------------------------------------------
# SSH port forwarding (HPC convenience)
# ---------------------------------------------------------------------------

def _compute_node() -> str:
    """Where is Grafana actually running? On a SLURM head node, that's us."""
    return socket.gethostname()


def build_ssh_command(cfg: DashboardConfig) -> str:
    """Return the exact ``ssh`` command to run from the user's laptop.

    Two modes depending on ``cfg.services_on_login``:

    * **services_on_login=True** (default): grafana/prometheus run on the
      login node itself.  Single-hop tunnel.  ``-L PORT:localhost:PORT``
      where ``localhost`` is resolved on the login-node side.
    * **services_on_login=False**: grafana/prometheus run on a compute
      node behind the login node.  Two-hop tunnel using ``-J`` ProxyJump.

    Honours ``ssh_port``, ``ssh_identity``, ``ssh_opts`` so non-default
    HPC setups (custom port, dedicated key, ``-A`` agent forwarding,
    etc.) work out of the box.
    """
    user      = cfg.ssh_user or os.environ.get("USER", "")
    user_at   = f"{user}@" if user else ""
    grafana_p = _port_from_url(cfg.grafana_url, _DEFAULT_GRAFANA_PORT)

    parts: list[str] = ["ssh", "-N"]
    if cfg.ssh_opts:
        parts.append(cfg.ssh_opts)
    if cfg.ssh_identity:
        parts += ["-i", cfg.ssh_identity]
    if cfg.ssh_port and cfg.ssh_port != 22:
        parts += ["-p", str(cfg.ssh_port)]

    if cfg.services_on_login:
        # Single-hop. The remote side of every -L is the login node, so
        # "localhost" here means "the login node's localhost" — exactly
        # where grafana / prometheus are listening.
        parts += [
            "-L", f"{cfg.local_grafana_port}:localhost:{grafana_p}",
            "-L", f"{cfg.local_prom_port}:localhost:{cfg.prometheus_port}",
            "-L", f"8265:localhost:8265",
            f"{user_at}{cfg.login_node}",
        ]
    else:
        # Two-hop via ProxyJump. Custom port goes inline as user@host:port
        # in the -J spec; the inner hop assumes standard port 22 on the
        # cluster's internal network (true on every HPC I've seen).
        compute = _compute_node()
        jump = f"{user_at}{cfg.login_node}"
        if cfg.ssh_port and cfg.ssh_port != 22:
            jump += f":{cfg.ssh_port}"
        parts += [
            "-L", f"{cfg.local_grafana_port}:{compute}:{grafana_p}",
            "-L", f"{cfg.local_prom_port}:{compute}:{cfg.prometheus_port}",
            "-L", f"8265:{compute}:8265",
            "-J", jump,
            f"{user_at}{compute}",
        ]

    return " ".join(parts)


def _port_from_url(url: str, default: int) -> int:
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        return p.port or default
    except Exception:
        return default


def open_ssh_tunnel(cfg: DashboardConfig) -> Optional[int]:
    """Spawn the ssh tunnel as a background process. Returns its PID, or None.

    Only works if this command is being run *from your laptop* — i.e. the
    machine you'd type the ssh command on. On a compute node this is a no-op
    (the tunnel needs to be initiated from the laptop side).
    """
    if not cfg.login_node:
        logger.info("No --login-node configured; skipping tunnel.")
        return None

    # Stale tunnel?
    if _TUNNEL_PID.exists():
        try:
            old = int(_TUNNEL_PID.read_text().strip())
            os.kill(old, 0)
            logger.info(f"SSH tunnel already running (pid={old}).")
            return old
        except (OSError, ValueError):
            try:
                _TUNNEL_PID.unlink()
            except OSError:
                pass

    cmd = build_ssh_command(cfg)
    logger.info(f"Opening SSH tunnel: {cmd}")
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _TUNNEL_PID.write_text(str(proc.pid))
    time.sleep(1.0)
    if proc.poll() is not None:
        logger.warning(
            "SSH tunnel exited immediately. Run the command above manually "
            "to see the error."
        )
        try:
            _TUNNEL_PID.unlink()
        except OSError:
            pass
        return None
    return proc.pid


def close_ssh_tunnel() -> bool:
    if not _TUNNEL_PID.exists():
        return False
    try:
        pid = int(_TUNNEL_PID.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        logger.info(f"Closed SSH tunnel (pid={pid}).")
    except (OSError, ValueError) as e:
        logger.warning(f"Could not close tunnel: {e}")
    try:
        _TUNNEL_PID.unlink()
    except OSError:
        pass
    return True


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner(cfg: DashboardConfig, dashboard_url: str, ssh_cmd: str) -> None:
    grafana_local = f"http://localhost:{cfg.local_grafana_port}"
    prom_local    = f"http://localhost:{cfg.local_prom_port}"

    bar = "═" * 75
    lines = [
        "",
        bar,
        "  Bhaskera Dashboard",
        "─" * 75,
        f"  Compute node       : {_compute_node()}",
        f"  Prometheus (compute): http://{_compute_node()}:{cfg.prometheus_port}",
        f"  Grafana   (compute): {cfg.grafana_url}",
        f"  Dashboard URL      : {dashboard_url}",
        "",
    ]
    if cfg.login_node:
        lines += [
            "  HPC port-forwarding — run this on your LAPTOP:",
            "",
            f"    {ssh_cmd}",
            "",
            "  Then open in your browser:",
            f"    Grafana    → {grafana_local}",
            f"    Prometheus → {prom_local}",
        ]
    else:
        lines += [
            "  Open in your browser:",
            f"    {dashboard_url}",
        ]
    lines.append(bar)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bhaskera-dashboard",
        description="Start / configure the Bhaskera observability stack.",
    )
    p.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=["start", "stop", "status", "tunnel", "untunnel"],
        help="What to do (default: start).",
    )

    # Prometheus
    p.add_argument("--prometheus-bin",  help="Path to the prometheus binary.")
    p.add_argument("--prometheus-data", help="Directory for prometheus.yml + tsdb.")
    p.add_argument("--prometheus-port", type=int, help=f"Default {_DEFAULT_PROM_PORT}.")
    p.add_argument("--retention",       help=f"TSDB retention (default {_DEFAULT_RETENTION}).")
    p.add_argument("--scrape-interval", help=f"Scrape interval (default {_DEFAULT_SCRAPE_EVERY}).")

    # Logs
    p.add_argument("--log-dir", help="Directory for prometheus stdout/stderr.")

    # Targets
    p.add_argument("--metrics-port", type=int, help="Port each node exposes /metrics on.")
    p.add_argument(
        "--nodes",
        help="Comma-separated hostnames to scrape. Auto-detected from SLURM if omitted.",
    )

    # Grafana
    p.add_argument("--grafana-url",      help="e.g. http://gpunode01:3000")
    p.add_argument("--grafana-user",     help="Grafana admin username.")
    p.add_argument("--grafana-password", help="Grafana admin password.")
    p.add_argument("--dashboard-json",   help="Path to bhaskera_finetuning.json.")

    # SSH / port forwarding
    p.add_argument("--login-node",       help="Cluster login node, for SSH ProxyJump.")
    p.add_argument("--ssh-user",         help="SSH username (default $USER).")
    p.add_argument("--ssh-port", type=int, help="SSH port on the login node (default 22, e.g. 4422).")
    p.add_argument("--ssh-identity",     help="Path to SSH identity file, e.g. ~/.sdh/id_rsa.")
    p.add_argument("--ssh-opts",         help="Extra raw ssh opts, quoted. E.g. \"-A\" for agent forwarding.")
    p.add_argument("--services-on-login", dest="services_on_login", action="store_true", default=None,
                   help="Grafana/Prometheus run on the LOGIN node (default).")
    p.add_argument("--services-on-compute", dest="services_on_login", action="store_false",
                   help="Grafana/Prometheus run on a compute node behind login (two-hop).")
    p.add_argument("--local-grafana-port", type=int,
                   help="Localhost port to forward Grafana to (default 3000).")
    p.add_argument("--local-prom-port", type=int,
                   help="Localhost port to forward Prometheus to (default 9090).")
    p.add_argument("--auto-tunnel", action="store_true",
                   help="Also open the ssh tunnel as a background process.")

    # Misc
    p.add_argument("--reset", action="store_true",
                   help=f"Delete saved config at {_CONFIG_PATH} and start over.")
    p.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args(argv)


def _merge_cli_into_cfg(cfg: DashboardConfig, args: argparse.Namespace) -> DashboardConfig:
    def take(attr: str, target: str | None = None) -> None:
        v = getattr(args, attr, None)
        if v is not None and v != "":
            setattr(cfg, target or attr, v)

    take("prometheus_bin")
    take("prometheus_data")
    take("prometheus_port")
    take("retention")
    take("scrape_interval")
    take("log_dir")
    take("metrics_port")
    take("grafana_url")
    take("grafana_user")
    take("grafana_password")
    take("dashboard_json")
    take("login_node")
    take("ssh_user")
    take("ssh_port")
    take("ssh_identity")
    take("ssh_opts")
    take("local_grafana_port")
    take("local_prom_port")

    # services_on_login is a tri-state at parse time (None=unset, True, False)
    if args.services_on_login is not None:
        cfg.services_on_login = args.services_on_login

    if args.nodes:
        cfg.nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    return cfg


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_start(cfg: DashboardConfig, args: argparse.Namespace) -> int:
    cfg.require_prometheus()
    cfg.require_grafana()

    nodes = discover_nodes(cfg.nodes or None)
    yml   = render_prometheus_yml(cfg, nodes)
    start_prometheus(cfg, yml)

    # Brief warm-up so Grafana can talk to Prometheus.
    time.sleep(1.5)

    try:
        uid = ensure_prom_datasource(cfg)
        dashboard_url = import_dashboard(cfg, uid)
    except SystemExit:
        raise
    except Exception as e:
        logger.warning(f"Grafana provisioning failed: {e}")
        dashboard_url = cfg.grafana_url

    ssh_cmd = build_ssh_command(cfg) if cfg.login_node else ""
    if args.auto_tunnel and cfg.login_node:
        open_ssh_tunnel(cfg)

    print_banner(cfg, dashboard_url, ssh_cmd)
    return 0


def _cmd_stop(cfg: DashboardConfig, args: argparse.Namespace) -> int:
    stopped = stop_prometheus()
    closed  = close_ssh_tunnel()
    if not (stopped or closed):
        print("Nothing to stop.")
    return 0


def _cmd_status(cfg: DashboardConfig, args: argparse.Namespace) -> int:
    s = prometheus_status(cfg)
    print(f"prometheus.pid       : {s['pid'] if s['pid'] else '—'}")
    print(f"prometheus.reachable : {'yes' if s['reachable'] else 'no'} "
          f"(http://localhost:{cfg.prometheus_port}/-/ready)")
    if cfg.grafana_url:
        try:
            import urllib.request
            with urllib.request.urlopen(
                cfg.grafana_url.rstrip("/") + "/api/health", timeout=2,
            ) as r:
                ok = r.status == 200
        except Exception:
            ok = False
        print(f"grafana.reachable    : {'yes' if ok else 'no'}  ({cfg.grafana_url})")
    if _TUNNEL_PID.exists():
        try:
            pid = int(_TUNNEL_PID.read_text().strip())
            os.kill(pid, 0)
            print(f"ssh_tunnel.pid       : {pid}")
        except (OSError, ValueError):
            print("ssh_tunnel.pid       : (stale)")
    return 0


def _cmd_tunnel(cfg: DashboardConfig, args: argparse.Namespace) -> int:
    if not cfg.login_node:
        sys.exit("No --login-node configured. Re-run with --login-node <hostname>.")
    cmd = build_ssh_command(cfg)
    print("Run this on your LAPTOP:\n")
    print(f"    {cmd}\n")
    print(f"Then open  http://localhost:{cfg.local_grafana_port}  in your browser.")
    if args.auto_tunnel:
        open_ssh_tunnel(cfg)
    return 0


def _cmd_untunnel(cfg: DashboardConfig, args: argparse.Namespace) -> int:
    if close_ssh_tunnel():
        print("Tunnel closed.")
    else:
        print("No active tunnel.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.reset and _CONFIG_PATH.exists():
        _CONFIG_PATH.unlink()
        logger.info(f"Removed {_CONFIG_PATH}")

    cfg = DashboardConfig.load()
    cfg = _merge_cli_into_cfg(cfg, args)
    # Persist the (possibly-augmented) config — except on `reset`-only runs.
    cfg.save()

    handlers = {
        "start":    _cmd_start,
        "stop":     _cmd_stop,
        "status":   _cmd_status,
        "tunnel":   _cmd_tunnel,
        "untunnel": _cmd_untunnel,
    }
    return handlers[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
