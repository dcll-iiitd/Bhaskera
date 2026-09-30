import json

import pytest

from bhaskera.launcher.calibrate import main
from bhaskera.serve.decision.calibration import Calibration


def _rows(path):
    rows = [{"type": "boolean", "logits": [0.0, 2.0], "label_index": i % 2} for i in range(20)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_fit_writes_a_loadable_calibration(tmp_path, capsys):
    rows, out = tmp_path / "rows.jsonl", tmp_path / "cal.json"
    _rows(rows)
    main(["--rows", str(rows), "--fingerprint", "fp", "--out", str(out)])
    calibration = Calibration.from_file(out)
    assert calibration.fingerprint == "fp" and "boolean" in calibration.temperatures
    assert "held-out" in capsys.readouterr().out


def test_never_overwrites(tmp_path):
    rows, out = tmp_path / "rows.jsonl", tmp_path / "cal.json"
    _rows(rows)
    out.write_text("keep")
    with pytest.raises(SystemExit):
        main(["--rows", str(rows), "--fingerprint", "fp", "--out", str(out)])
    assert out.read_text() == "keep"
