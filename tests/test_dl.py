"""Deep-learning checks (skipped when PyTorch is not installed)."""
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from covidfake.train_dl import build_model, main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/dl.yaml"))


@pytest.mark.parametrize("name", ["cnn", "lstm", "bigru"])
def test_forward_backward(name):
    c = {**CFG["common"], **CFG[name]}
    model = build_model(name, 30, c)
    x = torch.randint(2, 30, (4, 12))
    x[0, 5:] = 0
    out = model(x)
    assert out.shape == (4, 2)
    out.sum().backward()


def test_bigru_end_to_end(tmp_path):
    cfg = dict(CFG)
    cfg["common"] = {**CFG["common"], "epochs": 2, "max_len": 20}
    p = tmp_path / "dl.yaml"
    p.write_text(yaml.safe_dump(cfg))
    main(["--data", str(ROOT / "data/sample.csv"), "--model", "bigru", "--config", str(p), "--out", str(tmp_path)])
    assert (tmp_path / "bigru_metrics.json").exists()
