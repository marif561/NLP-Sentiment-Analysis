from pathlib import Path

import pytest

from covidfake.data import load_data, normalise_label, split
from covidfake.metrics import evaluate
from covidfake.preprocess import clean_text
from covidfake.train_ml import main as ml_main

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample.csv"


def test_clean_text():
    out = clean_text("Drinking HOT water kills the virus!!! https://t.co/abc @who #covid19")
    assert "http" not in out and "@" not in out and "!" not in out
    assert out.split()[:3] == ["drink", "hot", "water"]  # stemmed, stop words removed


def test_labels():
    assert normalise_label("Fake") == 1 and normalise_label("real") == 0
    with pytest.raises(ValueError):
        normalise_label("satire")


def test_split_is_stratified_70_30():
    df = load_data(str(SAMPLE))
    tr, te = split(df)
    assert len(df) == 40 and len(te) == 12
    assert te["y"].sum() == 6  # 50/50 classes preserved


def test_metrics_fake_is_positive():
    res = evaluate([1, 1, 0, 0], [1, 0, 0, 0])
    assert res["precision"] == 1.0 and res["recall"] == 0.5
    assert res["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}


def test_ml_end_to_end(tmp_path):
    ml_main(["--data", str(SAMPLE), "--config", str(ROOT / "configs/ml.yaml"),
             "--models", "svm", "lr", "nb", "--out", str(tmp_path)])
    assert (tmp_path / "svm_metrics.json").exists()
    assert (tmp_path / "lr_predictions.csv").exists()
