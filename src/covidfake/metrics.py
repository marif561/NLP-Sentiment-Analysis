"""Binary metrics with 'fake' as the positive class, plus macro/weighted averages."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def summary_line(res: Dict) -> str:
    return (f"acc {res['accuracy']:.4f} · P {res['precision']:.4f} · R {res['recall']:.4f} · "
            f"F1 {res['f1']:.4f} · macro-F1 {res['macro_f1']:.4f}")


def save(res: Dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
