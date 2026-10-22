"""Classical ML on TF-IDF features: SVM, Logistic Regression, Random Forest,
Decision Tree, KNN and Multinomial Naive Bayes (paper Section 3.4)."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import LABELS, load_data, split
from .metrics import evaluate, save, summary_line

ESTIMATORS = {"svm": SVC, "lr": LogisticRegression, "rf": RandomForestClassifier,
              "dt": DecisionTreeClassifier, "knn": KNeighborsClassifier, "nb": MultinomialNB}


def build(name: str, cfg: dict):
    return make_pipeline(TfidfVectorizer(**cfg.get("tfidf", {})), ESTIMATORS[name](**(cfg.get(name) or {})))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/ml.yaml")
    ap.add_argument("--models", nargs="+", default=list(ESTIMATORS))
    ap.add_argument("--text-col", default="tweet")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/ml")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    train, test = split(load_data(args.data, args.text_col, args.label_col), cfg["test_size"], args.seed)
    for name in args.models:
        clf = build(name, cfg).fit(train["clean"], train["y"])
        pred = clf.predict(test["clean"])
        res = evaluate(test["y"], pred)
        print(f"{name:<4} {summary_line(res)}")
        out = Path(args.out)
        save(res, out / f"{name}_metrics.json")
        test.assign(prediction=[LABELS[i] for i in pred]).to_csv(out / f"{name}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
