#!/usr/bin/env bash
# Run every model from the paper on one CSV file.
# Usage: bash scripts/run_all.sh data/covid_fake_news.csv
set -euo pipefail
DATA=${1:?"path to the COVID fake news CSV required"}

echo ">> Classical ML on TF-IDF"
python -m covidfake.train_ml --data "$DATA"

echo ">> Deep learning"
for M in cnn lstm bigru; do python -m covidfake.train_dl --data "$DATA" --model "$M"; done
echo "Done. Results in outputs/"
