# Data

The study uses the **COVID Fake News Dataset** from Kaggle (uploaded by *csmalarkodi*, retrieved 28 August 2023): 6,420 English posts, 3,360 labelled `real` and 3,060 `fake`. It is not redistributed here. Download it from Kaggle and follow its licence.

## Expected format

A single CSV, e.g. `data/covid_fake_news.csv`:

| Column | Content |
|---|---|
| `id` | Row ID (optional) |
| `tweet` | Post text (`text`, `content` or `news` are also detected) |
| `label` | `real` or `fake` (also `0`/`1`) |

The scripts make a stratified 70/30 train/test split themselves (seed 42). Use `--text-col` / `--label-col` for other column names.

## Sample

[`sample.csv`](sample.csv) holds 40 **invented** posts in this format, so the code and tests run without the real data. They are not from the Kaggle dataset.
