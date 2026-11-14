# WIE3007 Group Project

Synthetic Malaysian retail-banking dataset (1,500 customers) with numeric and text data, and AI-extracted features (sentiment, topic, text-risk category, customer segment). Target: `default_flag` (about 13% positive).

```
data/                          Member A. Already built. Do not edit.
src/generate_dataset.py        Member A
src/extract_features.py        Member A
src/extract_features_hf.py     Member A (Hugging Face back-end, needs internet)
src/common.py                  Shared. Load, frozen split, preprocessor, metrics. Do not edit once agreed.
notebooks/01_dataset_feature_eng.ipynb          Member A
notebooks/02_baseline_decision_tree.ipynb       Member B
notebooks/03_random_forest.ipynb                Member B
notebooks/04_xgboost.ipynb                      Member C
notebooks/05_logistic_regression.ipynb          Member D
notebooks/06_mlp.ipynb                          Member D
notebooks/07_comparison_and_interpretation.ipynb  written together at the end
results/                       one JSON per model, e.g. results/rf_metrics.json
report/
```

Modelling uses `data/bank_customers_enriched.csv` only. `data/ground_truth_text_labels.csv` is for checking the text extractors. It is not a model feature. Raw `customer_feedback` and `agent_note` are not model features either. `src/common.py` also leaves out `sentiment_label`, because `sentiment_score` already carries it.

## Setup

```bash
pip install -r requirements.txt
python src/common.py
```

`python src/common.py` prints the frozen split (seed 42, stratified 80/20). Every model notebook calls `common.train_test_data()` and `common.save_metrics(...)`. Notebook 07 only reads the JSON files in `results/`.

## Branches

Each member works on their own branch and only commits their notebook plus their metrics file.

| Branch | Notebook | Metrics file |
| --- | --- | --- |
| `member-b/decision-tree` | `notebooks/02_baseline_decision_tree.ipynb` | `results/dt_metrics.json` |
| `member-b/random-forest` | `notebooks/03_random_forest.ipynb` | `results/rf_metrics.json` |
| `member-c/xgboost` | `notebooks/04_xgboost.ipynb` | `results/xgb_metrics.json` |
| `member-d/logistic-regression` | `notebooks/05_logistic_regression.ipynb` | `results/logreg_metrics.json` |
| `member-d/mlp` | `notebooks/06_mlp.ipynb` | `results/mlp_metrics.json` |

```bash
git checkout main
git pull
git checkout -b member-b/random-forest
```

If that branch already exists locally, use `git checkout member-b/random-forest` instead of `-b`.

Clear notebook output before committing. A metrics file is small and should be committed. Another member's notebook should not be edited on your branch.

## Component 1 data

Run `python src/generate_dataset.py` only if you need to rebuild the raw file. The enriched file checked in here is the one the models train on. All of the data is synthetic.
