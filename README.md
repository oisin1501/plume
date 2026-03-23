# Plume

**No-code machine learning for everyone.**

Plume is a free, open-source desktop app that lets anyone build, train, and evaluate machine learning models — without writing a single line of code. No Python. No setup. Just download, open, and start.

## Why Plume?

Most ML tools assume you already know what you're doing. They're either expensive enterprise software (JMP, SPSS, SAS) or require coding skills (Python, R). That leaves millions of people — analysts, students, researchers, small business owners — locked out of powerful analysis because they "don't know how to code."

Plume changes that. It's lightweight, intuitive, and explains everything in plain language.

## What you can do

**Import your data**
Load CSV, Excel (.xlsx), or Parquet files. Drag and drop or browse. Plume auto-detects separators, column types, and data issues.

**Explore and clean it**
See column profiles, distributions, and missing values at a glance. Fill gaps, cast types, rename or drop columns, one-hot encode categoricals — all with one click. Undo anything.

**Engineer features**
Combine columns (add, subtract, multiply, divide), apply transforms (log, sqrt, z-score, normalize), or bin numeric columns into groups. No formulas to write.

**Visualize patterns**
Histograms, scatter plots, correlation heatmaps, and box plots — pick your columns and see the chart instantly.

**Train models**
Choose a task (classify, predict a number, or find groups), pick a target, select features, and hit Train. Plume supports Random Forest, XGBoost, LightGBM, Logistic/Linear Regression, K-Means, and DBSCAN. Tune hyperparameters with sliders. Run cross-validation. Compare all algorithms side by side.

**Understand results**
Every result comes with a plain-English summary: *"Your model correctly identifies 87% of cases. The most important factors are age, income, and region."* Metrics have hover explanations. Quality is color-coded (strong / reasonable / needs work). SHAP values explain individual predictions.

**Export everything**
Save predictions as CSV, export the trained model as a pickle file, or generate an HTML summary report.

## Install

### macOS (Apple Silicon)
1. Download `Plume-v1.0.0-macOS-Apple-Silicon.dmg` from [Releases](https://github.com/oisin1501/plume/releases)
2. Open the DMG, drag Plume to Applications
3. First launch: right-click Plume.app, click **Open**, then **Open** again (one-time Gatekeeper bypass — the app is not code-signed yet)

### Windows
1. Download the `.msi` installer from [Releases](https://github.com/oisin1501/plume/releases)
2. Run the installer

**No Python or packages needed** — everything is bundled inside the app.

## Validation

Plume's ML pipeline is validated against gold-standard datasets from scikit-learn. Every algorithm is trained through the exact same code path the app uses and must meet conservative performance floors — if any check fails, something is broken in preprocessing, encoding, or model fitting.

### Classification

| Dataset | Algorithm | Accuracy Criteria | Accuracy Actual | F1 Criteria | F1 Actual | Result |
|---------|-----------|:-----------------:|:---------------:|:-----------:|:---------:|:------:|
| Iris (150 rows, 3 classes) | Random Forest | ≥ 0.90 | 1.0000 | ≥ 0.88 | 1.0000 | PASS |
| Iris (150 rows, 3 classes) | Logistic Regression | ≥ 0.90 | 1.0000 | ≥ 0.88 | 1.0000 | PASS |
| Breast Cancer (569 rows, 2 classes) | Random Forest | ≥ 0.92 | 0.9649 | ≥ 0.90 | 0.9722 | PASS |
| Breast Cancer (569 rows, 2 classes) | Logistic Regression | ≥ 0.92 | 0.9561 | ≥ 0.90 | 0.9655 | PASS |
| Breast Cancer (569 rows, 2 classes) | XGBoost | ≥ 0.92 | 0.9561 | ≥ 0.90 | 0.9650 | PASS |
| Breast Cancer (569 rows, 2 classes) | LightGBM | ≥ 0.92 | 0.9649 | ≥ 0.90 | 0.9722 | PASS |
| Wine (178 rows, 3 classes) | Random Forest | ≥ 0.85 | 1.0000 | ≥ 0.83 | 1.0000 | PASS |
| Wine (178 rows, 3 classes) | XGBoost | ≥ 0.85 | 0.9722 | ≥ 0.83 | 0.9718 | PASS |
| Wine (178 rows, 3 classes) | LightGBM | ≥ 0.85 | 1.0000 | ≥ 0.83 | 1.0000 | PASS |

### Regression

| Dataset | Algorithm | R² Criteria | R² Actual | Result |
|---------|-----------|:-----------:|:---------:|:------:|
| Diabetes (442 rows) | Random Forest | ≥ 0.30 | 0.4428 | PASS |
| Diabetes (442 rows) | Linear Regression | ≥ 0.30 | 0.4526 | PASS |
| Diabetes (442 rows) | XGBoost | ≥ 0.30 | 0.3595 | PASS |
| Diabetes (442 rows) | LightGBM | ≥ 0.30 | 0.3959 | PASS |
| Synthetic (1000 rows) | Random Forest | ≥ 0.75 | 0.9337 | PASS |
| Synthetic (1000 rows) | Linear Regression | ≥ 0.75 | 0.9814 | PASS |
| Synthetic (1000 rows) | XGBoost | ≥ 0.75 | 0.9452 | PASS |
| Synthetic (1000 rows) | LightGBM | ≥ 0.75 | 0.9516 | PASS |

### Safety guardrails

| Check | Detail | Result |
|-------|--------|:------:|
| Reject datasets with < 2 rows | Returns error for 1-row dataset | PASS |
| Reject single-class targets | Returns error when target has only 1 unique value | PASS |
| Detect data leakage | Flags feature with > 0.95 correlation to target | PASS |
| Handle categorical features | Encodes string columns and trains successfully | PASS |
| Impute missing values | Imputes 10% NaN values and trains successfully | PASS |

> **23 / 23 checks passed.** Run `cd python && pytest test_validation.py -v` to verify, or open [`python/validation_report.html`](python/validation_report.html) for the full visual report.

## Built with

- [Tauri](https://tauri.app) — lightweight desktop framework (Rust backend, web frontend)
- [React](https://react.dev) + [TypeScript](https://www.typescriptlang.org) — UI
- [Polars](https://pola.rs) — high-performance data processing in Rust
- [scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.ai), [LightGBM](https://lightgbm.readthedocs.io) — ML algorithms
- [Recharts](https://recharts.org) — visualizations
- [Tailwind CSS](https://tailwindcss.com) — styling

## Development

```bash
# Prerequisites: Node.js 20+, Rust, Python 3.12+

# Install dependencies
npm install
pip install numpy pandas scikit-learn xgboost lightgbm shap

# Run in development mode
npx @tauri-apps/cli dev

# Build for production
cd python && pyinstaller --onedir --name plume_ml -y plume_ml.py && cd ..
npx @tauri-apps/cli build
```

## License

MIT

## Contributing

Plume is in its early days. If you're interested in contributing, open an issue or start a discussion. We'd love to hear from you.
