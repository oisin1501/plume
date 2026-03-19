# Plume — Roadmap to v1 Release

## Current State

What's built and working:
- [x] Tauri + React + Rust project scaffold
- [x] CSV import with auto-separator detection (comma, tab, pipe, semicolon)
- [x] Parquet import support
- [x] Data table view with pagination and column profiling
- [x] Column distribution charts (histogram for numeric, frequency for categorical) with hover showing count and % of total
- [x] First/Last page navigation and horizontal column scroll buttons
- [x] Column sorting (click header to sort asc/desc)
- [x] Column filtering (search/filter rows by value in profile dropdown)
- [x] Shape screen: column overview, fill missing, drop columns, cast types, undo
- [x] "Fill with most common" strategy for categorical columns
- [x] One-hot encoding action for categorical columns
- [x] Rename column action
- [x] Data type change recommendations with batch "Convert all" action
- [x] Expandable type recommendations banner — click to see individual columns, convert one at a time or all at once
- [x] Model screen: task selection (classification/regression/clustering), target/feature selection, algorithm picker, training via Python sidecar
- [x] Target column recommendations based on task type (dismissible)
- [x] Feature validation — warns on ID-like high-cardinality columns
- [x] Hyperparameter tuning UI per algorithm (trees, depth, learning rate, etc.)
- [x] Cross-validation option with configurable folds and per-fold score visualization
- [x] Plain-English explanation for cross-validation option (what it does, why it helps)
- [x] Type filter chips in Model tab — filter target and feature columns by data type (Text, Decimal, Integer, Boolean)
- [x] Model comparison — "Compare all" trains every algorithm for the task and shows results side by side
- [x] Comparison summary table (Compare/Details tabs) with best-model highlighting, all metrics, CV scores, and feature counts
- [x] Shared feature importance on Compare screen — shows features agreed upon across models with average importance
- [x] Results screen: headline metric, feature importance bars, confusion matrix, cluster summaries
- [x] Multiple training results stored and selectable for comparison
- [x] ROC curve visualization for classification (with AUC score)
- [x] Combined ROC curve overlay on comparison page with color-coded legend and AUC scores
- [x] Residual plot for regression (predicted vs residual scatter)
- [x] Cluster scatter plot via PCA projection to 2D (color-coded by group)
- [x] Export predictions as CSV (save dialog)
- [x] Export model as pickle file (.pkl)
- [x] Generate summary report (HTML) with metrics, feature importance, and comparison table
- [x] SHAP values for individual predictions (explains why the model predicted X)
- [x] Loading indicator (shimmer bar)
- [x] App icon (Simurgh bird)
- [x] Load new file after one is already loaded (full state reset)
- [x] Python sidecar resolves correct python3 path in bundled .app
- [x] Drag-and-drop via Tauri native event (with file type validation) — works app-wide, even with a dataset already loaded
- [x] Empty/invalid CSV validation (rejects empty files, no-column files)
- [x] Graceful training failure handling (in-UI errors, partial compare-all support, input validation)
- [x] Python training input validation (min rows, min classes, feature check)
- [x] Dark mode: plume brand colors, selection colors, and surface contrast
- [x] Pipeline visualization — connected-dot timeline in Shape tab showing transform history
- [x] Visualize tab with four chart types:
  - [x] Histogram — configurable bin count, numeric column selector
  - [x] Scatter plot — X/Y numeric column selectors, sampled to 2000 points
  - [x] Correlation heatmap — auto-computed Pearson correlation for all numeric columns
  - [x] Box plot — numeric column grouped by any column, with whiskers, IQR, and outliers
- [x] Feature engineering panel in Shape tab:
  - [x] Combine columns — arithmetic between two numeric columns (add, subtract, multiply, divide)
  - [x] Transform column — log, log10, sqrt, square, abs, standardize (z-score), normalize (0-1)
  - [x] Bin column — equal-width binning into N groups (2-20 bins)
- [x] Bundled Python runtime — PyInstaller onedir bundle includes Python 3.13 + numpy + pandas + scikit-learn + xgboost + lightgbm + shap. No user-installed Python required.
- [x] Excel (.xlsx/.xls) import via calamine crate
- [x] Plain-language result explanations:
  - [x] Auto-generated summary with accuracy/R²/cluster count and top features
  - [x] Color-coded quality assessment (strong/reasonable/struggling)
  - [x] Metric tooltips — styled popover on hover/click with plain-English definition for every metric (Accuracy, Precision, Recall, F1, R², MAE, RMSE, Silhouette, etc.)
  - [x] Contextual explanations for feature importance, confusion matrix, and ROC curves
  - [x] Overfitting detection — warns when training accuracy >> test accuracy with plain-English explanation and suggestions
  - [x] SHAP "What does this mean?" — plain-English explanation per prediction card showing which features pushed higher/lower
- [x] Tune & Retrain from Results tab — edit hyperparameters and retrain without leaving the results view, new result added for comparison
- [x] Resizable charts — S/M/L toggle on all charts (Visualize tab + Results tab) for histogram, scatter, box plot, ROC, residuals, cluster scatter
- [x] Branded title bar — "Plume · filename.csv · N rows" with contextual info
- [x] Contextual suggestions in Shape tab — actionable hints for missing values, low cardinality, one-hot candidates, constant columns, and high-cardinality text. Dismissible per column.
- [x] Smart defaults — auto-select all non-ID features when a target is chosen; auto-select for clustering

---

## Bug Fixes & Stability (completed)

- [x] **Fixed: Pickle export and SHAP passed null target** — `PickleExportButton` and `ShapButton` now use `result.target` and `result.hyperparams` from the training result instead of hardcoded null values
- [x] **Fixed: Training used stale file** — `train_model`, `export_model_pickle`, and `compute_shap` now export the current in-memory DataFrame to a temp CSV before sending to Python, so feature-engineered columns and all transforms are included
- [x] **Fixed: Mutex poison cascading** — All `Mutex::lock().unwrap()` calls replaced with `.unwrap_or_else(|p| p.into_inner())` to recover from poison after any panic, preventing all-commands-broken state
- [x] **Fixed: DataTable missing error handling** — `loadPage()` now wraps `invoke` in try/catch
- [x] **Fixed: Python all-null column crash** — `prepare_features` now falls back to 0 when median is NaN (all-null numeric columns)
- [x] **Fixed: NaN counted as classification class** — `n_classes` count now filters out NaN values
- [x] **Fixed: Missing library errors** — XGBoost and LightGBM imports now give user-friendly error messages instead of raw tracebacks
- [x] **Removed dead code** — `detect_separator_pub` removed (no longer needed after temp CSV export change)

---

## Must Test Before Publishing

### Functional Testing
- [ ] End-to-end: import CSV → shape data → train classification model → view results
- [ ] End-to-end: import CSV → train regression model → view results
- [ ] End-to-end: import CSV → train clustering model → view cluster summaries
- [ ] Undo/redo across multiple transform steps
- [ ] Large dataset performance (100K+ rows, 50+ columns)
- [ ] Files with special characters in path or column names
- [ ] Files with mixed encodings (UTF-8, Latin-1)
- [ ] All four algorithms per task type (12 combinations)

### UI/UX Testing
- [ ] Dark mode across all screens
- [ ] Window resize behavior at minimum size (900x600)
- [ ] Sidebar navigation state preservation (going back to Data tab after training)
- [ ] All animations smooth at 60fps
- [ ] Font rendering on Retina and non-Retina displays
- [ ] Keyboard accessibility (tab navigation, enter to confirm)

### Platform Testing
- [x] macOS production build (`npx @tauri-apps/cli build`)
- [ ] macOS code signing and notarization
- [ ] Windows build (cross-compile or on a Windows machine)
- [ ] Windows-specific UI adjustments (title bar, file paths)

---

## Nice to Have (Post-v1)

- [x] ~~Support Excel (.xlsx) import~~ — moved to v1
- [ ] Time-series forecasting (v2 feature)
- [ ] Auto-save projects to file-based project folders
- [x] ~~Data visualization tab (histograms, scatter plots, correlation heatmap)~~ — moved to v1
- [x] ~~Feature engineering (create derived columns with expressions)~~ — moved to v1
- [ ] AutoML mode ("Let Plume decide" — runs all algorithms, picks best)
- [ ] Export model as ONNX for production deployment
- [ ] Multi-language support
- [ ] Plugin system for custom algorithms
- [ ] Cloud sync / team collaboration (v3+)
- [ ] App-level AES-256 encryption for sensitive datasets
