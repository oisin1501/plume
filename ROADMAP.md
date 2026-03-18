# Plume — Roadmap to v1 Release

## Current State

What's built and working:
- [x] Tauri + React + Rust project scaffold
- [x] CSV import with auto-separator detection (comma, tab, pipe, semicolon)
- [x] Parquet import support
- [x] Data table view with pagination and column profiling
- [x] Column distribution charts (histogram for numeric, frequency for categorical)
- [x] First/Last page navigation and horizontal column scroll buttons
- [x] Column sorting (click header to sort asc/desc)
- [x] Column filtering (search/filter rows by value in profile dropdown)
- [x] Shape screen: column overview, fill missing, drop columns, cast types, undo
- [x] "Fill with most common" strategy for categorical columns
- [x] One-hot encoding action for categorical columns
- [x] Rename column action
- [x] Data type change recommendations with batch "Convert all" action
- [x] Model screen: task selection (classification/regression/clustering), target/feature selection, algorithm picker, training via Python sidecar
- [x] Target column recommendations based on task type (dismissible)
- [x] Feature validation — warns on ID-like high-cardinality columns
- [x] Hyperparameter tuning UI per algorithm (trees, depth, learning rate, etc.)
- [x] Cross-validation option with configurable folds and per-fold score visualization
- [x] Model comparison — "Compare all" trains every algorithm for the task and shows results side by side
- [x] Comparison summary table (Compare/Details tabs) with best-model highlighting, all metrics, CV scores, and feature counts
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
- [x] Drag-and-drop via Tauri native event (with file type validation)
- [x] Empty/invalid CSV validation (rejects empty files, no-column files)
- [x] Graceful training failure handling (in-UI errors, partial compare-all support, input validation)
- [x] Python training input validation (min rows, min classes, feature check)
- [x] Dark mode: plume brand colors, selection colors, and surface contrast
- [x] Pipeline visualization — connected-dot timeline in Shape tab showing transform history

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

- [ ] Support Excel (.xlsx) import
- [ ] Time-series forecasting (v2 feature)
- [ ] Auto-save projects to file-based project folders
- [ ] Data visualization tab (histograms, scatter plots, correlation heatmap)
- [ ] Feature engineering (create derived columns with expressions)
- [ ] AutoML mode ("Let Plume decide" — runs all algorithms, picks best)
- [ ] Export model as ONNX for production deployment
- [ ] Multi-language support
- [ ] Plugin system for custom algorithms
- [ ] Cloud sync / team collaboration (v3+)
- [ ] App-level AES-256 encryption for sensitive datasets
