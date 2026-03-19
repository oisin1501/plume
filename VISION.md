# Plume — Vision & Path Forward

## Mission

Make machine learning accessible to everyone. No code. No cost. No complexity.

Plume exists because powerful analysis shouldn't require a CS degree or an enterprise license. If you have data and a question, you should be able to get an answer.

---

## Where We Are (v1.0.0)

A working desktop app that covers the full ML pipeline:
- Import data (CSV, Parquet, Excel)
- Explore and clean it (profiling, type casting, missing values, feature engineering)
- Visualize it (histogram, scatter, correlation, box plot — all resizable)
- Train models (classification, regression, clustering across 12+ algorithms)
- Evaluate results (metrics with plain-English tooltips, ROC curves, residual plots, SHAP, model comparison, overfitting detection)
- Export (predictions, models, reports)

### Honest assessment of v1

**What works well:**
- The Data → Shape → Visualize → Model → Results flow is intuitive
- Desktop-first means no accounts, no cloud, data stays local
- Lightweight — the app is fast and focused

**What's holding us back from the mission:**
1. Requires Python + packages installed separately — a dealbreaker for non-technical users
2. The app helps you *operate* ML, but doesn't help you *understand* it
3. No guidance on whether results are good, bad, or meaningful
4. README and public presence don't communicate what Plume is
5. macOS-only, unsigned — limits who can use it

---

## Where We Want To Be

An app that a small business owner, a teacher, a student, or a nurse can download, open, load a spreadsheet, and get meaningful insights — without ever feeling lost or confused.

---

## How We Get There

### Phase 1: Remove the Python barrier ✓
**Goal:** Zero-dependency installation. Download, open, use.

- [x] Bundle Python runtime via PyInstaller (onedir mode, ~4s cold start)
- [x] Includes Python 3.13 + numpy + pandas + scikit-learn + xgboost + lightgbm + shap
- [x] Falls back to system Python in development mode
- [ ] Consider long-term: rewrite ML in Rust using `linfa` or `smartcore` to eliminate Python entirely and reduce app size from 1.4GB

**Success criteria:** ✓ A user with no developer tools installed can download Plume and train a model.

### Phase 2: Plain-language explanations ✓
**Goal:** Users understand what happened, not just what buttons to click.

- [x] Contextual tooltips on every metric (hover "?" for explanation)
- [x] Auto-generated plain-English summary after training with top features
- [x] Color-coded quality assessment (strong/reasonable/struggling)
- [x] Contextual explanations for feature importance, confusion matrix, and ROC curves
- [x] Add "What does this mean?" expandable sections for SHAP values
- [x] Contextual suggestions in Shape tab: "This column has 40% missing values — consider dropping it or filling with the average"
- [x] Overfitting detection: warn when training accuracy >> test accuracy
- [x] Styled metric tooltips — popover on hover/click with plain-English definitions
- [x] Plain-English cross-validation explanation in Model tab

**Success criteria:** ✓ Met. A non-technical user can understand their results in plain English, gets contextual guidance on data quality, and is warned about overfitting.

### Phase 3: Guided experience
**Goal:** The app guides first-time users through the workflow.

- [ ] First-launch onboarding: brief walkthrough of the 5 tabs and what they do
- [ ] Smart defaults: auto-select the most likely target column, pre-check recommended features
- [ ] "Quick Start" mode: load data → auto-detect task type → train best model → show results (one-click ML)
- [ ] Warnings that prevent bad decisions: "You're about to train on an ID column — this won't produce useful results"
- [ ] Progress indicator during training with estimated time

**Success criteria:** A first-time user can go from opening the app to seeing results without reading documentation.

### Phase 4: Cross-platform & distribution
**Goal:** Anyone on any OS can use Plume.

- [ ] GitHub Actions CI: auto-build macOS (.dmg) and Windows (.msi) on tag push
- [ ] macOS code signing and notarization (requires Apple Developer account, $99/year)
- [ ] Windows code signing (optional but reduces SmartScreen warnings)
- [ ] Linux build (.AppImage or .deb)
- [ ] Landing page / website explaining what Plume is (could be a simple GitHub Pages site)

**Success criteria:** Users on macOS, Windows, and Linux can install Plume without terminal commands or security bypass steps.

### Phase 5: Open-source community
**Goal:** Make Plume a real open-source project others can contribute to.

- [ ] Write a proper README: what Plume is, screenshots, installation, quick start
- [ ] Add a LICENSE file (MIT or Apache 2.0)
- [ ] Add CONTRIBUTING.md with development setup instructions
- [ ] Create GitHub issue templates for bugs and feature requests
- [ ] Add a code of conduct
- [ ] Create a few "good first issue" tickets to attract contributors

**Success criteria:** Someone finds Plume on GitHub, understands what it is in 30 seconds, and knows how to contribute.

### Phase 6: Excel import & quality of life
**Goal:** Meet users where they are — most non-technical people use Excel.

- [x] Support .xlsx and .xls import
- [ ] Auto-save and restore sessions (project files)
- [ ] Undo across the entire session (not just within Shape tab)
- [ ] Keyboard shortcuts for common actions
- [ ] Drag-and-drop columns to reorder
- [ ] Copy/paste data directly into the app

**Success criteria:** Partially met. Excel import works. Session persistence and keyboard shortcuts remain.

### Phase 7: Advanced (post-v2)
**Goal:** Grow capabilities without growing complexity.

- [ ] Time-series forecasting
- [ ] AutoML mode ("Let Plume decide")
- [ ] Data visualization tab improvements (pair plots, correlation filtering)
- [ ] Export model as ONNX for production deployment
- [ ] Natural language interface: "Predict sales based on marketing spend and season"

---

## Guiding Principles

1. **If a non-technical user wouldn't understand it, simplify it or explain it.** Don't add features for power users at the expense of clarity.
2. **Lightweight over feature-rich.** Every feature must earn its place. Three focused tools beat ten confusing ones.
3. **Local-first.** Data never leaves the user's machine. No accounts, no cloud, no tracking.
4. **Free forever.** Plume is open-source. The mission fails if it costs money.
5. **Guide, don't gatekeep.** The app should make users more capable, not more dependent on the app.
