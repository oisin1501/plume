"""
Pytest plugin that collects benchmark results from Plume's validation suite
and generates a visual HTML report at ``python/validation_report.html``.

Tests store results via the ``benchmark`` fixture (a callable).
The HTML is written in ``pytest_sessionfinish``.
"""

import os
import datetime
import pytest

# Shared across the entire session — tests append dicts here.
_results: list[dict] = []


@pytest.fixture
def benchmark():
    """Fixture that returns a callable for recording a benchmark result.

    Usage inside a test::

        benchmark(
            dataset="Iris",
            algorithm="Random Forest",
            task="classification",
            metrics={"accuracy": 0.97, "f1": 0.96},
            thresholds={"accuracy": 0.90, "f1": 0.88},
        )
    """

    def _record(*, dataset, algorithm, task, metrics, thresholds, extra=None):
        passed = all(metrics[k] >= thresholds[k] for k in thresholds)
        _results.append({
            "dataset": dataset,
            "algorithm": algorithm,
            "task": task,
            "metrics": metrics,
            "thresholds": thresholds,
            "passed": passed,
            "extra": extra,
        })

    return _record


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session, exitstatus):
    if not _results:
        return

    total = len(_results)
    passed = sum(1 for r in _results if r["passed"])
    failed = total - passed
    now = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

    # Group results by category
    classification = [r for r in _results if r["task"] == "classification"]
    regression = [r for r in _results if r["task"] == "regression"]
    cv_results = [r for r in _results if r["task"] == "cross-validation"]
    guardrails = [r for r in _results if r["task"] == "guardrail"]

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Plume ML — Validation Report</title>
<style>
  :root {{
    --pass: #10b981;
    --pass-bg: #ecfdf5;
    --fail: #ef4444;
    --fail-bg: #fef2f2;
    --border: #e5e7eb;
    --text: #1f2937;
    --muted: #6b7280;
    --bg: #f9fafb;
    --white: #ffffff;
    --accent: #6366f1;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}

  /* Header */
  .header {{
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: var(--white);
    border-radius: 12px;
    border: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
  }}
  .header h1 span {{ color: var(--accent); }}
  .header p {{ color: var(--muted); font-size: 0.95rem; }}

  /* Summary cards */
  .summary {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .card {{
    background: var(--white);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
    border: 1px solid var(--border);
  }}
  .card .number {{
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
  }}
  .card .label {{
    font-size: 0.85rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .card.pass .number {{ color: var(--pass); }}
  .card.fail .number {{ color: var(--fail); }}
  .card.total .number {{ color: var(--accent); }}

  /* Section */
  .section {{
    background: var(--white);
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
    overflow: hidden;
  }}
  .section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .section-title .icon {{ font-size: 1.2rem; }}

  /* Table */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th {{
    text-align: left;
    padding: 0.65rem 1.25rem;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    background: var(--bg);
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 0.7rem 1.25rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}

  /* Metric cells */
  .metric-cell {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .metric-value {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
  .threshold {{
    font-size: 0.78rem;
    color: var(--muted);
  }}
  .criteria-cell {{
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    color: var(--muted);
  }}

  /* Bar */
  .bar-track {{
    width: 100%;
    max-width: 140px;
    height: 8px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
  }}
  .bar-fill.pass {{ background: var(--pass); }}
  .bar-fill.fail {{ background: var(--fail); }}

  /* Threshold marker on bar */
  .bar-marker {{
    position: absolute;
    top: -2px;
    width: 2px;
    height: 12px;
    background: var(--text);
    border-radius: 1px;
    opacity: 0.4;
  }}

  /* Badge */
  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
  }}
  .badge.pass {{ background: var(--pass-bg); color: #047857; }}
  .badge.fail {{ background: var(--fail-bg); color: #b91c1c; }}
  .badge-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }}
  .badge.pass .badge-dot {{ background: var(--pass); }}
  .badge.fail .badge-dot {{ background: var(--fail); }}

  /* Algo name */
  .algo-name {{
    font-weight: 500;
  }}
  .dataset-name {{
    color: var(--muted);
    font-size: 0.82rem;
  }}

  /* Guardrail row */
  .guardrail-desc {{
    font-weight: 500;
  }}

  /* Footer */
  .footer {{
    text-align: center;
    padding: 1.5rem;
    color: var(--muted);
    font-size: 0.82rem;
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1><span>Plume</span> ML Validation Report</h1>
    <p>Generated {now} &mdash; proving Plume's ML pipeline produces correct results on gold-standard datasets</p>
  </div>

  <div class="summary">
    <div class="card total">
      <div class="number">{total}</div>
      <div class="label">Total Checks</div>
    </div>
    <div class="card pass">
      <div class="number">{passed}</div>
      <div class="label">Passed</div>
    </div>
    <div class="card {"pass" if failed == 0 else "fail"}">
      <div class="number">{failed}</div>
      <div class="label">Failed</div>
    </div>
  </div>
"""

    # --- Classification section ---
    if classification:
        html += _section("Classification Benchmarks", classification, is_classification=True)

    # --- Regression section ---
    if regression:
        html += _section("Regression Benchmarks", regression, is_classification=False)

    # --- Cross-validation section ---
    if cv_results:
        html += _cv_section(cv_results)

    # --- Guardrails section ---
    if guardrails:
        html += _guardrail_section(guardrails)

    html += """
  <div class="footer">
    Plume Validation Suite &mdash; datasets sourced from scikit-learn &mdash; thresholds are intentionally conservative
  </div>

</div>
</body>
</html>
"""

    out_path = os.path.join(os.path.dirname(__file__), "validation_report.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\n\n  Validation report written to: {out_path}\n")


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_ALGO_LABELS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "linear_regression": "Linear Regression",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def _algo_label(name):
    return _ALGO_LABELS.get(name, name)


def _bar_html(value, threshold, max_val=1.0):
    """Render a small bar with a threshold marker."""
    pct = min(value / max_val * 100, 100) if max_val else 0
    marker_pct = min(threshold / max_val * 100, 100) if max_val else 0
    status = "pass" if value >= threshold else "fail"
    return (
        f'<div class="bar-track">'
        f'<div class="bar-fill {status}" style="width:{pct:.1f}%"></div>'
        f'<div class="bar-marker" style="left:{marker_pct:.1f}%"></div>'
        f'</div>'
    )


def _badge(passed):
    cls = "pass" if passed else "fail"
    label = "PASS" if passed else "FAIL"
    return f'<span class="badge {cls}"><span class="badge-dot"></span>{label}</span>'


def _section(title, results, is_classification):
    icon = "&#x1f3af;" if is_classification else "&#x1f4c8;"
    metric_keys = ["accuracy", "f1"] if is_classification else ["r2", "mae"]
    metric_labels = {
        "accuracy": "Accuracy",
        "f1": "F1 Score",
        "r2": "R\u00b2",
        "mae": "MAE",
    }
    criteria_labels = {
        "accuracy": "&ge;",
        "f1": "&ge;",
        "r2": "&ge;",
        "mae": "&gt;",
    }

    rows = ""
    for r in results:
        algo = _algo_label(r["algorithm"])
        ds = r["dataset"]
        status = _badge(r["passed"])

        cells = ""
        for k in metric_keys:
            val = r["metrics"].get(k)
            thr = r["thresholds"].get(k)
            if val is None:
                cells += "<td>&mdash;</td><td>&mdash;</td>"
                continue

            op = criteria_labels[k]

            # Acceptance criteria column
            cells += f'<td class="criteria-cell">{op} {thr}</td>'

            # Actual value column with bar
            if k == "mae":
                cells += f'<td><span class="metric-value">{val:.4f}</span></td>'
            else:
                max_val = 1.0
                cells += (
                    f'<td>'
                    f'<span class="metric-value">{val:.4f}</span>'
                    f'{_bar_html(val, thr, max_val)}'
                    f'</td>'
                )

        rows += (
            f"<tr>"
            f'<td><span class="algo-name">{algo}</span><br>'
            f'<span class="dataset-name">{ds}</span></td>'
            f"{cells}"
            f"<td>{status}</td>"
            f"</tr>\n"
        )

    # Build column headers: for each metric, add "Criteria" + "Actual" pair
    col_headers = ""
    for k in metric_keys:
        label = metric_labels[k]
        col_headers += f"<th>{label} Criteria</th><th>{label} Actual</th>"

    return (
        f'<div class="section">'
        f'<div class="section-title"><span class="icon">{icon}</span> {title}</div>'
        f"<table>"
        f"<tr><th>Algorithm / Dataset</th>{col_headers}<th>Result</th></tr>"
        f"{rows}"
        f"</table></div>\n"
    )


def _cv_section(results):
    rows = ""
    for r in results:
        extra = r.get("extra", {})
        folds = extra.get("folds", "?")
        scores = extra.get("scores", [])
        mean = extra.get("mean", 0)
        std = extra.get("std", 0)
        ds = r["dataset"]
        algo = _algo_label(r["algorithm"])

        fold_cells = " ".join(
            f'<span class="metric-value">{s:.4f}</span>' for s in scores
        )
        thr_mean = r["thresholds"].get("mean", 0)
        rows += (
            f"<tr>"
            f'<td><span class="algo-name">{algo}</span><br>'
            f'<span class="dataset-name">{ds}</span></td>'
            f"<td>{folds}</td>"
            f"<td>{fold_cells}</td>"
            f'<td class="criteria-cell">&ge; {thr_mean}</td>'
            f'<td><span class="metric-value">{mean:.4f}</span>'
            f' <span class="threshold">(&plusmn; {std:.4f})</span></td>'
            f"<td>{_badge(r['passed'])}</td>"
            f"</tr>\n"
        )

    return (
        f'<div class="section">'
        f'<div class="section-title"><span class="icon">&#x1f504;</span> Cross-Validation</div>'
        f"<table>"
        f"<tr><th>Algorithm / Dataset</th><th>Folds</th><th>Per-Fold Scores</th><th>Mean Criteria</th><th>Mean Actual</th><th>Result</th></tr>"
        f"{rows}"
        f"</table></div>\n"
    )


def _guardrail_section(results):
    rows = ""
    for r in results:
        desc = r["dataset"]  # we store description in dataset field
        extra = r.get("extra", {})
        detail = extra.get("detail", "")
        rows += (
            f"<tr>"
            f'<td><span class="guardrail-desc">{desc}</span></td>'
            f"<td>{detail}</td>"
            f"<td>{_badge(r['passed'])}</td>"
            f"</tr>\n"
        )

    return (
        f'<div class="section">'
        f'<div class="section-title"><span class="icon">&#x1f6e1;</span> Safety Guardrails</div>'
        f"<table>"
        f"<tr><th>Check</th><th>Detail</th><th>Result</th></tr>"
        f"{rows}"
        f"</table></div>\n"
    )
