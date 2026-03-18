import { useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ScatterChart, Scatter, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../stores/appStore";
import type { TrainResult } from "../types/data";

const METRIC_EXPLANATIONS: Record<string, string> = {
  Accuracy: "The percentage of predictions that were correct",
  Precision: "Of all positive predictions, how many were actually correct",
  Recall: "Of all actual positives, how many did the model find",
  F1: "A balance between precision and recall (harmonic mean)",
  "R²": "How much of the variation in the data the model explains (1.0 = perfect)",
  MAE: "Average size of prediction errors (lower is better)",
  RMSE: "Average prediction error, penalizing large errors more (lower is better)",
  AUC: "How well the model distinguishes between classes (1.0 = perfect, 0.5 = random guessing)",
  Silhouette: "How well-separated the clusters are (-1 to 1, higher is better)",
  Inertia: "How compact the clusters are (lower is better)",
  Groups: "The number of distinct clusters found in your data",
};

const ALGO_LABELS: Record<string, string> = {
  random_forest: "Random Forest",
  logistic_regression: "Logistic Regression",
  linear_regression: "Linear Regression",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  kmeans: "K-Means",
  dbscan: "DBSCAN",
  hierarchical: "Hierarchical",
};

type ResultTab = "summary" | "details";

export function ResultsView() {
  const summary = useAppStore((s) => s.summary);
  const trainingResults = useAppStore((s) => s.trainingResults);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [tab, setTab] = useState<ResultTab>("summary");

  if (!summary || trainingResults.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-[13px] text-text-tertiary">
          Train a model to see results here
        </p>
      </div>
    );
  }

  const hasMultiple = trainingResults.length > 1;
  const result = trainingResults[selectedIdx] ?? trainingResults[0];
  const isClassification = result.task === "classification";
  const isRegression = result.task === "regression";
  const isClustering = result.task === "clustering";

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-[780px] mx-auto py-8 px-6">
        {/* Tab toggle for Summary vs Details */}
        {hasMultiple && (
          <div className="flex items-center gap-4 mb-6">
            <button
              onClick={() => setTab("summary")}
              className={`text-[13px] font-medium pb-1 border-b-2 transition-colors duration-200 cursor-pointer ${
                tab === "summary"
                  ? "border-plume-500 text-text-primary"
                  : "border-transparent text-text-tertiary hover:text-text-secondary"
              }`}
            >
              Compare
            </button>
            <button
              onClick={() => setTab("details")}
              className={`text-[13px] font-medium pb-1 border-b-2 transition-colors duration-200 cursor-pointer ${
                tab === "details"
                  ? "border-plume-500 text-text-primary"
                  : "border-transparent text-text-tertiary hover:text-text-secondary"
              }`}
            >
              Details
            </button>
          </div>
        )}

        {/* Comparison summary table */}
        {hasMultiple && tab === "summary" && (
          <ComparisonTable
            results={trainingResults}
            onSelect={(i) => { setSelectedIdx(i); setTab("details"); }}
          />
        )}

        {/* Individual result detail */}
        {(!hasMultiple || tab === "details") && (
          <>
            {/* Run selector */}
            {hasMultiple && (
              <div className="flex items-center gap-2 mb-6 flex-wrap">
                {trainingResults.map((r, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedIdx(i)}
                    className={`
                      px-3 py-1.5 text-[11px] rounded-[var(--radius-default)] border
                      transition-all duration-200 cursor-pointer
                      ${selectedIdx === i
                        ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
                        : "border-border text-text-tertiary hover:border-text-tertiary hover:text-text-secondary"
                      }
                    `}
                  >
                    {ALGO_LABELS[r.algorithm] ?? r.algorithm}
                    {r.task === "classification" && r.metrics.accuracy != null && (
                      <span className="ml-1.5 tabular-nums">{(r.metrics.accuracy * 100).toFixed(0)}%</span>
                    )}
                    {r.task === "regression" && r.metrics.r2 != null && (
                      <span className="ml-1.5 tabular-nums">R²{r.metrics.r2.toFixed(2)}</span>
                    )}
                  </button>
                ))}
              </div>
            )}

        <motion.div
          key={selectedIdx}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col gap-8"
        >
          {/* Headline */}
          <div>
            <p className="text-[11px] text-text-tertiary mb-1">
              {ALGO_LABELS[result.algorithm] ?? result.algorithm}
            </p>
            <h2 className="text-[20px] font-semibold text-text-primary leading-tight">
              {isClassification && `Your model correctly identifies ${Math.round((result.metrics.accuracy ?? 0) * 100)}% of cases.`}
              {isRegression && `Your model explains ${Math.round((result.metrics.r2 ?? 0) * 100)}% of the variation.`}
              {isClustering && `Plume found ${result.metrics.n_clusters} distinct groups in your data.`}
            </h2>
          </div>

          {/* Plain-English summary */}
          <div className="bg-plume-50 dark:bg-plume-500/10 rounded-[var(--radius-default)] px-4 py-3">
            <p className="text-[13px] text-text-secondary leading-relaxed">
              {isClassification && (
                <>
                  Your model can predict the target with {((result.metrics.accuracy ?? 0) * 100).toFixed(1)}% accuracy.
                  {result.feature_importance && result.feature_importance.length > 0 && (
                    <> The most important factors are {result.feature_importance.slice(0, 3).map((fi) => fi.feature).join(", ")}.</>
                  )}
                </>
              )}
              {isRegression && (
                <>
                  Your model explains {((result.metrics.r2 ?? 0) * 100).toFixed(1)}% of the variation in the target.
                  {result.feature_importance && result.feature_importance.length > 0 && (
                    <> The most important factors are {result.feature_importance.slice(0, 3).map((fi) => fi.feature).join(", ")}.</>
                  )}
                </>
              )}
              {isClustering && (
                <>
                  Your data was grouped into {result.metrics.n_clusters} clusters.
                  {result.clusters && result.clusters.length > 0 && (
                    <> The largest group has {Math.max(...result.clusters.map((c) => c.size)).toLocaleString()} items.</>
                  )}
                </>
              )}
            </p>
            {/* Quality assessment */}
            {isClassification && result.metrics.accuracy != null && (
              <p className={`text-[12px] mt-1.5 font-medium ${
                result.metrics.accuracy > 0.9
                  ? "text-emerald-600 dark:text-emerald-400"
                  : result.metrics.accuracy >= 0.7
                    ? "text-amber-600 dark:text-amber-400"
                    : "text-red-600 dark:text-red-400"
              }`}>
                {result.metrics.accuracy > 0.9
                  ? "This is a strong result."
                  : result.metrics.accuracy >= 0.7
                    ? "This is a reasonable result — there may be room for improvement."
                    : "This model is struggling — consider adding more features or trying different algorithms."}
              </p>
            )}
            {isRegression && result.metrics.r2 != null && (
              <p className={`text-[12px] mt-1.5 font-medium ${
                result.metrics.r2 > 0.8
                  ? "text-emerald-600 dark:text-emerald-400"
                  : result.metrics.r2 >= 0.5
                    ? "text-amber-600 dark:text-amber-400"
                    : "text-red-600 dark:text-red-400"
              }`}>
                {result.metrics.r2 > 0.8
                  ? "This is a strong result."
                  : result.metrics.r2 >= 0.5
                    ? "This is a reasonable result."
                    : "The model explains less than half the variation — consider different features."}
              </p>
            )}
          </div>

          {/* Metrics */}
          <div>
            <h3 className="text-[12px] text-text-secondary mb-3">Metrics</h3>
            <div className="grid grid-cols-3 gap-4">
              {isClassification && (
                <>
                  <MetricCard label="Accuracy" value={`${(result.metrics.accuracy * 100).toFixed(1)}%`} />
                  <MetricCard label="Precision" value={`${(result.metrics.precision * 100).toFixed(1)}%`} />
                  <MetricCard label="Recall" value={`${(result.metrics.recall * 100).toFixed(1)}%`} />
                </>
              )}
              {isRegression && (
                <>
                  <MetricCard label="R²" value={result.metrics.r2.toFixed(3)} />
                  <MetricCard label="MAE" value={result.metrics.mae.toFixed(3)} />
                  <MetricCard label="RMSE" value={result.metrics.rmse.toFixed(3)} />
                </>
              )}
              {isClustering && (
                <>
                  <MetricCard label="Groups" value={String(result.metrics.n_clusters)} />
                  {result.metrics.silhouette != null && (
                    <MetricCard label="Silhouette" value={result.metrics.silhouette.toFixed(3)} />
                  )}
                </>
              )}
            </div>
          </div>

          {/* Feature importance */}
          {result.feature_importance && result.feature_importance.length > 0 && (
            <div>
              <h3 className="text-[12px] text-text-secondary mb-1">What matters most</h3>
              <p className="text-[11px] text-text-tertiary mb-3">This shows which columns had the biggest impact on predictions. Taller bars = more influence.</p>
              <div className="flex flex-col gap-2">
                {result.feature_importance.slice(0, 10).map((fi, i) => {
                  const maxImp = result.feature_importance![0].importance;
                  const pct = maxImp > 0 ? (fi.importance / maxImp) * 100 : 0;
                  return (
                    <div key={fi.feature} className="flex items-center gap-3">
                      <span className="text-[12px] text-text-secondary w-[140px] text-right truncate shrink-0">
                        {fi.feature}
                      </span>
                      <div className="flex-1 h-[6px] bg-border rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${pct}%` }}
                          transition={{ duration: 0.4, delay: i * 0.05, ease: "easeOut" }}
                          className="h-full bg-plume-500 rounded-full"
                        />
                      </div>
                      <span className="text-[11px] text-text-tertiary w-[50px] tabular-nums">
                        {fi.importance.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Confusion matrix */}
          {isClassification && result.metrics.confusion_matrix && (
            <div>
              <h3 className="text-[12px] text-text-secondary mb-1">Confusion Matrix</h3>
              <p className="text-[11px] text-text-tertiary mb-3">Each cell shows how many items were predicted as one category but actually belonged to another. Diagonal cells (top-left to bottom-right) are correct predictions.</p>
              <ConfusionMatrix
                matrix={result.metrics.confusion_matrix.matrix}
                labels={result.metrics.confusion_matrix.labels}
              />
            </div>
          )}

          {/* Cluster summaries */}
          {isClustering && result.clusters && (
            <div>
              <h3 className="text-[12px] text-text-secondary mb-3">Group Details</h3>
              <div className="flex flex-col gap-3">
                {result.clusters.map((cluster) => (
                  <div
                    key={cluster.cluster}
                    className="p-4 border border-border rounded-[var(--radius-default)]"
                  >
                    <div className="flex items-baseline gap-2 mb-2">
                      <span className="text-[13px] font-medium text-text-primary">
                        Group {cluster.cluster + 1}
                      </span>
                      <span className="text-[11px] text-text-tertiary">
                        {cluster.size.toLocaleString()} items
                      </span>
                    </div>
                    <div className="flex flex-col gap-1">
                      {cluster.characteristics.map((c, i) => (
                        <span key={i} className="text-[12px] text-text-secondary">{c}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ROC Curve (classification) */}
          {isClassification && result.roc_curve && (
            <div>
              <h3 className="text-[12px] text-text-secondary mb-1">
                ROC Curve{result.roc_curve.auc != null && ` (AUC = ${result.roc_curve.auc.toFixed(3)})`}
              </h3>
              <p className="text-[11px] text-text-tertiary mb-3">The closer the curve is to the top-left corner, the better the model. The diagonal line represents random guessing.</p>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart
                  data={result.roc_curve.fpr.map((fpr, i) => ({
                    fpr,
                    tpr: result.roc_curve!.tpr[i],
                  }))}
                  margin={{ top: 5, right: 10, bottom: 20, left: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                    label={{ value: "False Positive Rate", position: "bottom", fontSize: 10, fill: "var(--color-text-tertiary)" }}
                  />
                  <YAxis
                    dataKey="tpr"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                    label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-text-tertiary)" }}
                  />
                  <ReferenceLine
                    segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                    stroke="var(--color-border)"
                    strokeDasharray="4 4"
                  />
                  <Tooltip
                    contentStyle={{ fontSize: 11, background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 6 }}
                    formatter={(v) => Number(v).toFixed(3)}
                  />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Residual Plot (regression) */}
          {isRegression && result.residuals && (
            <div>
              <h3 className="text-[12px] text-text-secondary mb-3">Residual Plot</h3>
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="pred"
                    type="number"
                    name="Predicted"
                    tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                    label={{ value: "Predicted", position: "bottom", fontSize: 10, fill: "var(--color-text-tertiary)" }}
                  />
                  <YAxis
                    dataKey="residual"
                    type="number"
                    name="Residual"
                    tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                    label={{ value: "Residual", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-text-tertiary)" }}
                  />
                  <ReferenceLine y={0} stroke="var(--color-border)" strokeDasharray="4 4" />
                  <Tooltip
                    contentStyle={{ fontSize: 11, background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 6 }}
                    formatter={(v) => Number(v).toFixed(3)}
                  />
                  <Scatter
                    data={result.residuals.y_true.map((yt, i) => ({
                      pred: result.residuals!.y_pred[i],
                      residual: yt - result.residuals!.y_pred[i],
                    }))}
                    fill="#8b5cf6"
                    fillOpacity={0.5}
                    r={3}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Cluster Scatter Plot */}
          {isClustering && result.scatter && (
            <div>
              <h3 className="text-[12px] text-text-secondary mb-3">
                Cluster Scatter (PCA — {result.scatter.explained_variance.map((v) => `${(v * 100).toFixed(0)}%`).join(" + ")} variance)
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="x"
                    type="number"
                    name={result.scatter.x_label}
                    tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                    label={{ value: result.scatter.x_label, position: "bottom", fontSize: 10, fill: "var(--color-text-tertiary)" }}
                  />
                  <YAxis
                    dataKey="y"
                    type="number"
                    name={result.scatter.y_label}
                    tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                    label={{ value: result.scatter.y_label, angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-text-tertiary)" }}
                  />
                  <Tooltip
                    contentStyle={{ fontSize: 11, background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 6 }}
                    formatter={(v) => Number(v).toFixed(3)}
                  />
                  {Array.from(new Set(result.scatter.labels)).map((clusterId) => {
                    const colors = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#10b981", "#ec4899", "#6366f1", "#84cc16"];
                    return (
                      <Scatter
                        key={clusterId}
                        name={`Group ${clusterId + 1}`}
                        data={result.scatter!.x
                          .map((xv, i) => ({ x: xv, y: result.scatter!.y[i], label: result.scatter!.labels[i] }))
                          .filter((d) => d.label === clusterId)}
                        fill={colors[clusterId % colors.length]}
                        fillOpacity={0.6}
                        r={3}
                      />
                    );
                  })}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Export predictions */}
          {result.predictions && result.predictions.length > 0 && (
            <div>
              <ExportButton result={result} />
            </div>
          )}

          {/* Cross-validation scores */}
          {result.cv_scores && (
            <div>
              <h3 className="text-[12px] text-text-secondary mb-3">
                Cross-Validation ({result.cv_scores.folds}-fold)
              </h3>
              <div className="flex items-baseline gap-4 mb-2">
                <span className="text-[15px] font-semibold text-text-primary tabular-nums">
                  {result.cv_scores.metric === "accuracy"
                    ? `${(result.cv_scores.mean * 100).toFixed(1)}%`
                    : result.cv_scores.mean.toFixed(3)
                  }
                </span>
                <span className="text-[11px] text-text-tertiary">
                  mean {result.cv_scores.metric} (± {result.cv_scores.std.toFixed(3)})
                </span>
              </div>
              <div className="flex gap-1.5 items-end" style={{ height: 32 }}>
                {result.cv_scores.scores.map((score, i) => {
                  const min = Math.min(...result.cv_scores!.scores);
                  const max = Math.max(...result.cv_scores!.scores);
                  const range = max - min || 1;
                  const pct = ((score - min) / range) * 0.7 + 0.3; // scale 30%-100%
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-plume-400/50 dark:bg-plume-500/30 rounded-sm"
                      style={{ height: `${pct * 100}%` }}
                      title={`Fold ${i + 1}: ${result.cv_scores!.metric === "accuracy" ? `${(score * 100).toFixed(1)}%` : score.toFixed(3)}`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* Train/test info */}
          {result.train_size && (
            <p className="text-[11px] text-text-tertiary">
              Trained on {result.train_size.toLocaleString()} rows, tested on {result.test_size?.toLocaleString()} rows.
            </p>
          )}

          {/* Action buttons */}
          <div className="flex flex-wrap gap-2 pt-2">
            {result.predictions && result.predictions.length > 0 && (
              <ExportButton result={result} />
            )}
            <PickleExportButton result={result} />
            <ReportExportButton results={trainingResults} />
            {result.task !== "clustering" && (
              <ShapButton result={result} />
            )}
          </div>
        </motion.div>
          </>
        )}
      </div>
    </div>
  );
}

function ComparisonTable({
  results,
  onSelect,
}: {
  results: import("../types/data").TrainResult[];
  onSelect: (idx: number) => void;
}) {
  const taskType = results[0]?.task;
  const isClassification = taskType === "classification";
  const isRegression = taskType === "regression";
  const isClustering = taskType === "clustering";

  // Find best result for highlighting
  const getBestIdx = () => {
    if (isClassification) {
      return results.reduce((best, r, i) =>
        (r.metrics.accuracy ?? 0) > (results[best].metrics.accuracy ?? 0) ? i : best, 0);
    }
    if (isRegression) {
      return results.reduce((best, r, i) =>
        (r.metrics.r2 ?? 0) > (results[best].metrics.r2 ?? 0) ? i : best, 0);
    }
    return -1;
  };
  const bestIdx = getBestIdx();

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="px-3 py-2 text-left text-text-tertiary font-medium">Algorithm</th>
              {isClassification && (
                <>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">Accuracy</th>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">Precision</th>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">Recall</th>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">F1</th>
                </>
              )}
              {isRegression && (
                <>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">R²</th>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">MAE</th>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">RMSE</th>
                </>
              )}
              {isClustering && (
                <>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">Groups</th>
                  <th className="px-3 py-2 text-right text-text-tertiary font-medium">Silhouette</th>
                </>
              )}
              {results.some((r) => r.cv_scores) && (
                <th className="px-3 py-2 text-right text-text-tertiary font-medium">CV Score</th>
              )}
              <th className="px-3 py-2 text-right text-text-tertiary font-medium">Features</th>
              <th className="px-3 py-2 w-[60px]" />
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => {
              const isBest = i === bestIdx;
              return (
                <tr
                  key={i}
                  className={`border-b border-border/50 transition-colors duration-100 ${
                    isBest ? "bg-plume-50/50 dark:bg-plume-500/5" : "hover:bg-surface-alt"
                  }`}
                >
                  <td className="px-3 py-2.5 font-medium text-text-primary">
                    {ALGO_LABELS[r.algorithm] ?? r.algorithm}
                    {isBest && (
                      <span className="ml-2 text-[9px] text-plume-600 dark:text-plume-400 font-normal">best</span>
                    )}
                  </td>
                  {isClassification && (
                    <>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-primary">
                        {(r.metrics.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                        {(r.metrics.precision * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                        {(r.metrics.recall * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                        {(r.metrics.f1 * 100).toFixed(1)}%
                      </td>
                    </>
                  )}
                  {isRegression && (
                    <>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-primary">
                        {r.metrics.r2.toFixed(3)}
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                        {r.metrics.mae.toFixed(3)}
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                        {r.metrics.rmse.toFixed(3)}
                      </td>
                    </>
                  )}
                  {isClustering && (
                    <>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-primary">
                        {r.metrics.n_clusters}
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                        {r.metrics.silhouette?.toFixed(3) ?? "—"}
                      </td>
                    </>
                  )}
                  {results.some((res) => res.cv_scores) && (
                    <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                      {r.cv_scores
                        ? `${r.cv_scores.metric === "accuracy"
                            ? `${(r.cv_scores.mean * 100).toFixed(1)}%`
                            : r.cv_scores.mean.toFixed(3)
                          } ± ${r.cv_scores.std.toFixed(3)}`
                        : "—"
                      }
                    </td>
                  )}
                  <td className="px-3 py-2.5 text-right tabular-nums text-text-tertiary">
                    {r.features_used?.length ?? "—"}
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    <button
                      onClick={() => onSelect(i)}
                      className="text-[11px] text-plume-600 dark:text-plume-500 hover:text-plume-700 cursor-pointer"
                    >
                      View
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Combined ROC curve overlay (classification only) */}
      {isClassification && results.some((r) => r.roc_curve) && (
        <div className="mt-6">
          <h3 className="text-[12px] text-text-secondary mb-1">ROC Curves</h3>
          <p className="text-[11px] text-text-tertiary mb-3">The closer the curve is to the top-left corner, the better the model. The diagonal line represents random guessing.</p>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="fpr"
                type="number"
                domain={[0, 1]}
                tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                label={{ value: "False Positive Rate", position: "bottom", fontSize: 10, fill: "var(--color-text-tertiary)" }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-text-tertiary)" }}
              />
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke="var(--color-border)"
                strokeDasharray="4 4"
              />
              <Tooltip
                contentStyle={{ fontSize: 11, background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 6 }}
                formatter={(v) => Number(v).toFixed(3)}
              />
              {(() => {
                const colors = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#10b981", "#ec4899", "#6366f1", "#84cc16"];
                return results
                  .filter((r) => r.roc_curve)
                  .map((r, i) => {
                    const data = r.roc_curve!.fpr.map((fpr, j) => ({
                      fpr,
                      [r.algorithm]: r.roc_curve!.tpr[j],
                    }));
                    return (
                      <Line
                        key={r.algorithm}
                        data={data}
                        type="monotone"
                        dataKey={r.algorithm}
                        name={`${ALGO_LABELS[r.algorithm] ?? r.algorithm}${r.roc_curve!.auc != null ? ` (AUC ${r.roc_curve!.auc.toFixed(3)})` : ""}`}
                        stroke={colors[i % colors.length]}
                        strokeWidth={2}
                        dot={false}
                      />
                    );
                  });
              })()}
            </LineChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-4 mt-2">
            {(() => {
              const colors = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#10b981", "#ec4899", "#6366f1", "#84cc16"];
              return results
                .filter((r) => r.roc_curve)
                .map((r, i) => (
                  <div key={r.algorithm} className="flex items-center gap-1.5">
                    <div className="w-3 h-[2px] rounded-full" style={{ backgroundColor: colors[i % colors.length] }} />
                    <span className="text-[11px] text-text-secondary">
                      {ALGO_LABELS[r.algorithm] ?? r.algorithm}
                      {r.roc_curve!.auc != null && (
                        <span className="text-text-tertiary ml-1">AUC {r.roc_curve!.auc.toFixed(3)}</span>
                      )}
                    </span>
                  </div>
                ));
            })()}
          </div>
        </div>
      )}

      {/* Features used summary */}
      {results[0]?.features_used && (
        <p className="text-[11px] text-text-tertiary mt-4">
          Features: {results[0].features_used.join(", ")}
        </p>
      )}
    </motion.div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  const explanation = METRIC_EXPLANATIONS[label];
  return (
    <div className="p-3 border border-border rounded-[var(--radius-default)]">
      <p className="text-[11px] text-text-tertiary mb-1">
        {label}
        {explanation && (
          <span
            className="ml-1 text-text-tertiary/60 cursor-help"
            title={explanation}
          >
            ?
          </span>
        )}
      </p>
      <p className="text-[18px] font-semibold text-text-primary tabular-nums">{value}</p>
    </div>
  );
}

function ExportButton({ result }: { result: TrainResult }) {
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    const filePath = await save({
      defaultPath: `plume_predictions_${result.algorithm}.csv`,
      filters: [{ name: "CSV", extensions: ["csv"] }],
    });
    if (!filePath) return;

    setExporting(true);
    try {
      const header = "row_index,prediction";
      const rows = result.predictions!.map((p, i) => `${i},${p}`);
      const csv = [header, ...rows].join("\n");
      await invoke("save_text_file", { path: filePath, content: csv });
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setExporting(false);
    }
  };

  return (
    <button
      onClick={handleExport}
      disabled={exporting}
      className="px-4 py-2 text-[12px] rounded-[var(--radius-default)] border border-border text-text-secondary hover:bg-surface-alt hover:text-text-primary disabled:opacity-40 transition-colors duration-200 cursor-pointer"
    >
      {exporting ? "Exporting..." : "Export predictions as CSV"}
    </button>
  );
}

function PickleExportButton({ result }: { result: TrainResult }) {
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    const filePath = await save({
      defaultPath: `plume_model_${result.algorithm}.pkl`,
      filters: [{ name: "Pickle", extensions: ["pkl"] }],
    });
    if (!filePath) return;

    setExporting(true);
    try {
      await invoke("export_model_pickle", {
        task: result.task,
        target: result.target ?? null,
        features: result.features_used ?? [],
        algorithm: result.algorithm,
        hyperparams: result.hyperparams ?? {},
        outputPath: filePath,
      });
    } catch (err) {
      console.error("Pickle export failed:", err);
      alert(`Export failed: ${err}`);
    } finally {
      setExporting(false);
    }
  };

  if (result.task === "clustering") return null;

  return (
    <button
      onClick={handleExport}
      disabled={exporting}
      className="px-4 py-2 text-[12px] rounded-[var(--radius-default)] border border-border text-text-secondary hover:bg-surface-alt hover:text-text-primary disabled:opacity-40 transition-colors duration-200 cursor-pointer"
    >
      {exporting ? "Exporting..." : "Export model (.pkl)"}
    </button>
  );
}

function ReportExportButton({ results }: { results: TrainResult[] }) {
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    const filePath = await save({
      defaultPath: "plume_report.html",
      filters: [{ name: "HTML", extensions: ["html"] }],
    });
    if (!filePath) return;

    setExporting(true);
    try {
      await invoke("generate_report", {
        results: results.map((r) => ({
          task: r.task,
          algorithm: r.algorithm,
          metrics: r.metrics,
          feature_importance: r.feature_importance,
          features_used: r.features_used,
          train_size: r.train_size,
          test_size: r.test_size,
          cv_scores: r.cv_scores,
        })),
        outputPath: filePath,
      });
    } catch (err) {
      console.error("Report generation failed:", err);
      alert(`Report failed: ${err}`);
    } finally {
      setExporting(false);
    }
  };

  return (
    <button
      onClick={handleExport}
      disabled={exporting}
      className="px-4 py-2 text-[12px] rounded-[var(--radius-default)] border border-border text-text-secondary hover:bg-surface-alt hover:text-text-primary disabled:opacity-40 transition-colors duration-200 cursor-pointer"
    >
      {exporting ? "Generating..." : "Export report (.html)"}
    </button>
  );
}

interface ShapExplanation {
  prediction: string;
  contributions: { feature: string; value: number; shap_value: number }[];
}

function ShapButton({ result }: { result: TrainResult }) {
  const [loading, setLoading] = useState(false);
  const [explanations, setExplanations] = useState<ShapExplanation[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCompute = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await invoke<{ explanations: ShapExplanation[] }>("compute_shap", {
        task: result.task,
        target: result.target ?? null,
        features: result.features_used ?? [],
        algorithm: result.algorithm,
        hyperparams: result.hyperparams ?? {},
        nSamples: 5,
      });
      setExplanations(resp.explanations);
    } catch (err) {
      const msg = String(err);
      if (msg.includes("not installed")) {
        setError("SHAP library not installed. Run: pip install shap");
      } else {
        setError(String(err));
      }
    } finally {
      setLoading(false);
    }
  };

  if (explanations) {
    return (
      <div className="w-full mt-4">
        <h3 className="text-[12px] text-text-secondary mb-3">SHAP Explanations (sample predictions)</h3>
        <div className="flex flex-col gap-4">
          {explanations.map((exp, i) => (
            <div key={i} className="p-3 border border-border rounded-[var(--radius-default)]">
              <p className="text-[12px] font-medium text-text-primary mb-2">
                Prediction: {exp.prediction}
              </p>
              <div className="flex flex-col gap-1">
                {exp.contributions.slice(0, 6).map((c) => (
                  <div key={c.feature} className="flex items-center gap-2 text-[11px]">
                    <span className="w-[100px] text-right text-text-secondary truncate shrink-0">{c.feature}</span>
                    <div className="flex-1 flex items-center">
                      <div
                        className={`h-[4px] rounded-full ${c.shap_value >= 0 ? "bg-emerald-400" : "bg-red-400"}`}
                        style={{ width: `${Math.min(Math.abs(c.shap_value) * 200, 100)}%` }}
                      />
                    </div>
                    <span className={`w-[50px] text-right tabular-nums ${c.shap_value >= 0 ? "text-emerald-600" : "text-red-500"}`}>
                      {c.shap_value > 0 ? "+" : ""}{c.shap_value.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleCompute}
        disabled={loading}
        className="px-4 py-2 text-[12px] rounded-[var(--radius-default)] border border-border text-text-secondary hover:bg-surface-alt hover:text-text-primary disabled:opacity-40 transition-colors duration-200 cursor-pointer"
      >
        {loading ? "Computing..." : "Explain predictions (SHAP)"}
      </button>
      {error && <span className="text-[11px] text-red-500">{error}</span>}
    </div>
  );
}

function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="inline-block">
      <table className="border-collapse text-[11px]">
        <thead>
          <tr>
            <th className="w-[60px]" />
            {labels.map((l) => (
              <th key={l} className="px-2 py-1 text-text-tertiary font-normal text-center">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="px-2 py-1 text-text-tertiary text-right">{labels[i]}</td>
              {row.map((val, j) => {
                const intensity = maxVal > 0 ? val / maxVal : 0;
                const isDiagonal = i === j;
                return (
                  <td
                    key={j}
                    className="px-2 py-1 text-center w-[50px] h-[36px]"
                    style={{
                      backgroundColor: isDiagonal
                        ? `rgba(99, 102, 241, ${intensity * 0.3})`
                        : `rgba(239, 68, 68, ${intensity * 0.2})`,
                    }}
                  >
                    <span className="text-[12px] font-medium text-text-primary">{val}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
