import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ScatterChart, Scatter, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../stores/appStore";
import type { TrainResult } from "../types/data";
import { ChartSizeToggle, useChartSize } from "./ChartSizeToggle";

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
  const [chartHeight, chartSize, setChartSize] = useChartSize("M");

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

          {/* Overfitting warning */}
          {result.train_metrics && (() => {
            const trainScore = isClassification
              ? result.train_metrics!.accuracy
              : isRegression
                ? result.train_metrics!.r2
                : null;
            const testScore = isClassification
              ? result.metrics.accuracy
              : isRegression
                ? result.metrics.r2
                : null;
            if (trainScore == null || testScore == null) return null;
            const gap = trainScore - testScore;
            if (gap < 0.1) return null;
            const trainPct = (trainScore * 100).toFixed(0);
            const testPct = (testScore * 100).toFixed(0);
            const metric = isClassification ? "accuracy" : "R²";
            return (
              <div className="bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-700 rounded-[var(--radius-default)] px-4 py-3">
                <p className="text-[12px] font-medium text-amber-700 dark:text-amber-400 mb-1">
                  Possible overfitting detected
                </p>
                <p className="text-[12px] text-amber-700/80 dark:text-amber-300/80 leading-relaxed">
                  Your model scores {trainPct}% {metric} on training data but only {testPct}% on new data.
                  This suggests the model memorized the training examples rather than learning general patterns.
                  Try reducing model complexity (fewer trees, shallower depth) or adding more data.
                </p>
              </div>
            );
          })()}

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
                      <span className="text-[12px] text-text-secondary min-w-[140px] max-w-[220px] text-right truncate shrink-0" title={fi.feature}>
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
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-[12px] text-text-secondary">
                  ROC Curve{result.roc_curve.auc != null && ` (AUC = ${result.roc_curve.auc.toFixed(3)})`}
                </h3>
                <ChartSizeToggle size={chartSize} onChange={setChartSize} />
              </div>
              <p className="text-[11px] text-text-tertiary mb-3">The closer the curve is to the top-left corner, the better the model. The diagonal line represents random guessing.</p>
              <ResponsiveContainer width="100%" height={chartHeight}>
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
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-[12px] text-text-secondary">Residual Plot</h3>
                <ChartSizeToggle size={chartSize} onChange={setChartSize} />
              </div>
              <ResponsiveContainer width="100%" height={chartHeight}>
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
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-[12px] text-text-secondary">
                  Cluster Scatter (PCA — {result.scatter.explained_variance.map((v) => `${(v * 100).toFixed(0)}%`).join(" + ")} variance)
                </h3>
                <ChartSizeToggle size={chartSize} onChange={setChartSize} />
              </div>
              <ResponsiveContainer width="100%" height={chartHeight}>
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
            <RetrainPanel result={result} onRetrained={() => setSelectedIdx(0)} />
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
  const [chartHeight, chartSize, setChartSize] = useChartSize("M");

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
          <div className="flex items-center justify-between mb-1">
            <h3 className="text-[12px] text-text-secondary">ROC Curves</h3>
            <ChartSizeToggle size={chartSize} onChange={setChartSize} />
          </div>
          <p className="text-[11px] text-text-tertiary mb-3">The closer the curve is to the top-left corner, the better the model. The diagonal line represents random guessing.</p>
          <ResponsiveContainer width="100%" height={chartHeight}>
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

      {/* Shared top features across models */}
      {results.some((r) => r.feature_importance && r.feature_importance.length > 0) && (
        <SharedFeatureImportance results={results} />
      )}
    </motion.div>
  );
}

function SharedFeatureImportance({ results }: { results: import("../types/data").TrainResult[] }) {
  // Collect feature importance across all models, averaging the normalized rank
  const modelsWithImportance = results.filter((r) => r.feature_importance && r.feature_importance.length > 0);
  if (modelsWithImportance.length < 2) return null;

  // For each model, get top 10 features. Score each feature by its average normalized importance across models.
  const featureScores = new Map<string, { totalImportance: number; modelCount: number }>();

  for (const r of modelsWithImportance) {
    const fi = r.feature_importance!;
    const maxImp = fi[0]?.importance ?? 1;
    for (const f of fi.slice(0, 10)) {
      const normalized = maxImp > 0 ? f.importance / maxImp : 0;
      const existing = featureScores.get(f.feature) ?? { totalImportance: 0, modelCount: 0 };
      existing.totalImportance += normalized;
      existing.modelCount += 1;
      featureScores.set(f.feature, existing);
    }
  }

  // Only show features that appear in at least 2 models
  const shared = Array.from(featureScores.entries())
    .filter(([, v]) => v.modelCount >= 2)
    .map(([feature, v]) => ({
      feature,
      avgImportance: v.totalImportance / modelsWithImportance.length,
      agreedBy: v.modelCount,
    }))
    .sort((a, b) => b.avgImportance - a.avgImportance)
    .slice(0, 10);

  if (shared.length === 0) return null;
  const maxAvg = shared[0].avgImportance;

  return (
    <div className="mt-6">
      <h3 className="text-[12px] text-text-secondary mb-1">Features agreed upon across models</h3>
      <p className="text-[11px] text-text-tertiary mb-3">
        These features were ranked as important by multiple models. The bar shows average relative importance.
      </p>
      <div className="flex flex-col gap-2">
        {shared.map((f, i) => {
          const pct = maxAvg > 0 ? (f.avgImportance / maxAvg) * 100 : 0;
          return (
            <div key={f.feature} className="flex items-center gap-3">
              <span className="text-[12px] text-text-secondary min-w-[140px] max-w-[220px] text-right truncate shrink-0" title={f.feature}>
                {f.feature}
              </span>
              <div className="flex-1 h-[6px] bg-border rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.4, delay: i * 0.05, ease: "easeOut" }}
                  className="h-full bg-plume-500 rounded-full"
                />
              </div>
              <span className="text-[10px] text-text-tertiary w-[60px] shrink-0">
                {f.agreedBy}/{modelsWithImportance.length} models
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  const explanation = METRIC_EXPLANATIONS[label];
  const [showTooltip, setShowTooltip] = useState(false);
  return (
    <div className="p-3 border border-border rounded-[var(--radius-default)]">
      <p className="text-[11px] text-text-tertiary mb-1">
        {label}
        {explanation && (
          <span className="relative inline-block ml-1">
            <span
              className="text-text-tertiary/60 cursor-help hover:text-plume-500 transition-colors duration-150"
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              onClick={() => setShowTooltip(!showTooltip)}
            >
              ?
            </span>
            {showTooltip && (
              <span className="absolute z-10 bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-[220px] px-3 py-2 text-[11px] leading-relaxed text-text-primary bg-surface border border-border rounded-[var(--radius-default)] shadow-md pointer-events-none">
                {explanation}
              </span>
            )}
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
        <h3 className="text-[12px] text-text-secondary mb-1">SHAP Explanations (sample predictions)</h3>
        <p className="text-[11px] text-text-tertiary mb-3">
          Each card shows one prediction and the factors that influenced it most.
          {result.task === "classification"
            ? " Green bars pushed toward the predicted outcome; red bars pushed away from it."
            : ` Green bars pushed ${result.target ?? "the prediction"} higher; red bars pushed it lower.`
          }{" "}Longer bars = stronger influence.
        </p>
        <div className="flex flex-col gap-4">
          {explanations.map((exp, i) => {
            const topPositive = exp.contributions.filter((c) => c.shap_value > 0).slice(0, 2);
            const topNegative = exp.contributions.filter((c) => c.shap_value < 0).slice(0, 2);
            const isClassification = result.task === "classification";
            const targetName = result.target ?? "the prediction";
            return (
              <div key={i} className="p-3 border border-border rounded-[var(--radius-default)]">
                <p className="text-[12px] font-medium text-text-primary mb-2">
                  Prediction: {exp.prediction}
                </p>
                <div className="flex flex-col gap-1">
                  {exp.contributions.slice(0, 6).map((c) => (
                    <div key={c.feature} className="flex items-center gap-2 text-[11px]">
                      <span className="min-w-[100px] max-w-[180px] text-right text-text-secondary truncate shrink-0" title={c.feature}>{c.feature}</span>
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
                {/* Plain-English explanation */}
                <div className="mt-2 pt-2 border-t border-border/50">
                  <p className="text-[11px] text-text-secondary leading-relaxed">
                    <span className="font-medium">What does this mean?</span>{" "}
                    {topPositive.length > 0 && (
                      <>
                        {topPositive.map((c) => (
                          <span key={c.feature}><span className="font-medium text-emerald-600 dark:text-emerald-400">{c.feature}</span> (value: {c.value})</span>
                        )).reduce<React.ReactNode[]>((acc, el, idx) => idx === 0 ? [el] : [...acc, " and ", el], [])}
                        {isClassification
                          ? <>{" "}pushed toward <span className="text-emerald-600 dark:text-emerald-400 font-medium">{exp.prediction}</span>.{" "}</>
                          : <>{" "}pushed <span className="text-emerald-600 dark:text-emerald-400 font-medium">{targetName}</span> higher.{" "}</>
                        }
                      </>
                    )}
                    {topNegative.length > 0 && (
                      <>
                        {topNegative.map((c) => (
                          <span key={c.feature}><span className="font-medium text-red-500 dark:text-red-400">{c.feature}</span> (value: {c.value})</span>
                        )).reduce<React.ReactNode[]>((acc, el, idx) => idx === 0 ? [el] : [...acc, " and ", el], [])}
                        {isClassification
                          ? <>{" "}pushed away from <span className="text-red-500 dark:text-red-400 font-medium">{exp.prediction}</span>.</>
                          : <>{" "}pushed <span className="text-red-500 dark:text-red-400 font-medium">{targetName}</span> lower.</>
                        }
                      </>
                    )}
                    {topPositive.length === 0 && topNegative.length === 0 && (
                      <>No single feature had a strong influence on this prediction.</>
                    )}
                  </p>
                </div>
              </div>
            );
          })}
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

const HYPERPARAM_DEFS: Record<string, { key: string; label: string; default: number; min: number; max: number; step: number }[]> = {
  random_forest: [
    { key: "n_estimators", label: "Trees", default: 100, min: 10, max: 1000, step: 10 },
    { key: "max_depth", label: "Max depth (0 = unlimited)", default: 0, min: 0, max: 100, step: 1 },
    { key: "min_samples_split", label: "Min samples to split", default: 2, min: 2, max: 50, step: 1 },
  ],
  logistic_regression: [
    { key: "C", label: "Regularization (C)", default: 1.0, min: 0.01, max: 100, step: 0.1 },
    { key: "max_iter", label: "Max iterations", default: 1000, min: 100, max: 10000, step: 100 },
  ],
  linear_regression: [],
  xgboost: [
    { key: "n_estimators", label: "Trees", default: 100, min: 10, max: 1000, step: 10 },
    { key: "max_depth", label: "Max depth (0 = unlimited)", default: 6, min: 0, max: 20, step: 1 },
    { key: "learning_rate", label: "Learning rate", default: 0.1, min: 0.01, max: 1, step: 0.01 },
  ],
  lightgbm: [
    { key: "n_estimators", label: "Trees", default: 100, min: 10, max: 1000, step: 10 },
    { key: "max_depth", label: "Max depth (0 = unlimited)", default: -1, min: -1, max: 50, step: 1 },
    { key: "learning_rate", label: "Learning rate", default: 0.1, min: 0.01, max: 1, step: 0.01 },
    { key: "num_leaves", label: "Num leaves", default: 31, min: 2, max: 256, step: 1 },
  ],
};

function RetrainPanel({ result, onRetrained }: { result: TrainResult; onRetrained: () => void }) {
  const [open, setOpen] = useState(false);
  const [training, setTraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<TrainResult | null>(null);
  // Capture the original result at first render so comparisons stay stable after selectedIdx changes
  const [originalResult] = useState<TrainResult>(result);
  const addTrainingResult = useAppStore((s) => s.addTrainingResult);
  const defs = HYPERPARAM_DEFS[originalResult.algorithm] ?? [];

  // Initialize from the original result's hyperparams
  const [hp, setHp] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    for (const def of defs) {
      initial[def.key] = originalResult.hyperparams?.[def.key] ?? def.default;
    }
    return initial;
  });

  if (defs.length === 0) return null;

  const handleRetrain = async () => {
    setTraining(true);
    setRetrainResult(null);
    try {
      const built: Record<string, number> = {};
      for (const def of defs) {
        built[def.key] = hp[def.key] ?? def.default;
      }
      const newResult = await invoke<TrainResult>("train_model", {
        task: originalResult.task,
        target: originalResult.target ?? null,
        features: originalResult.features_used ?? [],
        algorithm: originalResult.algorithm,
        nClusters: null,
        hyperparams: built,
        useCv: !!originalResult.cv_scores,
        cvFolds: originalResult.cv_scores?.folds ?? 5,
      });
      newResult.target = originalResult.target;
      newResult.hyperparams = built;
      addTrainingResult(newResult);
      setRetrainResult(newResult);
      onRetrained();
    } catch (err) {
      console.error("Retrain failed:", err);
    } finally {
      setTraining(false);
    }
  };

  // Build the metric comparison rows — always compare against the original result
  const getComparisonMetrics = (): { label: string; oldVal: number; newVal: number; format: (v: number) => string; higherIsBetter: boolean }[] => {
    if (!retrainResult) return [];
    const isClassification = originalResult.task === "classification";
    const isRegression = originalResult.task === "regression";
    const pctFmt = (v: number) => `${(v * 100).toFixed(1)}%`;
    const decFmt = (v: number) => v.toFixed(3);

    if (isClassification) {
      return [
        { label: "Accuracy", oldVal: originalResult.metrics.accuracy, newVal: retrainResult.metrics.accuracy, format: pctFmt, higherIsBetter: true },
        { label: "Precision", oldVal: originalResult.metrics.precision, newVal: retrainResult.metrics.precision, format: pctFmt, higherIsBetter: true },
        { label: "Recall", oldVal: originalResult.metrics.recall, newVal: retrainResult.metrics.recall, format: pctFmt, higherIsBetter: true },
        { label: "F1", oldVal: originalResult.metrics.f1, newVal: retrainResult.metrics.f1, format: pctFmt, higherIsBetter: true },
      ];
    }
    if (isRegression) {
      return [
        { label: "R²", oldVal: originalResult.metrics.r2, newVal: retrainResult.metrics.r2, format: decFmt, higherIsBetter: true },
        { label: "MAE", oldVal: originalResult.metrics.mae, newVal: retrainResult.metrics.mae, format: decFmt, higherIsBetter: false },
        { label: "RMSE", oldVal: originalResult.metrics.rmse, newVal: retrainResult.metrics.rmse, format: decFmt, higherIsBetter: false },
      ];
    }
    return [];
  };

  return (
    <>
      <button
        onClick={() => setOpen(!open)}
        className={`px-4 py-2 text-[12px] rounded-[var(--radius-default)] border transition-colors duration-200 cursor-pointer ${
          open
            ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
            : "border-border text-text-secondary hover:bg-surface-alt hover:text-text-primary"
        }`}
      >
        Tune & Retrain
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="w-full overflow-hidden"
          >
            <div className="p-4 border border-border rounded-[var(--radius-default)] bg-surface-alt mt-2">
              <p className="text-[11px] text-text-tertiary mb-3">
                Adjust the hyperparameters below and retrain with the same features and target.
                The new result will appear alongside your existing results for easy comparison.
              </p>
              <div className="grid grid-cols-2 gap-3 mb-4">
                {defs.map((def) => (
                  <div key={def.key} className="flex flex-col gap-1">
                    <label className="text-[10px] text-text-tertiary">{def.label}</label>
                    <input
                      type="number"
                      min={def.min}
                      max={def.max}
                      step={def.step}
                      value={hp[def.key] ?? def.default}
                      onChange={(e) =>
                        setHp((prev) => ({
                          ...prev,
                          [def.key]: parseFloat(e.target.value) || def.default,
                        }))
                      }
                      className="w-full px-2 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200 tabular-nums"
                    />
                  </div>
                ))}
              </div>
              <button
                onClick={handleRetrain}
                disabled={training}
                className="px-5 py-2 text-[12px] font-medium rounded-[var(--radius-default)] bg-plume-600 text-white hover:bg-plume-700 disabled:opacity-50 transition-colors duration-200 cursor-pointer"
              >
                {training ? (
                  <span className="flex items-center gap-2">
                    <motion.span
                      className="inline-block w-2 h-2 rounded-full bg-white/60"
                      animate={{ scale: [1, 1.3, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                    />
                    Retraining...
                  </span>
                ) : (
                  `Retrain ${ALGO_LABELS[originalResult.algorithm] ?? originalResult.algorithm}`
                )}
              </button>

              {/* Before/after comparison */}
              {retrainResult && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  className="mt-4 pt-4 border-t border-border"
                >
                  <h4 className="text-[11px] font-medium text-text-secondary mb-2">Performance comparison</h4>
                  <div className="flex flex-col gap-1.5">
                    {getComparisonMetrics().map((m) => {
                      const diff = m.newVal - m.oldVal;
                      const improved = m.higherIsBetter ? diff > 0 : diff < 0;
                      const unchanged = Math.abs(diff) < 0.0001;
                      const diffStr = m.label === "R²" || m.label === "MAE" || m.label === "RMSE"
                        ? `${diff > 0 ? "+" : ""}${diff.toFixed(3)}`
                        : `${diff > 0 ? "+" : ""}${(diff * 100).toFixed(1)}%`;
                      return (
                        <div key={m.label} className="flex items-center gap-2 text-[12px]">
                          <span className="w-[70px] text-text-tertiary text-right shrink-0">{m.label}</span>
                          <span className="text-text-secondary tabular-nums">{m.format(m.oldVal)}</span>
                          <span className="text-text-tertiary">→</span>
                          <span className="text-text-primary font-medium tabular-nums">{m.format(m.newVal)}</span>
                          <span className={`text-[11px] tabular-nums font-medium ${
                            unchanged
                              ? "text-text-tertiary"
                              : improved
                                ? "text-emerald-600 dark:text-emerald-400"
                                : "text-red-500 dark:text-red-400"
                          }`}>
                            {unchanged ? "—" : `${improved ? "↑" : "↓"} ${diffStr}`}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
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
