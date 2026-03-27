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

const HYPERPARAM_SHORT_LABELS: Record<string, string> = {
  n_estimators: "trees",
  max_depth: "depth",
  learning_rate: "lr",
  min_samples_split: "min split",
  C: "C",
  max_iter: "max iter",
  num_leaves: "leaves",
  eps: "eps",
  min_samples: "min samples",
};

/** Generate a display label for a training result, disambiguating duplicate algorithms. */
function getRunLabel(result: TrainResult, allResults: TrainResult[]): string {
  if (result.nickname) return result.nickname;
  const algoLabel = ALGO_LABELS[result.algorithm] ?? result.algorithm;
  const sameAlgo = allResults.filter((r) => r.algorithm === result.algorithm);
  if (sameAlgo.length <= 1) return algoLabel;
  // Disambiguate with hyperparams if they differ
  const hp = result.hyperparams;
  if (hp && Object.keys(hp).length > 0) {
    const parts = Object.entries(hp)
      .slice(0, 3)
      .map(([k, v]) => `${HYPERPARAM_SHORT_LABELS[k] ?? k} ${v}`);
    return `${algoLabel} (${parts.join(", ")})`;
  }
  // Disambiguate by feature count if they trained on different feature sets
  const featureCount = result.features_used?.length ?? 0;
  const sameAlgoFeatureCounts = sameAlgo.map((r) => r.features_used?.length ?? 0);
  const allSameFeatureCount = sameAlgoFeatureCounts.every((c) => c === featureCount);
  if (!allSameFeatureCount) {
    return `${algoLabel} (${featureCount} features)`;
  }
  // Fallback: use run number
  const idx = sameAlgo.indexOf(result);
  return `${algoLabel} #${sameAlgo.length - idx}`;
}

function EditableNickname({
  result,
  resultIndex,
  allResults,
}: {
  result: TrainResult;
  resultIndex: number;
  allResults: TrainResult[];
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(result.nickname ?? "");
  const updateNickname = useAppStore((s) => s.updateTrainingResultNickname);

  const save = () => {
    const trimmed = value.trim();
    updateNickname(resultIndex, trimmed);
    setEditing(false);
  };

  if (editing) {
    return (
      <input
        autoFocus
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={save}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") { setValue(result.nickname ?? ""); setEditing(false); }
        }}
        placeholder={getRunLabel(result, allResults)}
        className="text-[11px] px-1.5 py-0.5 border border-plume-500 rounded bg-surface text-text-primary outline-none w-[160px]"
      />
    );
  }

  return (
    <span
      onClick={() => { setValue(result.nickname ?? ""); setEditing(true); }}
      className="cursor-pointer hover:text-plume-600 dark:hover:text-plume-400 transition-colors duration-150"
      title="Click to rename"
    >
      {getRunLabel(result, allResults)}
    </span>
  );
}

type ResultTab = "summary" | "details";

export function ResultsView() {
  const summary = useAppStore((s) => s.summary);
  const trainingResults = useAppStore((s) => s.trainingResults);
  const removeTrainingResult = useAppStore((s) => s.removeTrainingResult);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [tab, setTab] = useState<ResultTab>("summary");
  const [chartHeight, chartSize, setChartSize] = useChartSize("M");
  const [overfitFixResult, setOverfitFixResult] = useState<TrainResult | null>(null);

  // Keep selectedIdx in bounds when results are removed
  const safeIdx = Math.min(selectedIdx, trainingResults.length - 1);
  if (safeIdx !== selectedIdx && trainingResults.length > 0) setSelectedIdx(safeIdx);

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
                  <div key={i} className="relative group">
                    <button
                      onClick={() => setSelectedIdx(i)}
                      className={`
                        px-3 py-1.5 text-[11px] rounded-[var(--radius-default)] border
                        transition-all duration-200 cursor-pointer pr-6
                        ${selectedIdx === i
                          ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
                          : "border-border text-text-tertiary hover:border-text-tertiary hover:text-text-secondary"
                        }
                      `}
                    >
                      {getRunLabel(r, trainingResults)}
                      {r.task === "classification" && r.metrics.accuracy != null && (
                        <span className="ml-1.5 tabular-nums">{(r.metrics.accuracy * 100).toFixed(0)}%</span>
                      )}
                      {r.task === "regression" && r.metrics.r2 != null && (
                        <span className="ml-1.5 tabular-nums">R²{r.metrics.r2.toFixed(2)}</span>
                      )}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeTrainingResult(i);
                        if (selectedIdx >= trainingResults.length - 1) setSelectedIdx(Math.max(0, trainingResults.length - 2));
                        else if (i < selectedIdx) setSelectedIdx(selectedIdx - 1);
                      }}
                      className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-surface border border-border text-text-tertiary hover:text-red-500 hover:border-red-300 text-[10px] leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-150 cursor-pointer"
                      title="Remove this run"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}

        <motion.div
          key={`${result.algorithm}-${result.trainedAt ?? selectedIdx}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col gap-8"
        >
          {/* Headline */}
          <div>
            <p className="text-[11px] text-text-tertiary mb-1">
              <EditableNickname result={result} resultIndex={selectedIdx} allResults={trainingResults} />
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

          {/* Class imbalance warning */}
          {result.imbalance_warning && (() => {
            const { class_distribution, minority_pct, stratified } = result.imbalance_warning;
            const maxCount = Math.max(...Object.values(class_distribution));
            return (
              <div className="bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-700 rounded-[var(--radius-default)] px-4 py-3">
                <p className="text-[12px] font-medium text-amber-700 dark:text-amber-400 mb-1">
                  Class imbalance detected
                </p>
                <p className="text-[12px] text-amber-700/80 dark:text-amber-300/80 leading-relaxed">
                  Your target has imbalanced classes — the smallest class represents only {minority_pct.toFixed(1)}% of the data.
                  {stratified ? " Plume used stratified splitting to ensure each class is fairly represented in both training and test sets." : " Stratified splitting was not possible due to very small class sizes."}
                </p>
                <div className="mt-2 flex flex-col gap-1">
                  {Object.entries(class_distribution).map(([cls, count]) => (
                    <div key={cls} className="flex items-center gap-2">
                      <span className="text-[11px] text-amber-700/80 dark:text-amber-300/80 min-w-[60px] text-right truncate shrink-0">{cls}</span>
                      <div className="flex-1 h-[5px] bg-amber-200/50 dark:bg-amber-700/30 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-400 rounded-full"
                          style={{ width: `${(count / maxCount) * 100}%` }}
                        />
                      </div>
                      <span className="text-[11px] text-amber-700/70 dark:text-amber-300/70 tabular-nums w-[40px] text-right shrink-0">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}

          {/* Data leakage warning */}
          {result.leakage_warnings && result.leakage_warnings.length > 0 && (() => {
            const warnings = result.leakage_warnings!;
            const count = warnings.length;
            return (
              <div className="bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-700 rounded-[var(--radius-default)] px-4 py-3">
                <p className="text-[12px] font-medium text-red-700 dark:text-red-400 mb-1">
                  Possible data leakage
                </p>
                <p className="text-[12px] text-red-700/80 dark:text-red-300/80 leading-relaxed mb-2">
                  {count === 1 ? "A feature has" : "Some features have"} suspiciously high correlation with the target, which may mean it contains information that wouldn't be available when making real predictions.
                </p>
                <div className="flex flex-col gap-1">
                  {warnings.map((w) => (
                    <div key={w.feature} className="flex items-center gap-2 text-[11px]">
                      <span className="font-medium text-red-700 dark:text-red-400">{w.feature}</span>
                      <span className="text-red-600/60 dark:text-red-400/60">correlation: {w.correlation.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
                <p className="text-[11px] text-red-600/70 dark:text-red-300/60 mt-2">
                  Consider removing {count === 1 ? "this feature" : "these features"} and retraining to see if performance is genuinely this good.
                </p>
              </div>
            );
          })()}

          {/* Overfitting warning with recommendations */}
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
              <OverfitWarning
                trainPct={trainPct}
                testPct={testPct}
                metric={metric}
                gap={gap}
                result={result}
                onFixApplied={(fixResult) => {
                  setSelectedIdx((prev) => prev + 1);
                  setOverfitFixResult(fixResult);
                }}
              />
            );
          })()}

          {/* Metrics */}
          <div>
            <h3 className="text-[12px] text-text-secondary mb-1">Metrics</h3>
            {result.train_metrics && (
              <p className="text-[11px] text-text-tertiary mb-3">
                Test performance is the main number. Training performance is shown below each metric for comparison.
              </p>
            )}
            <div className="grid grid-cols-3 gap-4">
              {isClassification && (
                <>
                  <MetricCard
                    label="Accuracy"
                    value={`${(result.metrics.accuracy * 100).toFixed(1)}%`}
                    trainValue={result.train_metrics?.accuracy != null ? `${(result.train_metrics.accuracy * 100).toFixed(1)}%` : undefined}
                  />
                  <MetricCard
                    label="Precision"
                    value={`${(result.metrics.precision * 100).toFixed(1)}%`}
                    trainValue={result.train_metrics?.precision != null ? `${(result.train_metrics.precision * 100).toFixed(1)}%` : undefined}
                  />
                  <MetricCard
                    label="Recall"
                    value={`${(result.metrics.recall * 100).toFixed(1)}%`}
                    trainValue={result.train_metrics?.recall != null ? `${(result.train_metrics.recall * 100).toFixed(1)}%` : undefined}
                  />
                </>
              )}
              {isRegression && (
                <>
                  <MetricCard
                    label="R²"
                    value={result.metrics.r2.toFixed(3)}
                    trainValue={result.train_metrics?.r2 != null ? result.train_metrics.r2.toFixed(3) : undefined}
                  />
                  <MetricCard
                    label="MAE"
                    value={result.metrics.mae.toFixed(3)}
                    trainValue={result.train_metrics?.mae != null ? result.train_metrics.mae.toFixed(3) : undefined}
                  />
                  <MetricCard
                    label="RMSE"
                    value={result.metrics.rmse.toFixed(3)}
                    trainValue={result.train_metrics?.rmse != null ? result.train_metrics.rmse.toFixed(3) : undefined}
                  />
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
                    {cluster.description && (
                      <p className="text-[12px] text-text-secondary mb-2 italic">{cluster.description}</p>
                    )}
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
          {isClassification && result.roc_curve && (() => {
            const showOverlay = overfitFixResult?.roc_curve != null;
            return (
            <div>
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-[12px] text-text-secondary">
                  ROC Curve{result.roc_curve.auc != null && ` (AUC = ${result.roc_curve.auc.toFixed(3)})`}
                  {showOverlay && overfitFixResult.roc_curve!.auc != null && (
                    <span className="text-emerald-600 dark:text-emerald-400"> vs regularized (AUC = {overfitFixResult.roc_curve!.auc.toFixed(3)})</span>
                  )}
                </h3>
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
                  <Line
                    data={result.roc_curve.fpr.map((fpr, i) => ({
                      fpr,
                      original: result.roc_curve!.tpr[i],
                    }))}
                    type="monotone"
                    dataKey="original"
                    name={`Original${result.roc_curve.auc != null ? ` (AUC ${result.roc_curve.auc.toFixed(3)})` : ""}`}
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    dot={false}
                  />
                  {showOverlay && (
                    <Line
                      data={overfitFixResult.roc_curve!.fpr.map((fpr, i) => ({
                        fpr,
                        regularized: overfitFixResult.roc_curve!.tpr[i],
                      }))}
                      type="monotone"
                      dataKey="regularized"
                      name={`Regularized${overfitFixResult.roc_curve!.auc != null ? ` (AUC ${overfitFixResult.roc_curve!.auc.toFixed(3)})` : ""}`}
                      stroke="#10b981"
                      strokeWidth={2}
                      strokeDasharray="6 3"
                      dot={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
              {showOverlay && (
                <div className="flex gap-4 mt-2">
                  <div className="flex items-center gap-1.5">
                    <div className="w-4 h-[2px] bg-[#8b5cf6] rounded-full" />
                    <span className="text-[11px] text-text-secondary">
                      Original{result.roc_curve.auc != null && <span className="text-text-tertiary ml-1">AUC {result.roc_curve.auc.toFixed(3)}</span>}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-4 h-[2px] bg-[#10b981] rounded-full" style={{ backgroundImage: "repeating-linear-gradient(90deg, #10b981 0 6px, transparent 6px 9px)" }} />
                    <span className="text-[11px] text-text-secondary">
                      Regularized{overfitFixResult.roc_curve!.auc != null && <span className="text-text-tertiary ml-1">AUC {overfitFixResult.roc_curve!.auc.toFixed(3)}</span>}
                    </span>
                  </div>
                </div>
              )}
            </div>
            );
          })()}

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

          {/* Next steps */}
          <NextSteps result={result} />
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
  const removeTrainingResult = useAppStore((s) => s.removeTrainingResult);
  const removeTrainingSession = useAppStore((s) => s.removeTrainingSession);
  const taskType = results[0]?.task;
  const isClassification = taskType === "classification";
  const isRegression = taskType === "regression";
  const isClustering = taskType === "clustering";
  const [chartHeight, chartSize, setChartSize] = useChartSize("M");
  // Check if any results have hyperparams worth showing
  const hasHyperparams = results.some((r) => r.hyperparams && Object.keys(r.hyperparams).length > 0);

  // ROC legend toggle: empty set = show all, otherwise show only selected indices
  const [visibleRocSet, setVisibleRocSet] = useState<Set<number>>(new Set());
  const toggleRocVisibility = (idx: number) => {
    setVisibleRocSet((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

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

  // Group results by session (preserving original indices for onSelect)
  const sessions: { sessionId: string; label: string; items: { result: TrainResult; globalIdx: number }[] }[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const sid = r.sessionId ?? `orphan-${i}`;
    if (!seen.has(sid)) {
      seen.add(sid);
      const sessionResults = results
        .map((res, idx) => ({ result: res, globalIdx: idx }))
        .filter((item) => (item.result.sessionId ?? `orphan-${item.globalIdx}`) === sid);
      const first = sessionResults[sessionResults.length - 1]?.result; // oldest in session (results are newest-first)
      const time = first?.trainedAt ? new Date(first.trainedAt) : null;
      const timeStr = time
        ? time.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
        : "";
      const featureCount = first?.features_used?.length ?? 0;
      const featurePreview = first?.features_used?.slice(0, 3).join(", ") ?? "";
      const more = featureCount > 3 ? ` +${featureCount - 3} more` : "";
      const label = `${timeStr}${timeStr ? " — " : ""}${featureCount} feature${featureCount !== 1 ? "s" : ""}${featurePreview ? ` (${featurePreview}${more})` : ""}`;
      sessions.push({ sessionId: sid, label, items: sessionResults });
    }
  }

  const hasSessions = sessions.length > 1 || (sessions.length === 1 && sessions[0].items.length > 1);

  // Count columns for session header colSpan
  let colCount = 3; // Algorithm + Features + actions
  if (isClassification) colCount += 4;
  if (isRegression) colCount += 3;
  if (isClustering) colCount += 2;
  if (results.some((r) => r.cv_scores)) colCount += 1;
  if (hasHyperparams) colCount += 1;

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
              {hasHyperparams && (
                <th className="px-3 py-2 text-left text-text-tertiary font-medium">Hyperparams</th>
              )}
              <th className="px-3 py-2 text-right text-text-tertiary font-medium">Features</th>
              <th className="px-3 py-2 w-[80px]" />
            </tr>
          </thead>
          <tbody>
            {sessions.map((session) => (
              <React.Fragment key={session.sessionId}>
                {/* Session header row */}
                {hasSessions && (
                  <tr className="bg-surface-alt/50">
                    <td colSpan={colCount} className="px-3 py-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-text-tertiary font-medium">
                          {session.label}
                        </span>
                        {session.items.length > 1 && !session.sessionId.startsWith("orphan-") && (
                          <button
                            onClick={() => removeTrainingSession(session.sessionId)}
                            className="text-[10px] text-text-tertiary hover:text-red-500 cursor-pointer transition-colors duration-150"
                            title="Remove all runs in this session"
                          >
                            Clear group
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
                {session.items.map(({ result: r, globalIdx: i }) => {
                  const isBest = i === bestIdx;
                  return (
                    <tr
                      key={i}
                      className={`border-b border-border/50 transition-colors duration-100 ${
                        isBest ? "bg-plume-50/50 dark:bg-plume-500/5" : "hover:bg-surface-alt"
                      }`}
                    >
                      <td className="px-3 py-2.5 font-medium text-text-primary">
                        <EditableNickname result={r} resultIndex={i} allResults={results} />
                        {isBest && (
                          <span className="ml-2 text-[9px] text-plume-600 dark:text-plume-400 font-normal">best</span>
                        )}
                      </td>
                      {isClassification && (
                        <>
                          <td className="px-3 py-2.5 text-right tabular-nums text-text-primary">
                            {(r.metrics.accuracy * 100).toFixed(1)}%
                            {r.train_metrics?.accuracy != null && (
                              <span className="block text-[10px] text-text-tertiary">train {(r.train_metrics.accuracy * 100).toFixed(1)}%</span>
                            )}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                            {(r.metrics.precision * 100).toFixed(1)}%
                            {r.train_metrics?.precision != null && (
                              <span className="block text-[10px] text-text-tertiary">train {(r.train_metrics.precision * 100).toFixed(1)}%</span>
                            )}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                            {(r.metrics.recall * 100).toFixed(1)}%
                            {r.train_metrics?.recall != null && (
                              <span className="block text-[10px] text-text-tertiary">train {(r.train_metrics.recall * 100).toFixed(1)}%</span>
                            )}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                            {(r.metrics.f1 * 100).toFixed(1)}%
                            {r.train_metrics?.f1 != null && (
                              <span className="block text-[10px] text-text-tertiary">train {(r.train_metrics.f1 * 100).toFixed(1)}%</span>
                            )}
                          </td>
                        </>
                      )}
                      {isRegression && (
                        <>
                          <td className="px-3 py-2.5 text-right tabular-nums text-text-primary">
                            {r.metrics.r2.toFixed(3)}
                            {r.train_metrics?.r2 != null && (
                              <span className="block text-[10px] text-text-tertiary">train {r.train_metrics.r2.toFixed(3)}</span>
                            )}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                            {r.metrics.mae.toFixed(3)}
                            {r.train_metrics?.mae != null && (
                              <span className="block text-[10px] text-text-tertiary">train {r.train_metrics.mae.toFixed(3)}</span>
                            )}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-text-secondary">
                            {r.metrics.rmse.toFixed(3)}
                            {r.train_metrics?.rmse != null && (
                              <span className="block text-[10px] text-text-tertiary">train {r.train_metrics.rmse.toFixed(3)}</span>
                            )}
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
                      {hasHyperparams && (
                        <td className="px-3 py-2.5 text-text-tertiary">
                          {r.hyperparams && Object.keys(r.hyperparams).length > 0 ? (
                            <span className="text-[10px] leading-relaxed">
                              {Object.entries(r.hyperparams).map(([k, v]) => (
                                <span key={k} className="inline-block mr-2">
                                  <span className="text-text-tertiary">{HYPERPARAM_SHORT_LABELS[k] ?? k}</span>{" "}
                                  <span className="text-text-secondary tabular-nums">{v}</span>
                                </span>
                              ))}
                            </span>
                          ) : (
                            <span className="text-[10px] text-text-tertiary">defaults</span>
                          )}
                        </td>
                      )}
                      <td className="px-3 py-2.5 text-right tabular-nums text-text-tertiary">
                        {r.features_used?.length ?? "—"}
                      </td>
                      <td className="px-3 py-2.5 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => onSelect(i)}
                            className="text-[11px] text-plume-600 dark:text-plume-500 hover:text-plume-700 cursor-pointer"
                          >
                            View
                          </button>
                          <button
                            onClick={() => removeTrainingResult(i)}
                            className="text-[11px] text-text-tertiary hover:text-red-500 cursor-pointer transition-colors duration-150"
                            title="Remove this run"
                          >
                            ×
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </React.Fragment>
            ))}
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
                const withRoc = results.filter((r) => r.roc_curve);
                const showAll = visibleRocSet.size === 0;
                return withRoc.map((r, i) => {
                    if (!showAll && !visibleRocSet.has(i)) return null;
                    const label = getRunLabel(r, results);
                    const dataKey = `run_${i}`;
                    const data = r.roc_curve!.fpr.map((fpr, j) => ({
                      fpr,
                      [dataKey]: r.roc_curve!.tpr[j],
                    }));
                    return (
                      <Line
                        key={`${dataKey}-${r.trainedAt ?? i}`}
                        data={data}
                        type="monotone"
                        dataKey={dataKey}
                        name={`${label}${r.roc_curve!.auc != null ? ` (AUC ${r.roc_curve!.auc.toFixed(3)})` : ""}`}
                        stroke={colors[i % colors.length]}
                        strokeWidth={2}
                        dot={false}
                      />
                    );
                  });
              })()}
            </LineChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 mt-2">
            {(() => {
              const colors = ["#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#10b981", "#ec4899", "#6366f1", "#84cc16"];
              const withRoc = results.filter((r) => r.roc_curve);
              const showAll = visibleRocSet.size === 0;
              return (
                <>
                  {withRoc.map((r, i) => {
                    const isVisible = showAll || visibleRocSet.has(i);
                    return (
                      <button
                        key={`${i}-${r.trainedAt}`}
                        onClick={() => toggleRocVisibility(i)}
                        className={`flex items-center gap-1.5 px-2 py-1 rounded-[var(--radius-default)] border transition-all duration-150 cursor-pointer ${
                          isVisible
                            ? "border-border bg-surface"
                            : "border-transparent opacity-40 hover:opacity-70"
                        }`}
                      >
                        <div
                          className="w-3 h-[2px] rounded-full shrink-0"
                          style={{ backgroundColor: colors[i % colors.length] }}
                        />
                        <span className="text-[11px] text-text-secondary">
                          {getRunLabel(r, results)}
                          {r.roc_curve!.auc != null && (
                            <span className="text-text-tertiary ml-1">AUC {r.roc_curve!.auc.toFixed(3)}</span>
                          )}
                        </span>
                      </button>
                    );
                  })}
                  {visibleRocSet.size > 0 && (
                    <button
                      onClick={() => setVisibleRocSet(new Set())}
                      className="text-[10px] text-text-tertiary hover:text-text-primary cursor-pointer px-1"
                    >
                      Show all
                    </button>
                  )}
                </>
              );
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

/** Algorithm-specific regularization levels for overfitting (mild → strong) */
interface RegLevel {
  label: string;
  hyperparams: Record<string, number>;
  description: string;
}

const OVERFIT_RECOMMENDATIONS: Record<string, {
  tips: string[];
  levels: RegLevel[];
}> = {
  random_forest: {
    tips: [
      "Reduce max depth to prevent trees from memorizing individual examples",
      "Increase min samples to split so each decision requires more evidence",
      "Use fewer trees — more trees rarely cause overfitting, but shallower ones generalize better",
    ],
    levels: [
      { label: "Mild", hyperparams: { n_estimators: 100, max_depth: 12, min_samples_split: 5 }, description: "Depth 12, min split 5" },
      { label: "Moderate", hyperparams: { n_estimators: 100, max_depth: 8, min_samples_split: 10 }, description: "Depth 8, min split 10" },
      { label: "Strong", hyperparams: { n_estimators: 80, max_depth: 5, min_samples_split: 20 }, description: "Depth 5, min split 20" },
      { label: "Very strong", hyperparams: { n_estimators: 60, max_depth: 3, min_samples_split: 30 }, description: "Depth 3, min split 30" },
    ],
  },
  xgboost: {
    tips: [
      "Reduce max depth — XGBoost trees are often too deep by default",
      "Lower the learning rate and use more trees for gentler learning",
      "Reduce the number of trees to stop the model from over-fitting to residuals",
    ],
    levels: [
      { label: "Mild", hyperparams: { n_estimators: 200, max_depth: 5, learning_rate: 0.08 }, description: "Depth 5, lr 0.08" },
      { label: "Moderate", hyperparams: { n_estimators: 200, max_depth: 4, learning_rate: 0.05 }, description: "Depth 4, lr 0.05" },
      { label: "Strong", hyperparams: { n_estimators: 150, max_depth: 3, learning_rate: 0.03 }, description: "Depth 3, lr 0.03" },
      { label: "Very strong", hyperparams: { n_estimators: 100, max_depth: 2, learning_rate: 0.01 }, description: "Depth 2, lr 0.01" },
    ],
  },
  lightgbm: {
    tips: [
      "Reduce num_leaves — this is the primary complexity control in LightGBM",
      "Lower the learning rate and increase the number of trees",
      "Set a max depth limit to prevent overly specific leaf nodes",
    ],
    levels: [
      { label: "Mild", hyperparams: { n_estimators: 200, max_depth: 8, learning_rate: 0.08, num_leaves: 25 }, description: "Leaves 25, depth 8, lr 0.08" },
      { label: "Moderate", hyperparams: { n_estimators: 200, max_depth: 6, learning_rate: 0.05, num_leaves: 20 }, description: "Leaves 20, depth 6, lr 0.05" },
      { label: "Strong", hyperparams: { n_estimators: 150, max_depth: 4, learning_rate: 0.03, num_leaves: 15 }, description: "Leaves 15, depth 4, lr 0.03" },
      { label: "Very strong", hyperparams: { n_estimators: 100, max_depth: 3, learning_rate: 0.01, num_leaves: 10 }, description: "Leaves 10, depth 3, lr 0.01" },
    ],
  },
  logistic_regression: {
    tips: [
      "Decrease C to apply stronger regularization (smaller C = more regularization)",
      "Consider reducing the number of features — too many can cause overfitting in linear models",
    ],
    levels: [
      { label: "Mild", hyperparams: { C: 0.5, max_iter: 1000 }, description: "C=0.5" },
      { label: "Moderate", hyperparams: { C: 0.1, max_iter: 1000 }, description: "C=0.1" },
      { label: "Strong", hyperparams: { C: 0.01, max_iter: 1000 }, description: "C=0.01" },
    ],
  },
  linear_regression: {
    tips: [
      "Linear regression has no built-in regularization — consider switching to a different algorithm",
      "Remove features that may be leaking information or are highly correlated",
    ],
    levels: [],
  },
};

function OverfitWarning({
  trainPct,
  testPct,
  metric,
  gap,
  result,
  onFixApplied,
}: {
  trainPct: string;
  testPct: string;
  metric: string;
  gap: number;
  result: TrainResult;
  onFixApplied: (fixResult: TrainResult) => void;
}) {
  const [retraining, setRetraining] = useState(false);
  const [attempts, setAttempts] = useState<{ level: RegLevel; result: TrainResult; gap: number }[]>([]);
  const [bestResult, setBestResult] = useState<TrainResult | null>(null);
  const addTrainingResult = useAppStore((s) => s.addTrainingResult);
  const trainingResults = useAppStore((s) => s.trainingResults);
  const recs = OVERFIT_RECOMMENDATIONS[result.algorithm];

  const severity = gap >= 0.25 ? "severe" : gap >= 0.15 ? "moderate" : "mild";
  const isClassification = result.task === "classification";

  const getGap = (r: TrainResult) => {
    const trainScore = isClassification ? r.train_metrics?.accuracy : r.train_metrics?.r2;
    const testScore = isClassification ? r.metrics.accuracy : r.metrics.r2;
    return (trainScore ?? 0) - (testScore ?? 0);
  };

  const getTestScore = (r: TrainResult) => {
    return isClassification ? r.metrics.accuracy : r.metrics.r2;
  };

  const handleAutoRegularize = async () => {
    if (!recs || recs.levels.length === 0) return;
    setRetraining(true);
    setAttempts([]);
    setBestResult(null);

    const algoLabel = ALGO_LABELS[result.algorithm] ?? result.algorithm;
    const allAttempts: { level: RegLevel; result: TrainResult; gap: number }[] = [];
    let best: TrainResult | null = null;
    let bestTestScore = getTestScore(result);
    let prevGap = gap;

    for (const level of recs.levels) {
      try {
        const newResult = await invoke<TrainResult>("train_model", {
          task: result.task,
          target: result.target ?? null,
          features: result.features_used ?? [],
          algorithm: result.algorithm,
          nClusters: null,
          hyperparams: level.hyperparams,
          useCv: !!result.cv_scores,
          cvFolds: result.cv_scores?.folds ?? 5,
          positiveClass: result.positive_class ?? null,
        });
        newResult.target = result.target;
        newResult.hyperparams = level.hyperparams;
        newResult.positive_class = result.positive_class;

        const newGap = getGap(newResult);
        const newTestScore = getTestScore(newResult);

        allAttempts.push({ level, result: newResult, gap: newGap });
        setAttempts([...allAttempts]);

        // Track the best result (lowest gap while maintaining reasonable test performance)
        if (newTestScore >= bestTestScore * 0.95 && newGap < prevGap) {
          best = newResult;
          bestTestScore = newTestScore;
          prevGap = newGap;
        }

        // Stop if gap is resolved or getting worse
        if (newGap < 0.05) break;
        if (newGap > prevGap + 0.02) break; // getting worse, stop
      } catch (err) {
        console.error(`Regularization level ${level.label} failed:`, err);
      }
    }

    // Add the best result to training results
    if (best) {
      const regPattern = new RegExp(`^${algoLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} \\(regularized`);
      const existingCount = trainingResults.filter((r) => r.nickname && regPattern.test(r.nickname)).length;
      best.nickname = existingCount === 0
        ? `${algoLabel} (regularized)`
        : `${algoLabel} (regularized ${existingCount + 1})`;
      best.sessionId = result.sessionId ?? `session-${Date.now()}`;
      addTrainingResult(best);
      onFixApplied(best);
      setBestResult(best);
    } else if (allAttempts.length > 0) {
      // No improvement — still add the mildest attempt for comparison
      const mildest = allAttempts[0].result;
      const regPattern = new RegExp(`^${algoLabel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} \\(regularized`);
      const existingCount = trainingResults.filter((r) => r.nickname && regPattern.test(r.nickname)).length;
      mildest.nickname = existingCount === 0
        ? `${algoLabel} (regularized)`
        : `${algoLabel} (regularized ${existingCount + 1})`;
      mildest.sessionId = result.sessionId ?? `session-${Date.now()}`;
      addTrainingResult(mildest);
      onFixApplied(mildest);
      setBestResult(mildest);
    }

    setRetraining(false);
  };

  const fixResult = bestResult;

  // Build before/after comparison data
  const getComparison = () => {
    if (!fixResult) return null;

    const pctFmt = (v: number) => `${(v * 100).toFixed(1)}%`;
    const decFmt = (v: number) => v.toFixed(3);

    const oldTrainScore = isClassification ? result.train_metrics!.accuracy : result.train_metrics!.r2;
    const oldTestScore = isClassification ? result.metrics.accuracy : result.metrics.r2;
    const newTrainScore = isClassification ? fixResult.train_metrics?.accuracy : fixResult.train_metrics?.r2;
    const newTestScore = isClassification ? fixResult.metrics.accuracy : fixResult.metrics.r2;
    const oldGap = oldTrainScore - oldTestScore;
    const newGap = (newTrainScore ?? 0) - (newTestScore ?? 0);

    type Row = { label: string; oldVal: number; newVal: number; format: (v: number) => string; higherIsBetter: boolean };
    const rows: Row[] = [];

    if (isClassification) {
      rows.push(
        { label: "Accuracy", oldVal: result.metrics.accuracy, newVal: fixResult.metrics.accuracy, format: pctFmt, higherIsBetter: true },
        { label: "Precision", oldVal: result.metrics.precision, newVal: fixResult.metrics.precision, format: pctFmt, higherIsBetter: true },
        { label: "Recall", oldVal: result.metrics.recall, newVal: fixResult.metrics.recall, format: pctFmt, higherIsBetter: true },
        { label: "F1", oldVal: result.metrics.f1, newVal: fixResult.metrics.f1, format: pctFmt, higherIsBetter: true },
      );
    } else {
      rows.push(
        { label: "R²", oldVal: result.metrics.r2, newVal: fixResult.metrics.r2, format: decFmt, higherIsBetter: true },
        { label: "MAE", oldVal: result.metrics.mae, newVal: fixResult.metrics.mae, format: decFmt, higherIsBetter: false },
        { label: "RMSE", oldVal: result.metrics.rmse, newVal: fixResult.metrics.rmse, format: decFmt, higherIsBetter: false },
      );
    }

    return { rows, oldGap, newGap };
  };

  return (
    <div className="bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-700 rounded-[var(--radius-default)] px-4 py-3">
      <p className="text-[12px] font-medium text-amber-700 dark:text-amber-400 mb-1">
        {severity === "severe" ? "Significant" : severity === "moderate" ? "Moderate" : "Mild"} overfitting detected
      </p>
      <p className="text-[12px] text-amber-700/80 dark:text-amber-300/80 leading-relaxed mb-3">
        Your model scores {trainPct}% {metric} on training data but only {testPct}% on new data.
        {severity === "severe"
          ? " This is a large gap — the model is heavily memorizing training examples."
          : severity === "moderate"
            ? " The model is learning some noise from the training data along with the real patterns."
            : " There's a small gap between training and test performance."
        }
      </p>

      {/* Algorithm-specific recommendations */}
      {recs && (
        <div className="mb-3">
          <p className="text-[11px] font-medium text-amber-700 dark:text-amber-400 mb-1.5">
            Recommendations for {ALGO_LABELS[result.algorithm] ?? result.algorithm}:
          </p>
          <ul className="flex flex-col gap-1">
            {recs.tips.map((tip, i) => (
              <li key={i} className="text-[11px] text-amber-700/80 dark:text-amber-300/80 leading-relaxed flex gap-1.5">
                <span className="shrink-0 mt-px">•</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Auto-regularize button */}
      {recs && recs.levels.length > 0 && !fixResult && (
        <div className="pt-2 border-t border-amber-200 dark:border-amber-700/50">
          <div className="flex items-center gap-3">
            <button
              onClick={handleAutoRegularize}
              disabled={retraining}
              className="px-3 py-1.5 text-[11px] font-medium rounded-[var(--radius-default)] bg-amber-600 text-white hover:bg-amber-700 disabled:opacity-50 transition-colors duration-200 cursor-pointer"
            >
              {retraining ? (
                <span className="flex items-center gap-1.5">
                  <motion.span
                    className="inline-block w-1.5 h-1.5 rounded-full bg-white/60"
                    animate={{ scale: [1, 1.3, 1] }}
                    transition={{ duration: 1, repeat: Infinity }}
                  />
                  Testing regularization levels ({attempts.length}/{recs.levels.length})...
                </span>
              ) : (
                "Auto-regularize"
              )}
            </button>
            <span className="text-[10px] text-amber-700/60 dark:text-amber-300/60">
              Tries {recs.levels.length} levels of regularization and picks the best
            </span>
          </div>
          {/* Live progress during iteration */}
          {retraining && attempts.length > 0 && (
            <div className="mt-2 flex flex-col gap-1">
              {attempts.map((a, i) => {
                const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
                return (
                  <div key={i} className="flex items-center gap-2 text-[10px]">
                    <span className="text-amber-700/60 dark:text-amber-300/60 w-[70px] shrink-0">{a.level.label}</span>
                    <span className="text-text-secondary tabular-nums">gap {pct(a.gap)}</span>
                    <span className="text-text-tertiary">·</span>
                    <span className="text-text-tertiary tabular-nums">{a.level.description}</span>
                    {a.gap < 0.1 && <span className="text-emerald-600 dark:text-emerald-400 font-medium">resolved</span>}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Summary of all attempts */}
      {!retraining && attempts.length > 1 && fixResult && (
        <div className="pt-2 mt-1 border-t border-amber-200 dark:border-amber-700/50 mb-2">
          <p className="text-[10px] text-text-tertiary mb-1.5">Regularization levels tested:</p>
          <div className="flex flex-col gap-1">
            {attempts.map((a, i) => {
              const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
              const testScore = getTestScore(a.result);
              const isBest = a.result === fixResult;
              return (
                <div key={i} className={`flex items-center gap-2 text-[10px] ${isBest ? "font-medium" : ""}`}>
                  <span className={`w-[70px] shrink-0 ${isBest ? "text-plume-600 dark:text-plume-400" : "text-text-tertiary"}`}>
                    {a.level.label}
                  </span>
                  <span className="text-text-secondary tabular-nums">gap {pct(a.gap)}</span>
                  <span className="text-text-tertiary">·</span>
                  <span className="text-text-secondary tabular-nums">
                    {isClassification ? "accuracy" : "R²"} {isClassification ? pct(testScore) : testScore.toFixed(3)}
                  </span>
                  {isBest && <span className="text-plume-600 dark:text-plume-400">← best</span>}
                  {a.gap < 0.1 && !isBest && <span className="text-emerald-600/60">resolved</span>}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Inline before/after comparison */}
      {fixResult && (() => {
        const comparison = getComparison();
        if (!comparison) return null;
        const { rows, oldGap, newGap } = comparison;
        const gapImproved = newGap < oldGap;
        const gapFixed = newGap < 0.1;
        const pctFmt = (v: number) => `${(v * 100).toFixed(1)}%`;

        return (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="pt-3 mt-1 border-t border-amber-200 dark:border-amber-700/50"
          >
            {/* Overfit gap comparison */}
            <div className="flex items-center gap-3 mb-3">
              <div className="flex-1">
                <p className="text-[10px] text-text-tertiary mb-1">Train/test gap</p>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] text-text-tertiary">Before</span>
                    <span className="text-[13px] font-semibold text-amber-700 dark:text-amber-400 tabular-nums">{pctFmt(oldGap)}</span>
                  </div>
                  <span className="text-text-tertiary">→</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] text-text-tertiary">After</span>
                    <span className={`text-[13px] font-semibold tabular-nums ${
                      gapFixed
                        ? "text-emerald-600 dark:text-emerald-400"
                        : gapImproved
                          ? "text-amber-600 dark:text-amber-400"
                          : "text-red-600 dark:text-red-400"
                    }`}>
                      {pctFmt(newGap)}
                    </span>
                  </div>
                  {gapFixed && (
                    <span className="text-[10px] font-medium text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/10 px-1.5 py-0.5 rounded">
                      Resolved
                    </span>
                  )}
                  {!gapFixed && gapImproved && (
                    <span className="text-[10px] font-medium text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-500/10 px-1.5 py-0.5 rounded">
                      Improved
                    </span>
                  )}
                  {!gapImproved && (
                    <span className="text-[10px] font-medium text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-500/10 px-1.5 py-0.5 rounded">
                      No improvement
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Metric-by-metric comparison */}
            <p className="text-[10px] text-text-tertiary mb-1.5">Test performance (before → after regularization)</p>
            <div className="flex flex-col gap-1">
              {rows.map((m) => {
                const diff = m.newVal - m.oldVal;
                const improved = m.higherIsBetter ? diff > 0 : diff < 0;
                const unchanged = Math.abs(diff) < 0.0001;
                return (
                  <div key={m.label} className="flex items-center gap-2 text-[11px]">
                    <span className="w-[60px] text-text-tertiary text-right shrink-0">{m.label}</span>
                    <span className="text-text-secondary tabular-nums">{m.format(m.oldVal)}</span>
                    <span className="text-text-tertiary">→</span>
                    <span className="text-text-primary font-medium tabular-nums">{m.format(m.newVal)}</span>
                    <span className={`text-[10px] tabular-nums font-medium ${
                      unchanged
                        ? "text-text-tertiary"
                        : improved
                          ? "text-emerald-600 dark:text-emerald-400"
                          : "text-red-500 dark:text-red-400"
                    }`}>
                      {unchanged ? "—" : improved ? "↑" : "↓"}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Verdict */}
            <p className={`text-[11px] mt-2.5 font-medium leading-relaxed ${
              gapFixed
                ? "text-emerald-700 dark:text-emerald-400"
                : gapImproved
                  ? "text-amber-700 dark:text-amber-400"
                  : "text-red-700 dark:text-red-400"
            }`}>
              {gapFixed
                ? "Regularization resolved the overfitting. The new model generalizes well to unseen data."
                : gapImproved
                  ? "Regularization reduced the gap but some overfitting remains. You could try the Tune & Retrain panel to adjust further."
                  : "Regularization didn't help here. Consider removing noisy features, adding more data, or trying a simpler algorithm."
              }
            </p>
          </motion.div>
        );
      })()}
    </div>
  );
}

interface NextStep {
  text: string;
  priority: "high" | "medium" | "low";
  action?: "drop-low-importance";
  features?: string[];
}

function getNextSteps(result: TrainResult): NextStep[] {
  const steps: NextStep[] = [];
  const isClassification = result.task === "classification";
  const isRegression = result.task === "regression";
  const isClustering = result.task === "clustering";

  // High priority suggestions
  if (result.leakage_warnings?.length) {
    steps.push({ text: "Remove leaked features and retrain — your current metrics may be unrealistically high.", priority: "high" });
  }
  if (result.imbalance_warning && result.imbalance_warning.minority_pct < 10) {
    steps.push({ text: "Your minority class is very small. Consider collecting more data for underrepresented classes.", priority: "high" });
  }

  // Performance-based suggestions
  if (isClassification && result.metrics.accuracy < 0.7) {
    steps.push({ text: "Accuracy is below 70%. Try adding more features, engineering new ones in the Shape tab, or switching algorithms.", priority: "high" });
  }
  if (isRegression && result.metrics.r2 < 0.5) {
    steps.push({ text: "R² is below 0.5 — the model explains less than half the variation. Try different features or a non-linear algorithm like Random Forest.", priority: "high" });
  }

  // Medium priority
  if (!result.cv_scores) {
    steps.push({ text: "Enable cross-validation for a more reliable performance estimate — a single train/test split can be lucky or unlucky.", priority: "medium" });
  }
  if (result.feature_importance?.length && result.feature_importance.length > 5) {
    const lowImportance = result.feature_importance.filter((f) => f.importance < 0.01);
    if (lowImportance.length > 0) {
      steps.push({
        text: `${lowImportance.length} feature${lowImportance.length > 1 ? "s have" : " has"} near-zero importance. Removing them may simplify the model without losing accuracy.`,
        priority: "medium",
        action: "drop-low-importance",
        features: lowImportance.map((f) => f.feature),
      });
    }
  }

  // Overfitting-related (only if not already shown in overfit warning)
  if (result.train_metrics) {
    const trainScore = isClassification ? result.train_metrics.accuracy : result.train_metrics.r2;
    const testScore = isClassification ? result.metrics.accuracy : result.metrics.r2;
    if (trainScore != null && testScore != null && trainScore - testScore > 0.05 && trainScore - testScore < 0.1) {
      steps.push({ text: "There's a small gap between training and test performance. Cross-validation can help confirm whether this is a concern.", priority: "low" });
    }
  }

  // Clustering
  if (isClustering && result.metrics.silhouette != null && result.metrics.silhouette < 0.25) {
    steps.push({ text: "The silhouette score is low, suggesting clusters aren't well-separated. Try different numbers of groups or standardize your features in the Shape tab.", priority: "medium" });
  }

  // Good results encouragement
  if (isClassification && result.metrics.accuracy >= 0.95) {
    steps.push({ text: "Accuracy is very high — double-check for data leakage. If the data is clean, this is a strong model.", priority: "low" });
  }

  return steps;
}

function NextSteps({ result }: { result: TrainResult }) {
  const steps = getNextSteps(result).slice(0, 5);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);
  const [retraining, setRetraining] = useState(false);
  const addTrainingResult = useAppStore((s) => s.addTrainingResult);

  if (steps.length === 0) return null;

  const dotColor = {
    high: "bg-red-500",
    medium: "bg-amber-400",
    low: "bg-gray-400 dark:bg-gray-500",
  };

  const handleDropAndRetrain = async (featuresToDrop: string[]) => {
    if (!result.features_used) return;
    setRetraining(true);
    try {
      const keptFeatures = result.features_used.filter((f) => !featuresToDrop.includes(f));
      const newResult = await invoke<TrainResult>("train_model", {
        task: result.task,
        target: result.target ?? null,
        features: keptFeatures,
        algorithm: result.algorithm,
        nClusters: null,
        hyperparams: result.hyperparams ?? null,
        useCv: !!result.cv_scores,
        cvFolds: result.cv_scores?.folds ?? 5,
        positiveClass: result.positive_class ?? null,
      });
      newResult.target = result.target;
      newResult.hyperparams = result.hyperparams;
      newResult.positive_class = result.positive_class;
      newResult.sessionId = `session-${Date.now()}`;
      addTrainingResult(newResult);
    } catch (err) {
      console.error("Retrain failed:", err);
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div>
      <h3 className="text-[12px] text-text-secondary mb-2">Next steps</h3>
      <div className="flex flex-col gap-1.5">
        {steps.map((step, i) => (
          <div key={i} className="flex flex-col">
            <div
              className={`flex items-start gap-2 ${step.action ? "cursor-pointer" : ""}`}
              onClick={step.action ? () => setExpandedStep(expandedStep === i ? null : i) : undefined}
            >
              <span className={`mt-1.5 w-[6px] h-[6px] rounded-full shrink-0 ${dotColor[step.priority]}`} />
              <span className="text-[11px] text-text-secondary leading-relaxed">
                {step.text}
                {step.action && (
                  <span className="ml-1 text-plume-600 dark:text-plume-400 hover:underline">
                    {expandedStep === i ? "Hide" : "Show features"}
                  </span>
                )}
              </span>
            </div>

            {/* Expandable detail for drop-low-importance */}
            <AnimatePresence>
              {step.action === "drop-low-importance" && expandedStep === i && step.features && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="ml-[14px] mt-2 p-2.5 rounded-[var(--radius-default)] border border-border bg-surface-secondary">
                    <p className="text-[11px] text-text-tertiary mb-2">
                      {step.features.length === 1 ? "This feature has" : "These features have"} less than 1% importance:
                    </p>
                    <div className="flex flex-wrap gap-1.5 mb-2.5">
                      {step.features.map((f) => (
                        <span
                          key={f}
                          className="px-2 py-0.5 text-[10px] font-medium rounded-full border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-300"
                        >
                          {f}
                        </span>
                      ))}
                    </div>
                    <button
                      onClick={() => handleDropAndRetrain(step.features!)}
                      disabled={retraining}
                      className="px-3 py-1.5 text-[11px] font-medium rounded-[var(--radius-default)] bg-plume-600 text-white hover:bg-plume-700 disabled:opacity-50 transition-colors duration-200 cursor-pointer"
                    >
                      {retraining ? (
                        <span className="flex items-center gap-1.5">
                          <motion.span
                            className="inline-block w-1.5 h-1.5 rounded-full bg-white/60"
                            animate={{ scale: [1, 1.3, 1] }}
                            transition={{ duration: 1, repeat: Infinity }}
                          />
                          Retraining...
                        </span>
                      ) : (
                        `Drop ${step.features.length === 1 ? "feature" : `${step.features.length} features`} & retrain`
                      )}
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value, trainValue }: { label: string; value: string; trainValue?: string }) {
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
      {trainValue && (
        <p className="text-[10px] text-text-tertiary mt-1 tabular-nums">
          <span className="text-text-tertiary/70">train</span> {trainValue}
        </p>
      )}
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
      // Try to compute SHAP for the first supervised result
      let shapData = null;
      const supervisedResult = results.find((r) => r.task !== "clustering" && r.target);
      if (supervisedResult) {
        try {
          const resp = await invoke<{ explanations: ShapExplanation[] }>("compute_shap", {
            task: supervisedResult.task,
            target: supervisedResult.target ?? null,
            features: supervisedResult.features_used ?? [],
            algorithm: supervisedResult.algorithm,
            hyperparams: supervisedResult.hyperparams ?? {},
            nSamples: 3,
          });
          shapData = resp.explanations;
        } catch {
          // SHAP not available — export without it
        }
      }

      await invoke("generate_report", {
        results: results.map((r) => ({
          task: r.task,
          algorithm: r.algorithm,
          nickname: r.nickname ?? null,
          label: getRunLabel(r, results),
          metrics: r.metrics,
          train_metrics: r.train_metrics,
          feature_importance: r.feature_importance,
          features_used: r.features_used,
          train_size: r.train_size,
          test_size: r.test_size,
          cv_scores: r.cv_scores,
          positive_class: r.positive_class ?? null,
          target: r.target ?? null,
          hyperparams: r.hyperparams ?? null,
        })),
        shapData,
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

const HYPERPARAM_DEFS: Record<string, { key: string; label: string; tooltip: string; default: number; min: number; max: number; step: number }[]> = {
  random_forest: [
    { key: "n_estimators", label: "Trees", tooltip: "How many decision trees to build. More trees usually means better accuracy but slower training.", default: 100, min: 10, max: 1000, step: 10 },
    { key: "max_depth", label: "Max depth (0 = unlimited)", tooltip: "How many levels deep each tree can grow. Deeper trees capture more detail but may memorize your data instead of learning patterns.", default: 0, min: 0, max: 100, step: 1 },
    { key: "min_samples_split", label: "Min samples to split", tooltip: "Minimum number of examples needed before a tree can make a new decision. Higher values make the model more conservative.", default: 2, min: 2, max: 50, step: 1 },
  ],
  logistic_regression: [
    { key: "C", label: "Regularization (C)", tooltip: "Controls how much the model is allowed to fit the training data. Lower values prevent overfitting by keeping the model simpler.", default: 1.0, min: 0.01, max: 100, step: 0.1 },
    { key: "max_iter", label: "Max iterations", tooltip: "Maximum number of times the algorithm cycles through the data to find the best fit.", default: 1000, min: 100, max: 10000, step: 100 },
  ],
  linear_regression: [],
  xgboost: [
    { key: "n_estimators", label: "Trees", tooltip: "How many decision trees to build. More trees usually means better accuracy but slower training.", default: 100, min: 10, max: 1000, step: 10 },
    { key: "max_depth", label: "Max depth (0 = unlimited)", tooltip: "How many levels deep each tree can grow. Deeper trees capture more detail but may memorize your data instead of learning patterns.", default: 6, min: 0, max: 20, step: 1 },
    { key: "learning_rate", label: "Learning rate", tooltip: "How much each new tree corrects the previous ones. Smaller values learn more carefully but need more trees to compensate.", default: 0.1, min: 0.01, max: 1, step: 0.01 },
  ],
  lightgbm: [
    { key: "n_estimators", label: "Trees", tooltip: "How many decision trees to build. More trees usually means better accuracy but slower training.", default: 100, min: 10, max: 1000, step: 10 },
    { key: "max_depth", label: "Max depth (0 = unlimited)", tooltip: "How many levels deep each tree can grow. Deeper trees capture more detail but may memorize your data instead of learning patterns.", default: -1, min: -1, max: 50, step: 1 },
    { key: "learning_rate", label: "Learning rate", tooltip: "How much each new tree corrects the previous ones. Smaller values learn more carefully but need more trees to compensate.", default: 0.1, min: 0.01, max: 1, step: 0.01 },
    { key: "num_leaves", label: "Num leaves", tooltip: "Maximum number of leaf nodes per tree. More leaves capture more detail but increase the risk of memorizing data.", default: 31, min: 2, max: 256, step: 1 },
  ],
};

function RetrainPanel({ result, onRetrained }: { result: TrainResult; onRetrained: () => void }) {
  const [open, setOpen] = useState(false);
  const [training, setTraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<TrainResult | null>(null);
  // Capture the original result at first render so comparisons stay stable after selectedIdx changes
  const [originalResult] = useState<TrainResult>(result);
  const addTrainingResult = useAppStore((s) => s.addTrainingResult);
  const [hpTooltip, setHpTooltip] = useState<string | null>(null);
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
        positiveClass: originalResult.positive_class ?? null,
      });
      newResult.target = originalResult.target;
      newResult.hyperparams = built;
      newResult.positive_class = originalResult.positive_class;
      newResult.sessionId = originalResult.sessionId ?? `session-${Date.now()}`;
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
                    <label className="text-[10px] text-text-tertiary">
                      {def.label}
                      <span className="relative inline-block ml-1">
                        <span
                          className="text-text-tertiary/60 cursor-help hover:text-plume-500 transition-colors duration-150"
                          onMouseEnter={() => setHpTooltip(def.key)}
                          onMouseLeave={() => setHpTooltip(null)}
                        >
                          ?
                        </span>
                        {hpTooltip === def.key && (
                          <span className="absolute z-10 bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-[220px] px-3 py-2 text-[11px] leading-relaxed text-text-primary bg-surface border border-border rounded-[var(--radius-default)] shadow-md pointer-events-none">
                            {def.tooltip}
                          </span>
                        )}
                      </span>
                    </label>
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
