import { useState, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
import type { ColumnProfile, TrainResult } from "../types/data";

type Task = "classification" | "regression" | "clustering" | null;

const ALGO_OPTIONS: Record<string, { value: string; label: string }[]> = {
  classification: [
    { value: "random_forest", label: "Random Forest" },
    { value: "logistic_regression", label: "Logistic Regression" },
    { value: "xgboost", label: "XGBoost" },
    { value: "lightgbm", label: "LightGBM" },
  ],
  regression: [
    { value: "random_forest", label: "Random Forest" },
    { value: "linear_regression", label: "Linear Regression" },
    { value: "xgboost", label: "XGBoost" },
    { value: "lightgbm", label: "LightGBM" },
  ],
  clustering: [
    { value: "kmeans", label: "K-Means" },
    { value: "dbscan", label: "DBSCAN" },
    { value: "hierarchical", label: "Hierarchical" },
  ],
};

interface HyperparamDef {
  key: string;
  label: string;
  default: number;
  min: number;
  max: number;
  step: number;
}

const HYPERPARAMS: Record<string, HyperparamDef[]> = {
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
  kmeans: [],
  dbscan: [
    { key: "eps", label: "Epsilon", default: 0.5, min: 0.01, max: 10, step: 0.05 },
    { key: "min_samples", label: "Min samples", default: 5, min: 1, max: 50, step: 1 },
  ],
  hierarchical: [],
};

export function ModelView() {
  const summary = useAppStore((s) => s.summary);
  const [task, setTask] = useState<Task>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [algorithm, setAlgorithm] = useState<string>("random_forest");
  const [nClusters, setNClusters] = useState(3);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [training, setTraining] = useState(false);
  const [hideRecommendations, setHideRecommendations] = useState(false);
  const [hyperparams, setHyperparams] = useState<Record<string, number>>({});
  const [useCv, setUseCv] = useState(false);
  const [cvFolds, setCvFolds] = useState(5);
  const [comparing, setComparing] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);

  const [profiles, setProfiles] = useState<ColumnProfile[]>([]);
  const [typeFilter, setTypeFilter] = useState<string | null>(null);

  useEffect(() => {
    if (summary) {
      invoke<ColumnProfile[]>("get_all_column_profiles").then(setProfiles).catch(console.error);
    }
  }, [summary]);

  const columns = useMemo(() => summary?.column_names ?? [], [summary]);
  const columnTypes = useMemo(() => summary?.column_types ?? [], [summary]);

  const availableFeatures = useMemo(() => {
    if (!target) return columns;
    return columns.filter((c) => c !== target);
  }, [columns, target]);

  // Distinct column types for the filter chips
  const distinctTypes = useMemo(() => {
    const types = new Set(columnTypes);
    return Array.from(types).sort();
  }, [columnTypes]);

  // Map column name → type for quick lookup
  const columnTypeMap = useMemo(() => {
    const map = new Map<string, string>();
    columns.forEach((col, i) => { map.set(col, columnTypes[i]); });
    return map;
  }, [columns, columnTypes]);

  // Type-friendly labels
  const typeLabel = (t: string) => {
    const labels: Record<string, string> = { str: "Text", f64: "Decimal", i64: "Integer", bool: "Boolean", f32: "Decimal", i32: "Integer", u8: "Integer", u16: "Integer", u32: "Integer", u64: "Integer", cat: "Category" };
    return labels[t] ?? t;
  };

  // Detect ID-like columns (high cardinality relative to row count)
  const idLikeColumns = useMemo(() => {
    if (!summary || profiles.length === 0) return new Set<string>();
    const threshold = Math.min(summary.rows * 0.9, summary.rows - 1);
    return new Set(
      profiles
        .filter((p) => p.unique_count != null && p.unique_count >= threshold)
        .map((p) => p.name)
    );
  }, [profiles, summary]);

  // Warn if all selected features are ID-like
  const featureWarning = useMemo(() => {
    if (selectedFeatures.length === 0) return null;
    const idFeatures = selectedFeatures.filter((f) => idLikeColumns.has(f));
    if (idFeatures.length === selectedFeatures.length) {
      return "All selected features look like ID columns (very high cardinality). Your model may memorize row identifiers instead of learning patterns.";
    }
    if (idFeatures.length > 0) {
      return `${idFeatures.join(", ")} ${idFeatures.length === 1 ? "looks" : "look"} like ID ${idFeatures.length === 1 ? "column" : "columns"} and may hurt model performance.`;
    }
    return null;
  }, [selectedFeatures, idLikeColumns]);

  // Recommend target columns based on task type
  const recommendedTargets = useMemo(() => {
    if (!task || task === "clustering" || profiles.length === 0 || !summary) return new Set<string>();
    return new Set(
      profiles
        .filter((p) => {
          if (task === "classification") {
            // Good classification targets: low unique count, non-numeric or categorical
            return p.unique_count != null && p.unique_count >= 2 && p.unique_count <= 20;
          }
          // Regression: numeric columns with many unique values
          return (
            (p.dtype === "f64" || p.dtype === "i64" || p.dtype === "f32" || p.dtype === "i32") &&
            p.unique_count != null &&
            p.unique_count > 10
          );
        })
        .map((p) => p.name)
    );
  }, [task, profiles, summary]);

  const handleTaskSelect = (t: Task) => {
    setTask(t);
    setTarget(null);
    setShowAdvanced(false);
    setHideRecommendations(false);
    setHyperparams({});
    setTypeFilter(null);
    setAlgorithm(t === "clustering" ? "kmeans" : "random_forest");
    // For clustering, auto-select all non-ID features since there's no target
    if (t === "clustering") {
      setSelectedFeatures(columns.filter((c) => !idLikeColumns.has(c)));
    } else {
      setSelectedFeatures([]);
    }
  };

  const handleTargetSelect = (col: string) => {
    if (target === col) {
      setTarget(null);
      setSelectedFeatures([]);
      return;
    }
    setTarget(col);
    // Auto-select all features except the target and ID-like columns
    const autoFeatures = columns.filter((c) => c !== col && !idLikeColumns.has(c));
    setSelectedFeatures(autoFeatures);
  };

  const toggleFeature = (col: string) => {
    setSelectedFeatures((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const selectAllFeatures = () => {
    setSelectedFeatures(availableFeatures);
  };

  const deselectAllFeatures = () => {
    setSelectedFeatures([]);
  };

  const addTrainingResult = useAppStore((s) => s.addTrainingResult);

  const buildHyperparams = () => {
    const defs = HYPERPARAMS[algorithm] || [];
    if (defs.length === 0) return undefined;
    const hp: Record<string, number> = {};
    for (const def of defs) {
      hp[def.key] = hyperparams[def.key] ?? def.default;
    }
    return hp;
  };

  const formatTrainError = (err: unknown): string => {
    const msg = String(err);
    // Extract the useful part from Python tracebacks
    const lines = msg.split("\n");
    const lastLine = lines.filter((l) => l.trim()).pop() ?? msg;
    if (lastLine.length > 200) return lastLine.slice(0, 200) + "...";
    return lastLine;
  };

  const handleTrain = async () => {
    setTraining(true);
    setTrainError(null);
    try {
      const result = await invoke<TrainResult>("train_model", {
        task,
        target,
        features: selectedFeatures,
        algorithm,
        nClusters: task === "clustering" ? nClusters : null,
        hyperparams: buildHyperparams(),
        useCv: useCv,
        cvFolds: cvFolds,
      });

      result.target = target;
      result.hyperparams = buildHyperparams();
      addTrainingResult(result);
      useAppStore.setState({ view: "results" });
    } catch (err) {
      console.error("Training failed:", err);
      setTrainError(formatTrainError(err));
    } finally {
      setTraining(false);
    }
  };

  const handleCompareAll = async () => {
    if (!task) return;
    setComparing(true);
    setTrainError(null);
    const algos = ALGO_OPTIONS[task];
    const failures: string[] = [];
    for (const algo of algos) {
      try {
        const result = await invoke<TrainResult>("train_model", {
          task,
          target,
          features: selectedFeatures,
          algorithm: algo.value,
          nClusters: task === "clustering" ? nClusters : null,
          hyperparams: undefined,
          useCv: useCv,
          cvFolds: cvFolds,
        });
        result.target = target;
        addTrainingResult(result);
      } catch (err) {
        console.error(`${algo.label} failed:`, err);
        failures.push(algo.label);
      }
    }
    setComparing(false);
    if (failures.length > 0 && failures.length < algos.length) {
      setTrainError(`${failures.join(", ")} failed. Other models trained successfully.`);
    } else if (failures.length === algos.length) {
      setTrainError("All models failed to train. Check your data and feature selection.");
      return;
    }
    useAppStore.setState({ view: "results" });
  };

  if (!summary) return null;

  // Step 1: Pick task
  if (!task) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="flex gap-4"
        >
          <TaskCard
            title="Predict a Category"
            description="Is it A or B?"
            onClick={() => handleTaskSelect("classification")}
          />
          <TaskCard
            title="Predict a Number"
            description="How much? How many?"
            onClick={() => handleTaskSelect("regression")}
          />
          <TaskCard
            title="Find Groups"
            description="What patterns exist?"
            onClick={() => handleTaskSelect("clustering")}
          />
        </motion.div>
      </div>
    );
  }

  const isSupervised = task !== "clustering";
  const canTrain = selectedFeatures.length > 0 && (isSupervised ? target !== null : true);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Back + task label */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-border">
        <button
          onClick={() => handleTaskSelect(null)}
          className="text-[12px] text-text-tertiary hover:text-text-primary transition-colors duration-200 cursor-pointer"
        >
          ← Back
        </button>
        <span className="text-[13px] font-medium text-text-primary">
          {task === "classification" && "Predict a Category"}
          {task === "regression" && "Predict a Number"}
          {task === "clustering" && "Find Groups"}
        </span>
      </div>

      <div className="flex-1 overflow-auto">
        <div className="max-w-[600px] mx-auto py-6 px-6">
          {/* Target selection (supervised only) */}
          {isSupervised && (
            <div className="mb-8">
              <div className="flex items-center justify-between mb-1">
                <label className="text-[12px] text-text-secondary">
                  What do you want to predict?
                </label>
                <div className="flex gap-3">
                  {!hideRecommendations && recommendedTargets.size > 0 && (
                    <button
                      onClick={() => setHideRecommendations(true)}
                      className="text-[11px] text-text-tertiary hover:text-text-primary cursor-pointer"
                    >
                      Hide suggestions
                    </button>
                  )}
                  {(target || selectedFeatures.length > 0) && (
                    <button
                      onClick={() => { setTarget(null); setSelectedFeatures([]); }}
                      className="text-[11px] text-text-tertiary hover:text-text-primary cursor-pointer"
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>
              {distinctTypes.length > 1 && (
                <div className="flex gap-1.5 mb-2 flex-wrap">
                  <button
                    onClick={() => setTypeFilter(null)}
                    className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors duration-150 cursor-pointer ${
                      typeFilter === null
                        ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
                        : "border-border text-text-tertiary hover:border-text-tertiary"
                    }`}
                  >
                    All types
                  </button>
                  {distinctTypes.map((t) => (
                    <button
                      key={t}
                      onClick={() => setTypeFilter(typeFilter === t ? null : t)}
                      className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors duration-150 cursor-pointer ${
                        typeFilter === t
                          ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
                          : "border-border text-text-tertiary hover:border-text-tertiary"
                      }`}
                    >
                      {typeLabel(t)}
                    </button>
                  ))}
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                {columns.filter((_, i) => !typeFilter || columnTypes[i] === typeFilter).map((col) => {
                  const isSelected = target === col;
                  const colType = columnTypeMap.get(col) ?? "";
                  const showAsRecommended = !hideRecommendations && recommendedTargets.has(col);
                  // Dim numeric columns for classification, string columns for regression
                  const isDimmed =
                    (task === "classification" && (colType === "f64" || colType === "f32")) ||
                    (task === "regression" && colType === "str");
                  return (
                    <button
                      key={col}
                      onClick={() => handleTargetSelect(col)}
                      className={`
                        px-3 py-1.5 text-[12px] rounded-[var(--radius-default)] border
                        transition-all duration-200 cursor-pointer
                        ${isSelected
                          ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
                          : isDimmed
                            ? "border-border/50 text-text-tertiary opacity-50"
                            : showAsRecommended
                              ? "border-emerald-300 dark:border-emerald-700 text-text-secondary hover:border-emerald-500"
                              : "border-border text-text-secondary hover:border-text-tertiary"
                        }
                      `}
                    >
                      {col}
                      {showAsRecommended && !isSelected && (
                        <span className="ml-1.5 text-[9px] text-emerald-600 dark:text-emerald-400">recommended</span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Feature selection */}
          {(isSupervised ? target : true) && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="mb-8"
            >
              <div className="flex items-center justify-between mb-2">
                <label className="text-[12px] text-text-secondary">
                  {isSupervised ? "Using these columns" : "Columns to analyze"}
                </label>
                <div className="flex gap-2">
                  <button onClick={selectAllFeatures} className="text-[11px] text-plume-600 dark:text-plume-500 hover:text-plume-700 cursor-pointer">All</button>
                  <button onClick={deselectAllFeatures} className="text-[11px] text-text-tertiary hover:text-text-primary cursor-pointer">None</button>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {availableFeatures.filter((col) => !typeFilter || columnTypeMap.get(col) === typeFilter).map((col) => {
                  const isSelected = selectedFeatures.includes(col);
                  const isIdLike = idLikeColumns.has(col);
                  return (
                    <button
                      key={col}
                      onClick={() => toggleFeature(col)}
                      className={`
                        px-3 py-1.5 text-[12px] rounded-[var(--radius-default)] border
                        transition-all duration-200 cursor-pointer
                        ${isSelected
                          ? isIdLike
                            ? "border-amber-400 dark:border-amber-600 bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400"
                            : "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
                          : "border-border text-text-tertiary hover:border-text-tertiary hover:text-text-secondary"
                        }
                      `}
                    >
                      {col}
                      {isIdLike && isSelected && (
                        <span className="ml-1.5 text-[9px] text-amber-600 dark:text-amber-400">ID?</span>
                      )}
                    </button>
                  );
                })}
              </div>
              {featureWarning && (
                <p className="text-[11px] text-amber-600 dark:text-amber-400 mt-2">
                  {featureWarning}
                </p>
              )}
              {selectedFeatures.length > 0 && idLikeColumns.size > 0 && selectedFeatures.every((f) => !idLikeColumns.has(f)) && (
                <p className="text-[11px] text-text-tertiary mt-2">
                  {idLikeColumns.size} ID-like {idLikeColumns.size === 1 ? "column was" : "columns were"} excluded automatically. You can add them back if needed.
                </p>
              )}
            </motion.div>
          )}

          {/* Clustering: n_clusters */}
          {task === "clustering" && algorithm !== "dbscan" && (
            <div className="mb-8">
              <label className="text-[12px] text-text-secondary mb-2 block">
                Number of groups
              </label>
              <input
                type="number"
                min={2}
                max={20}
                value={nClusters}
                onChange={(e) => setNClusters(parseInt(e.target.value) || 3)}
                className="w-[80px] px-3 py-1.5 text-[13px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200"
              />
            </div>
          )}

          {/* Advanced: algorithm picker + hyperparams + CV */}
          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2, ease: "easeOut" }}
                className="overflow-hidden mb-6"
              >
                <label className="text-[12px] text-text-secondary mb-2 block">Algorithm</label>
                <div className="flex flex-wrap gap-2 mb-4">
                  {ALGO_OPTIONS[task].map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => { setAlgorithm(opt.value); setHyperparams({}); }}
                      className={`
                        px-3 py-1.5 text-[12px] rounded-[var(--radius-default)] border
                        transition-all duration-200 cursor-pointer
                        ${algorithm === opt.value
                          ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
                          : "border-border text-text-secondary hover:border-text-tertiary"
                        }
                      `}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>

                {/* Hyperparameters */}
                {(HYPERPARAMS[algorithm] || []).length > 0 && (
                  <div className="mb-4">
                    <label className="text-[12px] text-text-secondary mb-2 block">Hyperparameters</label>
                    <div className="grid grid-cols-2 gap-3">
                      {(HYPERPARAMS[algorithm] || []).map((def) => (
                        <div key={def.key} className="flex flex-col gap-1">
                          <label className="text-[10px] text-text-tertiary">{def.label}</label>
                          <input
                            type="number"
                            min={def.min}
                            max={def.max}
                            step={def.step}
                            value={hyperparams[def.key] ?? def.default}
                            onChange={(e) =>
                              setHyperparams((prev) => ({
                                ...prev,
                                [def.key]: parseFloat(e.target.value) || def.default,
                              }))
                            }
                            className="w-full px-2 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200 tabular-nums"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Cross-validation */}
                {isSupervised && (
                  <div className="mb-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={useCv}
                        onChange={(e) => setUseCv(e.target.checked)}
                        className="accent-plume-500"
                      />
                      <span className="text-[12px] text-text-secondary">Cross-validation</span>
                    </label>
                    <p className="text-[11px] text-text-tertiary mt-1 ml-5 leading-relaxed max-w-[440px]">
                      Instead of testing your model once, cross-validation trains and tests it multiple times on different slices of your data. This gives a more reliable estimate of how well the model will perform on new data.
                    </p>
                    {useCv && (
                      <div className="flex items-center gap-2 mt-2 ml-5">
                        <label className="text-[10px] text-text-tertiary">Folds</label>
                        <input
                          type="number"
                          min={2}
                          max={20}
                          value={cvFolds}
                          onChange={(e) => setCvFolds(parseInt(e.target.value) || 5)}
                          className="w-[60px] px-2 py-1 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200"
                        />
                        <span className="text-[11px] text-text-tertiary">
                          ({cvFolds} rounds of train-and-test)
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Train button + customize link + compare */}
          {canTrain && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="flex flex-col items-start gap-3"
            >
              <div className="flex items-center gap-3">
                <button
                  onClick={handleTrain}
                  disabled={training || comparing}
                  className="px-6 py-2.5 text-[13px] font-medium rounded-[var(--radius-default)] bg-plume-600 text-white hover:bg-plume-700 disabled:opacity-50 transition-colors duration-200 cursor-pointer"
                >
                  {training ? (
                    <span className="flex items-center gap-2">
                      <motion.span
                        className="inline-block w-2 h-2 rounded-full bg-white/60"
                        animate={{ scale: [1, 1.3, 1] }}
                        transition={{ duration: 1, repeat: Infinity }}
                      />
                      Training...
                    </span>
                  ) : (
                    task === "clustering" ? "Find Groups" : "Train"
                  )}
                </button>

                {showAdvanced && (
                  <button
                    onClick={handleCompareAll}
                    disabled={training || comparing}
                    className="px-4 py-2.5 text-[13px] font-medium rounded-[var(--radius-default)] border border-plume-500 text-plume-600 dark:text-plume-400 hover:bg-plume-50 dark:hover:bg-plume-500/10 disabled:opacity-50 transition-colors duration-200 cursor-pointer"
                  >
                    {comparing ? (
                      <span className="flex items-center gap-2">
                        <motion.span
                          className="inline-block w-2 h-2 rounded-full bg-plume-500/60"
                          animate={{ scale: [1, 1.3, 1] }}
                          transition={{ duration: 1, repeat: Infinity }}
                        />
                        Comparing...
                      </span>
                    ) : (
                      "Compare all"
                    )}
                  </button>
                )}
              </div>

              {!showAdvanced && (
                <button
                  onClick={() => setShowAdvanced(true)}
                  className="text-[11px] text-text-tertiary hover:text-text-secondary transition-colors duration-200 cursor-pointer"
                >
                  Customize
                </button>
              )}
            </motion.div>
          )}

          {trainError && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-3 rounded-[var(--radius-default)] border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-900/20"
            >
              <p className="text-[12px] text-red-700 dark:text-red-300">{trainError}</p>
              <button
                onClick={() => setTrainError(null)}
                className="text-[11px] text-red-500 hover:text-red-700 mt-1 cursor-pointer"
              >
                Dismiss
              </button>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

function TaskCard({
  title,
  description,
  onClick,
}: {
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <motion.button
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="w-[180px] h-[140px] flex flex-col items-center justify-center gap-2 rounded-[var(--radius-default)] border border-border bg-surface hover:border-plume-500/50 hover:shadow-sm transition-colors duration-200 cursor-pointer"
    >
      <span className="text-[14px] font-medium text-text-primary">{title}</span>
      <span className="text-[12px] text-text-tertiary">{description}</span>
    </motion.button>
  );
}
