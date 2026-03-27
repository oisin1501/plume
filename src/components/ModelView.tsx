import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
import type { AutoTuneResult, ColumnProfile, TrainResult } from "../types/data";

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
  tooltip: string;
  default: number;
  min: number;
  max: number;
  step: number;
}

const HYPERPARAMS: Record<string, HyperparamDef[]> = {
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
  kmeans: [],
  dbscan: [
    { key: "eps", label: "Epsilon", tooltip: "How close points must be to count as neighbors. Smaller values find tighter, more compact groups.", default: 0.5, min: 0.01, max: 10, step: 0.05 },
    { key: "min_samples", label: "Min samples", tooltip: "Minimum number of nearby points required to form the core of a group. Higher values ignore smaller groups as noise.", default: 5, min: 1, max: 50, step: 1 },
  ],
  hierarchical: [],
};

export function ModelView() {
  const summary = useAppStore((s) => s.summary);
  const trainingResults = useAppStore((s) => s.trainingResults);
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
  const [positiveClass, setPositiveClass] = useState<string | null>(null);
  const [targetClasses, setTargetClasses] = useState<string[]>([]);
  const [showRecall, setShowRecall] = useState(false);
  const recallRef = useRef<HTMLDivElement>(null);
  const [comparing, setComparing] = useState(false);
  const [compareProgress, setCompareProgress] = useState<{ current: number; total: number; label: string } | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);

  const [profiles, setProfiles] = useState<ColumnProfile[]>([]);
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [featureSearch, setFeatureSearch] = useState("");
  const [useRegex, setUseRegex] = useState(false);
  const [smartFilter, setSmartFilter] = useState<string | null>(null);
  const lastClickedFeature = useRef<string | null>(null);
  const [hpTooltip, setHpTooltip] = useState<string | null>(null);
  const [autoTuning, setAutoTuning] = useState(false);
  const [autoTuneData, setAutoTuneData] = useState<AutoTuneResult | null>(null);
  const [autoTuneError, setAutoTuneError] = useState<string | null>(null);

  // Close recall dropdown on outside click
  useEffect(() => {
    if (!showRecall) return;
    const handler = (e: MouseEvent) => {
      if (recallRef.current && !recallRef.current.contains(e.target as Node)) {
        setShowRecall(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showRecall]);

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

  // Detect low-variance columns (single unique value = constant)
  const lowVarianceColumns = useMemo(() => {
    if (profiles.length === 0) return new Set<string>();
    return new Set(
      profiles
        .filter((p) => p.unique_count != null && p.unique_count <= 1)
        .map((p) => p.name)
    );
  }, [profiles]);

  // Detect high-null columns (>50% missing)
  const highNullColumns = useMemo(() => {
    if (profiles.length === 0) return new Set<string>();
    return new Set(
      profiles
        .filter((p) => p.null_percent > 50)
        .map((p) => p.name)
    );
  }, [profiles]);

  // Smart filter sets for quick lookup
  const smartFilterSets: Record<string, Set<string>> = useMemo(() => ({
    id: idLikeColumns,
    low_variance: lowVarianceColumns,
    high_nulls: highNullColumns,
  }), [idLikeColumns, lowVarianceColumns, highNullColumns]);

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

  // Algorithm recommendation based on dataset properties
  const recommendation = useMemo<{ algorithm: string; reason: string } | null>(() => {
    if (!task || !summary) return null;
    const rows = summary.rows;
    const featureCount = availableFeatures.length;

    if (task === "classification") {
      // Check if most features are categorical
      const catCount = availableFeatures.filter((col) => {
        const t = columnTypeMap.get(col);
        return t === "str" || t === "cat";
      }).length;
      const mostlyCategorical = featureCount > 0 && catCount > featureCount * 0.5;

      if (rows >= 10000) {
        return { algorithm: "lightgbm", reason: mostlyCategorical
          ? "Tree-based algorithms handle categorical data naturally."
          : "LightGBM handles large datasets efficiently and often achieves the best accuracy." };
      }
      if (rows >= 1000) {
        return { algorithm: mostlyCategorical ? "random_forest" : "random_forest", reason: mostlyCategorical
          ? "Tree-based algorithms handle categorical data naturally."
          : "Random Forest is a reliable all-rounder for medium-sized datasets." };
      }
      // rows < 1000
      if (mostlyCategorical) {
        return { algorithm: "random_forest", reason: "Tree-based algorithms handle categorical data naturally." };
      }
      if (featureCount > rows * 0.5) {
        return { algorithm: "logistic_regression", reason: "When you have many features relative to rows, simpler models avoid memorizing noise." };
      }
      return { algorithm: "logistic_regression", reason: "With a smaller dataset, Logistic Regression is less likely to overfit and gives interpretable results." };
    }

    if (task === "regression") {
      if (rows >= 10000) {
        return { algorithm: "lightgbm", reason: "LightGBM is fast and powerful for larger datasets." };
      }
      if (rows >= 1000) {
        return { algorithm: "random_forest", reason: "Random Forest captures non-linear patterns well for medium-sized datasets." };
      }
      return { algorithm: "linear_regression", reason: "With a smaller dataset, Linear Regression avoids overfitting and is easy to interpret." };
    }

    if (task === "clustering") {
      if (nClusters === 3) {
        // Default value — user likely hasn't set it deliberately
        return { algorithm: "dbscan", reason: "DBSCAN automatically finds the number of groups and handles irregular shapes." };
      }
      return { algorithm: "kmeans", reason: "K-Means is a fast, reliable choice when you know roughly how many groups to expect." };
    }

    return null;
  }, [task, summary, availableFeatures, columnTypeMap, nClusters]);

  // Get the display label for a recommended algorithm
  const recommendedAlgoLabel = useMemo(() => {
    if (!recommendation || !task) return "";
    const options = ALGO_OPTIONS[task];
    return options?.find((o) => o.value === recommendation.algorithm)?.label ?? recommendation.algorithm;
  }, [recommendation, task]);

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
    // For clustering, auto-select all non-ID features since there's no target
    if (t === "clustering") {
      setSelectedFeatures(columns.filter((c) => !idLikeColumns.has(c)));
    } else {
      setSelectedFeatures([]);
    }
  };

  // Set default algorithm from recommendation whenever it changes
  useEffect(() => {
    if (recommendation) {
      setAlgorithm(recommendation.algorithm);
    }
  }, [recommendation]);

  const handleTargetSelect = (col: string) => {
    if (target === col) {
      setTarget(null);
      setSelectedFeatures([]);
      setTargetClasses([]);
      setPositiveClass(null);
      return;
    }
    setTarget(col);
    setPositiveClass(null);
    setTargetClasses([]);
    // Auto-select all features except the target and ID-like columns
    const autoFeatures = columns.filter((c) => c !== col && !idLikeColumns.has(c));
    setSelectedFeatures(autoFeatures);
    // Fetch unique classes for binary classification positive class picker
    if (task === "classification") {
      invoke<{ labels: string[]; counts: number[] }>("get_column_distribution", { columnName: col })
        .then((dist) => {
          if (dist.labels.length === 2) {
            setTargetClasses(dist.labels);
          }
        })
        .catch(console.error);
    }
  };

  const toggleFeature = (col: string) => {
    setSelectedFeatures((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  // Track visible features for shift+click (set from render)
  const visibleFeaturesRef = useRef<string[]>([]);

  // Drag-select: click and drag across feature buttons to select/deselect in bulk
  const dragMode = useRef<"select" | "deselect" | null>(null);
  const dragTouched = useRef<Set<string>>(new Set());

  const handleFeaturePointerDown = useCallback((col: string, e: React.PointerEvent) => {
    // Shift+click: range select between last clicked and current
    if (e.shiftKey && lastClickedFeature.current) {
      const visible = visibleFeaturesRef.current;
      const lastIdx = visible.indexOf(lastClickedFeature.current);
      const curIdx = visible.indexOf(col);
      if (lastIdx !== -1 && curIdx !== -1) {
        const start = Math.min(lastIdx, curIdx);
        const end = Math.max(lastIdx, curIdx);
        const range = visible.slice(start, end + 1);
        setSelectedFeatures((prev) => {
          const prevSet = new Set(prev);
          for (const c of range) prevSet.add(c);
          return Array.from(prevSet);
        });
        lastClickedFeature.current = col;
        return;
      }
    }

    const isSelected = selectedFeatures.includes(col);
    dragMode.current = isSelected ? "deselect" : "select";
    dragTouched.current = new Set([col]);
    toggleFeature(col);
    lastClickedFeature.current = col;
  }, [selectedFeatures, toggleFeature]);

  const handleFeaturePointerEnter = useCallback((col: string) => {
    if (dragMode.current === null) return;
    if (dragTouched.current.has(col)) return;
    dragTouched.current.add(col);
    const isSelected = selectedFeatures.includes(col);
    if (dragMode.current === "select" && !isSelected) {
      setSelectedFeatures((prev) => [...prev, col]);
    } else if (dragMode.current === "deselect" && isSelected) {
      setSelectedFeatures((prev) => prev.filter((c) => c !== col));
    }
  }, [selectedFeatures]);

  useEffect(() => {
    const handlePointerUp = () => { dragMode.current = null; };
    window.addEventListener("pointerup", handlePointerUp);
    return () => window.removeEventListener("pointerup", handlePointerUp);
  }, []);

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
    const sessionId = `session-${Date.now()}`;
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
        positiveClass: positiveClass,
      });

      result.target = target;
      result.hyperparams = buildHyperparams();
      result.sessionId = sessionId;
      result.positive_class = positiveClass;
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
    const sessionId = `session-${Date.now()}`;
    const algos = ALGO_OPTIONS[task];
    const failures: string[] = [];
    for (let i = 0; i < algos.length; i++) {
      const algo = algos[i];
      setCompareProgress({ current: i + 1, total: algos.length, label: algo.label });
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
          positiveClass: positiveClass,
        });
        result.target = target;
        result.sessionId = sessionId;
        result.positive_class = positiveClass;
        addTrainingResult(result);
      } catch (err) {
        console.error(`${algo.label} failed:`, err);
        failures.push(algo.label);
      }
    }
    setComparing(false);
    setCompareProgress(null);
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

          {/* Positive class picker (binary classification only) */}
          {task === "classification" && target && targetClasses.length === 2 && (
            <div className="mb-6 -mt-4">
              <label className="text-[11px] text-text-tertiary mb-1.5 block">
                Which class do you want to detect? (positive class)
              </label>
              <div className="flex gap-2">
                {targetClasses.map((cls) => (
                  <button
                    key={cls}
                    onClick={() => setPositiveClass(positiveClass === cls ? null : cls)}
                    className={`
                      px-3 py-1.5 text-[12px] rounded-[var(--radius-default)] border
                      transition-all duration-200 cursor-pointer
                      ${positiveClass === cls
                        ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10 text-plume-700 dark:text-plume-400"
                        : "border-border text-text-secondary hover:border-text-tertiary"
                      }
                    `}
                  >
                    {cls}
                  </button>
                ))}
              </div>
              {!positiveClass && (
                <p className="text-[10px] text-text-tertiary mt-1">
                  Optional — affects how precision, recall, and F1 are calculated. If not set, Plume picks automatically.
                </p>
              )}
            </div>
          )}

          {/* Feature selection */}
          {(isSupervised ? target : true) && (() => {
            const searchLower = featureSearch.toLowerCase();
            let searchRegex: RegExp | null = null;
            if (useRegex && featureSearch) {
              try { searchRegex = new RegExp(featureSearch, "i"); } catch { /* invalid regex, fall through */ }
            }
            const activeSmartSet = smartFilter ? smartFilterSets[smartFilter] : null;
            const visibleFeatures = availableFeatures
              .filter((col) => !typeFilter || columnTypeMap.get(col) === typeFilter)
              .filter((col) => {
                if (!featureSearch) return true;
                if (searchRegex) return searchRegex.test(col);
                return col.toLowerCase().includes(searchLower);
              })
              .filter((col) => !activeSmartSet || activeSmartSet.has(col));
            visibleFeaturesRef.current = visibleFeatures;
            return (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="mb-8"
            >
              <div className="flex items-center justify-between mb-1">
                <label className="text-[12px] text-text-secondary">
                  {isSupervised ? "Using these columns" : "Columns to analyze"}
                  <span className="ml-2 text-text-tertiary font-normal">
                    {selectedFeatures.length} of {availableFeatures.length} selected
                  </span>
                </label>
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      const toAdd = visibleFeatures.filter((c) => !selectedFeatures.includes(c));
                      if (toAdd.length > 0) setSelectedFeatures((prev) => [...prev, ...toAdd]);
                    }}
                    className="text-[11px] text-plume-600 dark:text-plume-500 hover:text-plume-700 cursor-pointer"
                  >
                    {featureSearch || typeFilter || smartFilter ? "Select visible" : "All"}
                  </button>
                  <button
                    onClick={() => {
                      if (featureSearch || typeFilter || smartFilter) {
                        const visibleSet = new Set(visibleFeatures);
                        setSelectedFeatures((prev) => prev.filter((c) => !visibleSet.has(c)));
                      } else {
                        deselectAllFeatures();
                      }
                    }}
                    className="text-[11px] text-text-tertiary hover:text-text-primary cursor-pointer"
                  >
                    {featureSearch || typeFilter || smartFilter ? "Deselect visible" : "None"}
                  </button>
                  <button
                    onClick={() => {
                      const selectedSet = new Set(selectedFeatures);
                      setSelectedFeatures(availableFeatures.filter((c) => !selectedSet.has(c)));
                    }}
                    className="text-[11px] text-text-tertiary hover:text-text-primary cursor-pointer"
                  >
                    Invert
                  </button>
                  {trainingResults.length > 0 && (
                    <div className="relative" ref={recallRef}>
                      <button
                        onClick={() => setShowRecall(!showRecall)}
                        className="text-[11px] text-plume-600 dark:text-plume-500 hover:text-plume-700 cursor-pointer"
                      >
                        Recall
                      </button>
                      <AnimatePresence>
                        {showRecall && (
                          <motion.div
                            initial={{ opacity: 0, y: -4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -4 }}
                            transition={{ duration: 0.15 }}
                            className="absolute right-0 top-6 z-50 w-[280px] max-h-[240px] overflow-y-auto bg-surface border border-border rounded-[var(--radius-default)] shadow-lg p-1.5"
                          >
                            {(() => {
                              // Deduplicate by feature set — show unique combinations
                              const seen = new Map<string, { label: string; features: string[] }>();
                              for (const r of trainingResults) {
                                if (!r.features_used?.length) continue;
                                const key = [...r.features_used].sort().join(",");
                                if (seen.has(key)) continue;
                                const algoEntry = Object.values(ALGO_OPTIONS).flat().find((a) => a.value === r.algorithm);
                                const algo = algoEntry?.label ?? r.algorithm;
                                const time = r.trainedAt ? new Date(r.trainedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "";
                                const label = `${algo} — ${r.features_used.length} features${time ? ` (${time})` : ""}`;
                                seen.set(key, { label, features: r.features_used });
                              }
                              const options = Array.from(seen.values());
                              if (options.length === 0) {
                                return (
                                  <p className="text-[11px] text-text-tertiary px-2 py-1.5">No previous feature sets</p>
                                );
                              }
                              return options.map((opt, i) => (
                                <button
                                  key={i}
                                  onClick={() => {
                                    // Only select features that still exist in the current dataset
                                    const available = new Set(availableFeatures);
                                    setSelectedFeatures(opt.features.filter((f) => available.has(f)));
                                    setShowRecall(false);
                                  }}
                                  className="w-full text-left px-2.5 py-2 text-[11px] rounded hover:bg-surface-alt transition-colors duration-100 cursor-pointer"
                                >
                                  <span className="text-text-primary font-medium">{opt.label}</span>
                                  <span className="block text-[10px] text-text-tertiary mt-0.5 truncate">
                                    {opt.features.slice(0, 5).join(", ")}
                                    {opt.features.length > 5 && ` +${opt.features.length - 5} more`}
                                  </span>
                                </button>
                              ));
                            })()}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}
                </div>
              </div>
              {availableFeatures.length > 15 && (
                <div className="relative mb-2">
                  <input
                    type="text"
                    placeholder={useRegex ? "Regex pattern e.g. ^feat_\\d+" : "Search columns..."}
                    value={featureSearch}
                    onChange={(e) => setFeatureSearch(e.target.value)}
                    className={`w-full px-3 py-1.5 pr-14 text-[12px] border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none transition-colors duration-200 ${
                      useRegex && featureSearch && (() => { try { new RegExp(featureSearch); return false; } catch { return true; } })()
                        ? "border-red-400 focus:border-red-500"
                        : "border-border focus:border-plume-500"
                    }`}
                  />
                  <button
                    onClick={() => setUseRegex(!useRegex)}
                    className={`absolute right-2 top-1/2 -translate-y-1/2 px-1.5 py-0.5 text-[10px] rounded font-mono cursor-pointer transition-colors duration-150 ${
                      useRegex
                        ? "bg-plume-100 dark:bg-plume-500/20 text-plume-700 dark:text-plume-400"
                        : "text-text-tertiary hover:text-text-secondary"
                    }`}
                    title="Toggle regex matching"
                  >
                    .*
                  </button>
                </div>
              )}
              {/* Smart filter chips */}
              {profiles.length > 0 && (
                <div className="flex gap-1.5 mb-2 flex-wrap">
                  {idLikeColumns.size > 0 && (
                    <button
                      onClick={() => setSmartFilter(smartFilter === "id" ? null : "id")}
                      className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors duration-150 cursor-pointer ${
                        smartFilter === "id"
                          ? "border-amber-400 bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400"
                          : "border-border text-text-tertiary hover:border-text-tertiary"
                      }`}
                    >
                      ID-like ({idLikeColumns.size})
                    </button>
                  )}
                  {lowVarianceColumns.size > 0 && (
                    <button
                      onClick={() => setSmartFilter(smartFilter === "low_variance" ? null : "low_variance")}
                      className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors duration-150 cursor-pointer ${
                        smartFilter === "low_variance"
                          ? "border-amber-400 bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400"
                          : "border-border text-text-tertiary hover:border-text-tertiary"
                      }`}
                    >
                      Constant ({lowVarianceColumns.size})
                    </button>
                  )}
                  {highNullColumns.size > 0 && (
                    <button
                      onClick={() => setSmartFilter(smartFilter === "high_nulls" ? null : "high_nulls")}
                      className={`px-2 py-0.5 text-[10px] rounded-full border transition-colors duration-150 cursor-pointer ${
                        smartFilter === "high_nulls"
                          ? "border-amber-400 bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400"
                          : "border-border text-text-tertiary hover:border-text-tertiary"
                      }`}
                    >
                      High nulls ({highNullColumns.size})
                    </button>
                  )}
                </div>
              )}
              <div
                className="flex flex-wrap gap-2 max-h-[240px] overflow-auto select-none"
                style={{ touchAction: "none" }}
              >
                {visibleFeatures.map((col) => {
                  const isSelected = selectedFeatures.includes(col);
                  const isIdLike = idLikeColumns.has(col);
                  return (
                    <button
                      key={col}
                      onPointerDown={(e) => { e.preventDefault(); handleFeaturePointerDown(col, e); }}
                      onPointerEnter={() => handleFeaturePointerEnter(col)}
                      className={`
                        px-3 py-1.5 text-[12px] rounded-[var(--radius-default)] border
                        transition-all duration-200 cursor-pointer select-none
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
                {visibleFeatures.length === 0 && featureSearch && (
                  <p className="text-[11px] text-text-tertiary py-2">No columns match "{featureSearch}"</p>
                )}
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
            );
          })()}

          {/* Algorithm suggestion (shown before Customize is opened) */}
          {canTrain && !showAdvanced && recommendation && (
            <div className="mb-4">
              <p className="text-[11px] text-text-tertiary">
                Suggested starting point: <span className="text-plume-600 dark:text-plume-400">{recommendedAlgoLabel}</span> — {recommendation.reason}
              </p>
            </div>
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
                      {recommendation && recommendation.algorithm === opt.value && (
                        <span className="ml-1.5 text-[9px] text-emerald-600 dark:text-emerald-400">suggested</span>
                      )}
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
                    {/* Auto-tune button */}
                    {isSupervised && algorithm !== "linear_regression" && (
                      <div className="col-span-2 mt-1">
                        <div className="flex items-center gap-3">
                          <button
                            onClick={async () => {
                              setAutoTuning(true);
                              setAutoTuneData(null);
                              setAutoTuneError(null);
                              try {
                                const result = await invoke<AutoTuneResult>("auto_tune", {
                                  task,
                                  target,
                                  features: selectedFeatures,
                                  algorithm,
                                  useCv: true,
                                  cvFolds: cvFolds,
                                });
                                setHyperparams(result.best_hyperparams);
                                setAutoTuneData(result);
                              } catch (err) {
                                setAutoTuneError(String(err).split("\n").pop() ?? String(err));
                              } finally {
                                setAutoTuning(false);
                              }
                            }}
                            disabled={autoTuning || training || comparing || selectedFeatures.length === 0}
                            className="px-3 py-1.5 text-[11px] font-medium rounded-[var(--radius-default)] border border-plume-500 text-plume-600 dark:text-plume-400 hover:bg-plume-50 dark:hover:bg-plume-500/10 disabled:opacity-50 transition-colors duration-200 cursor-pointer"
                          >
                            {autoTuning ? (
                              <span className="flex items-center gap-1.5">
                                <motion.span
                                  className="inline-block w-1.5 h-1.5 rounded-full bg-plume-500/60"
                                  animate={{ scale: [1, 1.3, 1] }}
                                  transition={{ duration: 1, repeat: Infinity }}
                                />
                                Finding best settings...
                              </span>
                            ) : (
                              "Find best settings"
                            )}
                          </button>
                          {autoTuneError && (
                            <span className="text-[10px] text-red-500">{autoTuneError}</span>
                          )}
                        </div>

                        {/* Auto-tune results summary */}
                        {autoTuneData && (
                          <motion.div
                            initial={{ opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.2 }}
                            className="mt-3 p-3 rounded-[var(--radius-default)] border border-plume-200 dark:border-plume-800 bg-plume-50/50 dark:bg-plume-500/5"
                          >
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-[11px] font-medium text-plume-700 dark:text-plume-400">
                                Best {autoTuneData.metric}: {autoTuneData.metric === "accuracy"
                                  ? `${(autoTuneData.best_score * 100).toFixed(1)}%`
                                  : autoTuneData.best_score.toFixed(3)}
                              </span>
                              <span className="text-[10px] text-text-tertiary">
                                from {autoTuneData.all_results.length} combinations tested
                              </span>
                            </div>
                            <div className="flex flex-col gap-1 max-h-[160px] overflow-auto">
                              {autoTuneData.all_results.map((entry, i) => {
                                const isBest = i === 0;
                                const defs = HYPERPARAMS[algorithm] || [];
                                const paramStr = Object.entries(entry.hyperparams)
                                  .map(([k, v]) => {
                                    const def = defs.find((d) => d.key === k);
                                    const shortLabel = def ? def.label.split("(")[0].trim().toLowerCase() : k;
                                    return `${shortLabel} ${typeof v === "number" && v % 1 !== 0 ? v.toFixed(3) : v}`;
                                  })
                                  .join(", ");
                                const scoreFmt = autoTuneData.metric === "accuracy"
                                  ? `${(entry.score * 100).toFixed(1)}%`
                                  : entry.score.toFixed(3);
                                return (
                                  <div
                                    key={i}
                                    onClick={() => setHyperparams(entry.hyperparams)}
                                    className={`flex items-center gap-2 text-[10px] px-1.5 py-1 rounded cursor-pointer transition-colors duration-100 ${
                                      isBest
                                        ? "bg-plume-100 dark:bg-plume-500/10 font-medium"
                                        : "hover:bg-plume-100/50 dark:hover:bg-plume-500/5"
                                    }`}
                                    title="Click to use these settings"
                                  >
                                    <span className={`w-[50px] shrink-0 tabular-nums ${isBest ? "text-plume-700 dark:text-plume-400" : "text-text-secondary"}`}>
                                      {scoreFmt}
                                    </span>
                                    <span className="text-text-tertiary truncate">{paramStr}</span>
                                    {isBest && <span className="text-plume-600 dark:text-plume-400 shrink-0">← best</span>}
                                  </div>
                                );
                              })}
                            </div>
                            <p className="text-[9px] text-text-tertiary mt-2">
                              Click any row to use those settings. Best settings are already applied to the sliders above.
                            </p>
                          </motion.div>
                        )}
                      </div>
                    )}
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
                    {comparing && compareProgress ? (
                      <span className="flex items-center gap-2">
                        <motion.span
                          className="inline-block w-2 h-2 rounded-full bg-plume-500/60"
                          animate={{ scale: [1, 1.3, 1] }}
                          transition={{ duration: 1, repeat: Infinity }}
                        />
                        Training {compareProgress.label} ({compareProgress.current}/{compareProgress.total})
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
