import { useState, useEffect, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
import type { ColumnProfile, DataSummary, TablePage, TypeRecommendation } from "../types/data";

const NUMERIC_TYPES = new Set(["f64", "i64", "i32", "f32", "u8", "u16", "u32", "u64"]);

type FeatureTab = "combine" | "transform" | "bin";

interface TransformPreview {
  action: string;
  column: string;
  strategy?: string;
  targetType?: string;
  data: {
    before: {
      dtype: string;
      null_count: number;
      null_percent: number;
      mean: number | null;
      std: number | null;
      min: string | null;
      max: string | null;
      unique_count: number | null;
    };
    after: {
      dtype: string;
      null_count: number;
      null_percent: number;
      mean: number | null;
      std: number | null;
      min: string | null;
      max: string | null;
      unique_count: number | null;
    };
    sample_before: string[];
    sample_after: string[];
  };
}

interface TransformEntry {
  action: string;
  column: string | null;
  detail: string;
}

export function ShapeView() {
  const summary = useAppStore((s) => s.summary);
  const [profiles, setProfiles] = useState<ColumnProfile[]>([]);
  const [expandedCol, setExpandedCol] = useState<string | null>(null);
  const [historyLen, setHistoryLen] = useState(0);
  const [working, setWorking] = useState(false);
  const [typeRecs, setTypeRecs] = useState<TypeRecommendation[]>([]);
  const [transformLog, setTransformLog] = useState<TransformEntry[]>([]);
  const [showFeatureEng, setShowFeatureEng] = useState(false);
  const [featureTab, setFeatureTab] = useState<FeatureTab>("combine");
  const [featureError, setFeatureError] = useState<string | null>(null);
  const [featureSuccess, setFeatureSuccess] = useState(false);

  // Combine columns state
  const [combineColA, setCombineColA] = useState("");
  const [combineColB, setCombineColB] = useState("");
  const [combineOp, setCombineOp] = useState("add");
  const [combineName, setCombineName] = useState("");

  // Transform column state
  const [transformCol, setTransformCol] = useState("");
  const [transformType, setTransformType] = useState("log");
  const [transformName, setTransformName] = useState("");

  // Preview state
  const [preview, setPreview] = useState<TransformPreview | null>(null);

  // Bin column state
  const [binCol, setBinCol] = useState("");
  const [binCount, setBinCount] = useState(5);
  const [binName, setBinName] = useState("");

  const numericColumns = useMemo(() => {
    if (!summary) return [];
    return summary.column_names.filter((_, i) => NUMERIC_TYPES.has(summary.column_types[i]));
  }, [summary]);

  const refreshProfiles = useCallback(async () => {
    try {
      const [p, len, recs, log] = await Promise.all([
        invoke<ColumnProfile[]>("get_all_column_profiles"),
        invoke<number>("get_history_length"),
        invoke<TypeRecommendation[]>("get_type_recommendations"),
        invoke<TransformEntry[]>("get_transform_log"),
      ]);
      setProfiles(p);
      setHistoryLen(len);
      setTypeRecs(recs);
      setTransformLog(log);
    } catch (err) {
      console.error("Failed to load profiles:", err);
    }
  }, []);

  useEffect(() => {
    if (summary) refreshProfiles();
  }, [summary, refreshProfiles]);

  const updateAfterTransform = useCallback(async () => {
    const newSummary = await invoke<DataSummary>("get_summary");
    const newPage = await invoke<TablePage>("get_table_page", { offset: 0, limit: 100 });
    useAppStore.setState({ summary: newSummary, tablePage: newPage });
    await refreshProfiles();
    setWorking(false);
  }, [refreshProfiles]);

  const requestPreview = useCallback(async (
    action: string,
    column: string,
    strategy?: string,
    targetType?: string,
  ) => {
    setWorking(true);
    try {
      const data = await invoke<TransformPreview["data"]>("preview_transform", {
        action,
        column,
        strategy: strategy ?? null,
        targetType: targetType ?? null,
      });
      setPreview({ action, column, strategy, targetType, data });
    } catch (err) {
      console.error("Preview failed:", err);
    }
    setWorking(false);
  }, []);

  const handleApplyPreview = useCallback(async () => {
    if (!preview) return;
    setWorking(true);
    setPreview(null);
    try {
      if (preview.action === "fill_missing") {
        await invoke("fill_missing", { column: preview.column, strategy: preview.strategy });
      } else if (preview.action === "cast_column") {
        await invoke("cast_column", { column: preview.column, targetType: preview.targetType });
      } else if (preview.action === "fill_mode") {
        await invoke("fill_mode", { column: preview.column });
      }
      await updateAfterTransform();
    } catch (err) {
      console.error("Apply transform failed:", err);
      setWorking(false);
    }
  }, [preview, updateAfterTransform]);

  const handleCancelPreview = useCallback(() => {
    setPreview(null);
  }, []);

  const handleFillMissing = useCallback(async (column: string, strategy: string) => {
    await requestPreview("fill_missing", column, strategy);
  }, [requestPreview]);

  const handleDropColumn = useCallback(async (column: string) => {
    setWorking(true);
    setExpandedCol(null);
    try {
      await invoke("drop_column", { column });
      await updateAfterTransform();
    } catch (err) {
      console.error("Drop column failed:", err);
      setWorking(false);
    }
  }, [updateAfterTransform]);

  const handleCast = useCallback(async (column: string, targetType: string) => {
    await requestPreview("cast_column", column, undefined, targetType);
  }, [requestPreview]);

  const handleFillMode = useCallback(async (column: string) => {
    await requestPreview("fill_mode", column);
  }, [requestPreview]);

  const handleOneHot = useCallback(async (column: string) => {
    setWorking(true);
    setExpandedCol(null);
    try {
      await invoke("one_hot_encode", { column });
      await updateAfterTransform();
    } catch (err) {
      console.error("One-hot encode failed:", err);
      setWorking(false);
    }
  }, [updateAfterTransform]);

  const handleRename = useCallback(async (oldName: string, newName: string) => {
    setWorking(true);
    try {
      await invoke("rename_column", { oldName, newName });
      await updateAfterTransform();
    } catch (err) {
      console.error("Rename failed:", err);
      setWorking(false);
    }
  }, [updateAfterTransform]);

  const handleUndo = useCallback(async () => {
    setWorking(true);
    try {
      await invoke("undo_transform");
      await updateAfterTransform();
    } catch (err) {
      console.error("Undo failed:", err);
      setWorking(false);
    }
  }, [updateAfterTransform]);

  // Auto-generate combine name when inputs change
  useEffect(() => {
    if (combineColA && combineColB) {
      const opLabels: Record<string, string> = { add: "plus", subtract: "minus", multiply: "times", divide: "div" };
      setCombineName(`${combineColA}_${opLabels[combineOp] ?? combineOp}_${combineColB}`);
    }
  }, [combineColA, combineColB, combineOp]);

  // Auto-generate transform name when inputs change
  useEffect(() => {
    if (transformCol) {
      const tLabels: Record<string, string> = { log: "log", log10: "log10", sqrt: "sqrt", square: "squared", abs: "abs", standardize: "zscore", normalize: "norm" };
      setTransformName(`${transformCol}_${tLabels[transformType] ?? transformType}`);
    }
  }, [transformCol, transformType]);

  // Auto-generate bin name when inputs change
  useEffect(() => {
    if (binCol) {
      setBinName(`${binCol}_binned`);
    }
  }, [binCol]);

  const showSuccessFlash = useCallback(() => {
    setFeatureSuccess(true);
    setTimeout(() => setFeatureSuccess(false), 2000);
  }, []);

  const handleCombineColumns = useCallback(async () => {
    if (!combineColA || !combineColB || !combineName.trim()) return;
    setWorking(true);
    setFeatureError(null);
    try {
      await invoke("math_columns", { colA: combineColA, colB: combineColB, op: combineOp, newName: combineName.trim() });
      await updateAfterTransform();
      showSuccessFlash();
    } catch (err) {
      setFeatureError(String(err));
      setWorking(false);
    }
  }, [combineColA, combineColB, combineOp, combineName, updateAfterTransform, showSuccessFlash]);

  const handleTransformColumn = useCallback(async () => {
    if (!transformCol || !transformName.trim()) return;
    setWorking(true);
    setFeatureError(null);
    try {
      await invoke("transform_column", { column: transformCol, transform: transformType, newName: transformName.trim() });
      await updateAfterTransform();
      showSuccessFlash();
    } catch (err) {
      setFeatureError(String(err));
      setWorking(false);
    }
  }, [transformCol, transformType, transformName, updateAfterTransform, showSuccessFlash]);

  const handleBinColumn = useCallback(async () => {
    if (!binCol || !binName.trim()) return;
    setWorking(true);
    setFeatureError(null);
    try {
      await invoke("bin_column", { column: binCol, nBins: binCount, newName: binName.trim() });
      await updateAfterTransform();
      showSuccessFlash();
    } catch (err) {
      setFeatureError(String(err));
      setWorking(false);
    }
  }, [binCol, binCount, binName, updateAfterTransform, showSuccessFlash]);

  const handleApplyAllRecommendations = useCallback(async () => {
    if (typeRecs.length === 0) return;
    setWorking(true);
    try {
      const casts = typeRecs.map((r) => [r.column, r.recommended_type]);
      await invoke("cast_columns_batch", { casts });
      await updateAfterTransform();
    } catch (err) {
      console.error("Batch cast failed:", err);
      setWorking(false);
    }
  }, [typeRecs, updateAfterTransform]);

  if (!summary) return null;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border">
        <div className="text-[12px] text-text-secondary">
          {profiles.length} columns · {summary.rows.toLocaleString()} rows
        </div>
        <div className="flex items-center gap-3">
          {working && (
            <div className="w-[80px] h-[2px] bg-border rounded-full overflow-hidden">
              <motion.div
                className="h-full w-[40%] bg-plume-500 rounded-full"
                animate={{ x: ["-100%", "350%"] }}
                transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
              />
            </div>
          )}
          <button
            onClick={() => setShowFeatureEng(!showFeatureEng)}
            className={`
              px-3 py-1.5 text-[12px] font-medium rounded-[var(--radius-default)]
              transition-colors duration-200 cursor-pointer
              ${showFeatureEng
                ? "bg-plume-500 text-white"
                : "text-text-secondary hover:text-text-primary hover:bg-surface-alt"
              }
            `}
          >
            Feature Engineering
          </button>
          {historyLen > 0 && (
            <button
              onClick={handleUndo}
              disabled={working}
              className="text-[12px] text-plume-600 dark:text-plume-500 hover:text-plume-700 disabled:opacity-40 transition-colors duration-200"
            >
              Undo
            </button>
          )}
        </div>
      </div>

      {/* Pipeline visualization */}
      {transformLog.length > 0 && (
        <div className="px-6 py-3 border-b border-border overflow-x-auto">
          <div className="flex items-center gap-0 min-w-0">
            {/* Start dot */}
            <div className="flex flex-col items-center shrink-0">
              <div className="w-2.5 h-2.5 rounded-full bg-plume-500" />
              <span className="text-[9px] text-text-tertiary mt-1">Raw</span>
            </div>

            {transformLog.map((entry, i) => (
              <div key={i} className="flex items-center shrink-0">
                {/* Connector line */}
                <div className="w-8 h-px bg-border" />
                {/* Step dot + label */}
                <div className="flex flex-col items-center" title={entry.detail ? `${entry.action}: ${entry.detail}` : entry.action}>
                  <div className="w-2.5 h-2.5 rounded-full bg-plume-400 dark:bg-plume-500" />
                  <span className="text-[9px] text-text-tertiary mt-1 whitespace-nowrap max-w-[72px] truncate text-center">
                    {entry.column
                      ? `${entry.action} ${entry.column}`
                      : `${entry.action} ${entry.detail}`
                    }
                  </span>
                </div>
              </div>
            ))}

            {/* End connector + current state */}
            <div className="flex items-center shrink-0">
              <div className="w-8 h-px bg-border" />
              <div className="flex flex-col items-center">
                <div className="w-3 h-3 rounded-full bg-emerald-500 ring-2 ring-emerald-500/20" />
                <span className="text-[9px] text-emerald-600 dark:text-emerald-400 mt-1 font-medium">Now</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Type recommendations banner */}
      {typeRecs.length > 0 && (
        <TypeRecsBanner
          typeRecs={typeRecs}
          onApplyAll={handleApplyAllRecommendations}
          onApplyOne={(col, type) => handleCast(col, type)}
          disabled={working}
        />
      )}

      {/* Feature Engineering panel */}
      <AnimatePresence>
        {showFeatureEng && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="bg-surface-alt border-b border-border px-6 py-4">
              {/* Tabs */}
              <div className="flex items-center gap-1 mb-4">
                {([
                  { id: "combine" as FeatureTab, label: "Combine columns" },
                  { id: "transform" as FeatureTab, label: "Transform column" },
                  { id: "bin" as FeatureTab, label: "Bin column" },
                ]).map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => { setFeatureTab(tab.id); setFeatureError(null); }}
                    className={`
                      px-3 py-1.5 text-[12px] font-medium rounded-[var(--radius-default)]
                      transition-colors duration-200 cursor-pointer
                      ${featureTab === tab.id
                        ? "bg-plume-500 text-white"
                        : "text-text-secondary hover:text-text-primary hover:bg-surface"
                      }
                    `}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab content */}
              {featureTab === "combine" && (
                <div className="flex items-end gap-3 flex-wrap">
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">Column A</label>
                    <select
                      value={combineColA}
                      onChange={(e) => setCombineColA(e.target.value)}
                      className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200"
                    >
                      <option value="">Select...</option>
                      {numericColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">Operation</label>
                    <select
                      value={combineOp}
                      onChange={(e) => setCombineOp(e.target.value)}
                      className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200"
                    >
                      <option value="add">Add (+)</option>
                      <option value="subtract">Subtract (&minus;)</option>
                      <option value="multiply">Multiply (&times;)</option>
                      <option value="divide">Divide (&divide;)</option>
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">Column B</label>
                    <select
                      value={combineColB}
                      onChange={(e) => setCombineColB(e.target.value)}
                      className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200"
                    >
                      <option value="">Select...</option>
                      {numericColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">New column name</label>
                    <input
                      type="text"
                      value={combineName}
                      onChange={(e) => setCombineName(e.target.value)}
                      className="w-[160px] px-2 py-1 text-[11px] border border-plume-400 rounded-[var(--radius-default)] bg-surface text-text-primary outline-none"
                    />
                  </div>
                  <button
                    onClick={handleCombineColumns}
                    disabled={working || !combineColA || !combineColB || !combineName.trim()}
                    className="bg-plume-500 text-white hover:bg-plume-600 rounded-[var(--radius-default)] px-4 py-1.5 text-[12px] disabled:opacity-40 transition-colors duration-200 cursor-pointer"
                  >
                    Create
                  </button>
                </div>
              )}

              {featureTab === "transform" && (
                <div className="flex items-end gap-3 flex-wrap">
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">Column</label>
                    <select
                      value={transformCol}
                      onChange={(e) => setTransformCol(e.target.value)}
                      className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200"
                    >
                      <option value="">Select...</option>
                      {numericColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">Transform</label>
                    <select
                      value={transformType}
                      onChange={(e) => setTransformType(e.target.value)}
                      className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200"
                    >
                      <option value="log">Log (ln)</option>
                      <option value="log10">Log10</option>
                      <option value="sqrt">Square root</option>
                      <option value="square">Square</option>
                      <option value="abs">Absolute value</option>
                      <option value="standardize">Standardize (z-score)</option>
                      <option value="normalize">Normalize (0-1)</option>
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">New column name</label>
                    <input
                      type="text"
                      value={transformName}
                      onChange={(e) => setTransformName(e.target.value)}
                      className="w-[160px] px-2 py-1 text-[11px] border border-plume-400 rounded-[var(--radius-default)] bg-surface text-text-primary outline-none"
                    />
                  </div>
                  <button
                    onClick={handleTransformColumn}
                    disabled={working || !transformCol || !transformName.trim()}
                    className="bg-plume-500 text-white hover:bg-plume-600 rounded-[var(--radius-default)] px-4 py-1.5 text-[12px] disabled:opacity-40 transition-colors duration-200 cursor-pointer"
                  >
                    Create
                  </button>
                </div>
              )}

              {featureTab === "bin" && (
                <div className="flex items-end gap-3 flex-wrap">
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">Column</label>
                    <select
                      value={binCol}
                      onChange={(e) => setBinCol(e.target.value)}
                      className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200"
                    >
                      <option value="">Select...</option>
                      {numericColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">Number of bins: {binCount}</label>
                    <input
                      type="range"
                      min={2}
                      max={20}
                      value={binCount}
                      onChange={(e) => setBinCount(Number(e.target.value))}
                      className="w-[120px] accent-plume-500"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[11px] text-text-tertiary">New column name</label>
                    <input
                      type="text"
                      value={binName}
                      onChange={(e) => setBinName(e.target.value)}
                      className="w-[160px] px-2 py-1 text-[11px] border border-plume-400 rounded-[var(--radius-default)] bg-surface text-text-primary outline-none"
                    />
                  </div>
                  <button
                    onClick={handleBinColumn}
                    disabled={working || !binCol || !binName.trim()}
                    className="bg-plume-500 text-white hover:bg-plume-600 rounded-[var(--radius-default)] px-4 py-1.5 text-[12px] disabled:opacity-40 transition-colors duration-200 cursor-pointer"
                  >
                    Create
                  </button>
                </div>
              )}

              {/* Error / success messages */}
              {featureError && (
                <p className="text-[11px] text-red-500 mt-2">{featureError}</p>
              )}
              {featureSuccess && (
                <p className="text-[11px] text-emerald-600 dark:text-emerald-400 mt-2">Column created!</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Column list */}
      <div className="flex-1 overflow-auto">
        <div className="max-w-[720px] mx-auto py-4 px-6">
          {profiles.map((profile) => (
            <div key={profile.name}>
              <ColumnRow
                profile={profile}
                totalRows={summary.rows}
                isExpanded={expandedCol === profile.name}
                onToggle={() => setExpandedCol(expandedCol === profile.name ? null : profile.name)}
                onFillMissing={handleFillMissing}
                onFillMode={handleFillMode}
                onOneHot={handleOneHot}
                onRename={handleRename}
                onDrop={handleDropColumn}
                onCast={handleCast}
                disabled={working}
                recommendation={typeRecs.find((r) => r.column === profile.name)}
              />
              {preview && preview.column === profile.name && (
                <PreviewPanel
                  preview={preview}
                  onApply={handleApplyPreview}
                  onCancel={handleCancelPreview}
                  disabled={working}
                />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ColumnRow({
  profile,
  totalRows,
  isExpanded,
  onToggle,
  onFillMissing,
  onFillMode,
  onOneHot,
  onRename,
  onDrop,
  onCast,
  disabled,
  recommendation,
}: {
  profile: ColumnProfile;
  totalRows: number;
  isExpanded: boolean;
  onToggle: () => void;
  onFillMissing: (col: string, strategy: string) => void;
  onFillMode: (col: string) => void;
  onOneHot: (col: string) => void;
  onRename: (oldName: string, newName: string) => void;
  onDrop: (col: string) => void;
  onCast: (col: string, type: string) => void;
  disabled: boolean;
  recommendation?: TypeRecommendation;
}) {
  const hasMissing = profile.null_count > 0;
  const [renaming, setRenaming] = useState(false);
  const [newName, setNewName] = useState(profile.name);
  const isCategorical = profile.dtype === "str" || profile.dtype === "cat";
  const [dismissedHints, setDismissedHints] = useState<Set<string>>(new Set());

  // Build contextual suggestions
  const hints: { id: string; text: string; action?: { label: string; onClick: () => void } }[] = [];

  // High missing values
  if (hasMissing && profile.null_percent >= 40) {
    hints.push({
      id: "high_missing",
      text: `${profile.null_percent.toFixed(0)}% of values are missing — consider dropping this column or filling with ${profile.mean != null ? "the average" : "the most common value"}.`,
      action: profile.mean != null
        ? { label: "Fill with mean", onClick: () => onFillMissing(profile.name, "mean") }
        : { label: "Fill with most common", onClick: () => onFillMode(profile.name) },
    });
  } else if (hasMissing && profile.null_percent >= 10) {
    hints.push({
      id: "moderate_missing",
      text: `${profile.null_percent.toFixed(0)}% of values are missing. You can fill them or drop the missing rows.`,
    });
  }

  // Low cardinality — might be boolean or categorical
  if (profile.unique_count != null && profile.unique_count === 2 && profile.dtype !== "bool") {
    hints.push({
      id: "binary",
      text: "Only 2 unique values — this could work well as a boolean column.",
      action: { label: "Convert to bool", onClick: () => onCast(profile.name, "bool") },
    });
  }

  // Categorical with low cardinality — suggest one-hot encoding
  if (isCategorical && profile.unique_count != null && profile.unique_count >= 3 && profile.unique_count <= 10) {
    hints.push({
      id: "one_hot",
      text: `${profile.unique_count} unique categories — one-hot encoding could make this column usable for training.`,
      action: { label: "One-hot encode", onClick: () => onOneHot(profile.name) },
    });
  }

  // High cardinality text — probably not useful as-is
  if (isCategorical && profile.unique_count != null && profile.unique_count > 50 && totalRows > 0 && profile.unique_count / totalRows > 0.5) {
    hints.push({
      id: "high_cardinality",
      text: "Very high number of unique values — this may be an identifier or free text. Consider dropping it before training.",
    });
  }

  // Constant column
  if (profile.unique_count != null && profile.unique_count <= 1) {
    hints.push({
      id: "constant",
      text: "This column has only one value (or is entirely empty) — it won't help a model learn anything.",
      action: { label: "Drop column", onClick: () => onDrop(profile.name) },
    });
  }

  const visibleHints = hints.filter((h) => !dismissedHints.has(h.id));

  return (
    <div className="border-b border-border/50 last:border-b-0">
      <button
        onClick={onToggle}
        disabled={disabled}
        className="w-full flex items-center gap-4 px-4 py-3 text-left hover:bg-surface-alt transition-colors duration-200 disabled:opacity-60 cursor-pointer"
      >
        {/* Missing indicator dot */}
        <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${hasMissing ? "bg-amber-500" : "bg-emerald-500"}`} />

        <div className="flex-1 min-w-0">
          <span className="text-[13px] font-medium text-text-primary">{profile.name}</span>
        </div>

        <span className="text-[11px] text-text-tertiary w-[60px] text-right shrink-0">{profile.dtype}</span>

        {hasMissing && (
          <span className="text-[11px] text-amber-600 dark:text-amber-400 w-[80px] text-right shrink-0">
            {profile.null_count} missing
          </span>
        )}

        {profile.unique_count != null && (
          <span className="text-[11px] text-text-tertiary w-[70px] text-right shrink-0">
            {profile.unique_count.toLocaleString()} unique
          </span>
        )}
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1 flex flex-col gap-3">
              {/* Stats row */}
              <div className="flex gap-6 text-[11px] pl-5">
                {profile.mean != null && (
                  <span><span className="text-text-tertiary">Mean</span> <span className="text-text-primary">{profile.mean.toFixed(2)}</span></span>
                )}
                {profile.std != null && (
                  <span><span className="text-text-tertiary">Std</span> <span className="text-text-primary">{profile.std.toFixed(2)}</span></span>
                )}
                {profile.min != null && (
                  <span><span className="text-text-tertiary">Min</span> <span className="text-text-primary">{profile.min}</span></span>
                )}
                {profile.max != null && (
                  <span><span className="text-text-tertiary">Max</span> <span className="text-text-primary">{profile.max}</span></span>
                )}
              </div>

              {/* Contextual suggestions */}
              {visibleHints.length > 0 && (
                <div className="flex flex-col gap-1.5 pl-5">
                  {visibleHints.map((hint) => (
                    <div
                      key={hint.id}
                      className="flex items-start gap-2 py-1.5 text-[11px] bg-amber-50 dark:bg-amber-500/5 rounded-[var(--radius-default)] px-3"
                    >
                      <span className="text-amber-600 dark:text-amber-400 leading-relaxed flex-1">
                        {hint.text}
                      </span>
                      {hint.action && (
                        <button
                          onClick={(e) => { e.stopPropagation(); hint.action!.onClick(); }}
                          disabled={disabled}
                          className="px-2 py-0.5 text-[10px] rounded border border-amber-300 dark:border-amber-700 text-amber-700 dark:text-amber-400 hover:bg-amber-100 dark:hover:bg-amber-500/10 disabled:opacity-40 transition-colors duration-200 cursor-pointer shrink-0"
                        >
                          {hint.action.label}
                        </button>
                      )}
                      <button
                        onClick={(e) => { e.stopPropagation(); setDismissedHints((prev) => new Set(prev).add(hint.id)); }}
                        className="text-[9px] text-amber-400 dark:text-amber-600 hover:text-amber-600 dark:hover:text-amber-400 cursor-pointer shrink-0 mt-0.5"
                        title="Dismiss"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Type recommendation */}
              {recommendation && (
                <div className="flex items-center gap-2 pl-5 py-1.5 text-[11px]">
                  <span className="text-blue-600 dark:text-blue-400">
                    Suggestion: convert to {recommendation.recommended_type}
                  </span>
                  <span className="text-text-tertiary">— {recommendation.reason}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); onCast(profile.name, recommendation.recommended_type); }}
                    disabled={disabled}
                    className="px-2 py-0.5 text-[10px] rounded border border-blue-300 dark:border-blue-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 disabled:opacity-40 transition-colors duration-200 cursor-pointer"
                  >
                    Apply
                  </button>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2 pl-5 flex-wrap">
                {hasMissing && profile.mean != null && (
                  <ActionButton label="Fill with mean" onClick={() => onFillMissing(profile.name, "mean")} disabled={disabled} />
                )}
                {hasMissing && (
                  <ActionButton label="Fill with zero" onClick={() => onFillMissing(profile.name, "zero")} disabled={disabled} />
                )}
                {hasMissing && (
                  <ActionButton label="Fill with most common" onClick={() => onFillMode(profile.name)} disabled={disabled} />
                )}
                {hasMissing && (
                  <ActionButton label="Drop missing rows" onClick={() => onFillMissing(profile.name, "drop")} disabled={disabled} />
                )}

                {isCategorical && (
                  <ActionButton label="One-hot encode" onClick={() => onOneHot(profile.name)} disabled={disabled} />
                )}

                {profile.dtype !== "f64" && profile.dtype !== "i64" && (
                  <ActionButton label="Convert to number" onClick={() => onCast(profile.name, "f64")} disabled={disabled} />
                )}
                {profile.dtype !== "str" && (
                  <ActionButton label="Convert to text" onClick={() => onCast(profile.name, "str")} disabled={disabled} />
                )}

                {!renaming ? (
                  <ActionButton label="Rename" onClick={() => { setRenaming(true); setNewName(profile.name); }} disabled={disabled} />
                ) : (
                  <span className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="text"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && newName.trim() && newName !== profile.name) { onRename(profile.name, newName.trim()); setRenaming(false); } if (e.key === "Escape") setRenaming(false); }}
                      className="w-[120px] px-2 py-1 text-[11px] border border-plume-400 rounded-[var(--radius-default)] bg-surface text-text-primary outline-none"
                      autoFocus
                    />
                    <button
                      onClick={() => { if (newName.trim() && newName !== profile.name) { onRename(profile.name, newName.trim()); } setRenaming(false); }}
                      className="px-2 py-1 text-[10px] rounded-[var(--radius-default)] bg-plume-500 text-white cursor-pointer"
                    >
                      OK
                    </button>
                  </span>
                )}

                <ActionButton label="Drop column" onClick={() => onDrop(profile.name)} disabled={disabled} warn />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TypeRecsBanner({
  typeRecs,
  onApplyAll,
  onApplyOne,
  disabled,
}: {
  typeRecs: TypeRecommendation[];
  onApplyAll: () => void;
  onApplyOne: (col: string, type: string) => void;
  disabled: boolean;
}) {
  const [expanded, setExpanded] = useState(false);

  // Group recommendations by recommended type
  const grouped = useMemo(() => {
    const map = new Map<string, TypeRecommendation[]>();
    for (const r of typeRecs) {
      const existing = map.get(r.recommended_type) ?? [];
      existing.push(r);
      map.set(r.recommended_type, existing);
    }
    return map;
  }, [typeRecs]);

  const summaryLabel = typeRecs.every((r) => r.recommended_type === typeRecs[0].recommended_type)
    ? typeRecs[0].recommended_type === "f64" ? "numbers stored as text" : typeRecs[0].recommended_type
    : "a different type";

  return (
    <div className="border-b border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-500/10">
      <div className="flex items-center justify-between px-6 py-2.5">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-[12px] text-blue-700 dark:text-blue-300 hover:text-blue-900 dark:hover:text-blue-100 transition-colors duration-200 cursor-pointer"
        >
          <span className="mr-1.5 text-[10px]">{expanded ? "▼" : "▶"}</span>
          {typeRecs.length} {typeRecs.length === 1 ? "column looks" : "columns look"} like {summaryLabel}
        </button>
        <button
          onClick={onApplyAll}
          disabled={disabled}
          className="px-3 py-1 text-[11px] rounded-[var(--radius-default)] bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 transition-colors duration-200 cursor-pointer"
        >
          Convert all {typeRecs.length}
        </button>
      </div>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div className="px-6 pb-3">
              {Array.from(grouped.entries()).map(([type, recs]) => (
                <div key={type} className="mb-2 last:mb-0">
                  {grouped.size > 1 && (
                    <p className="text-[10px] text-blue-600/70 dark:text-blue-400/70 mb-1 uppercase tracking-wide font-medium">
                      Convert to {type === "f64" ? "number" : type}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-1.5">
                    {recs.map((r) => (
                      <button
                        key={r.column}
                        onClick={() => onApplyOne(r.column, r.recommended_type)}
                        disabled={disabled}
                        className="px-2.5 py-1 text-[11px] rounded-[var(--radius-default)] border border-blue-200 dark:border-blue-700 text-blue-700 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-500/20 disabled:opacity-40 transition-colors duration-200 cursor-pointer"
                        title={r.reason}
                      >
                        {r.column}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function PreviewPanel({
  preview,
  onApply,
  onCancel,
  disabled,
}: {
  preview: TransformPreview;
  onApply: () => void;
  onCancel: () => void;
  disabled: boolean;
}) {
  const { before, after, sample_before, sample_after } = preview.data;

  const actionLabel =
    preview.action === "fill_missing"
      ? `Fill missing (${preview.strategy})`
      : preview.action === "cast_column"
        ? `Cast to ${preview.targetType}`
        : "Fill with most common";

  // Stats that changed
  const statRows: { label: string; before: string; after: string }[] = [];

  if (before.null_count !== after.null_count) {
    statRows.push({
      label: "Nulls",
      before: `${before.null_count} (${before.null_percent.toFixed(1)}%)`,
      after: `${after.null_count} (${after.null_percent.toFixed(1)}%)`,
    });
  }
  if (before.dtype !== after.dtype) {
    statRows.push({ label: "Type", before: before.dtype, after: after.dtype });
  }
  if (before.mean !== after.mean) {
    statRows.push({
      label: "Mean",
      before: before.mean != null ? before.mean.toFixed(4) : "-",
      after: after.mean != null ? after.mean.toFixed(4) : "-",
    });
  }
  if (before.std !== after.std) {
    statRows.push({
      label: "Std",
      before: before.std != null ? before.std.toFixed(4) : "-",
      after: after.std != null ? after.std.toFixed(4) : "-",
    });
  }
  if (before.min !== after.min) {
    statRows.push({ label: "Min", before: before.min ?? "-", after: after.min ?? "-" });
  }
  if (before.max !== after.max) {
    statRows.push({ label: "Max", before: before.max ?? "-", after: after.max ?? "-" });
  }
  if (before.unique_count !== after.unique_count) {
    statRows.push({
      label: "Unique",
      before: before.unique_count != null ? String(before.unique_count) : "-",
      after: after.unique_count != null ? String(after.unique_count) : "-",
    });
  }

  // If nothing changed in stats, show key stats anyway
  if (statRows.length === 0) {
    statRows.push({
      label: "Nulls",
      before: `${before.null_count}`,
      after: `${after.null_count}`,
    });
    if (before.mean != null) {
      statRows.push({
        label: "Mean",
        before: before.mean.toFixed(4),
        after: after.mean != null ? after.mean.toFixed(4) : "-",
      });
    }
  }

  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: "auto", opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      className="overflow-hidden"
    >
      <div className="mx-4 mb-3 p-4 rounded-[var(--radius-default)] border border-plume-300 dark:border-plume-700 bg-plume-50 dark:bg-plume-500/5">
        <div className="flex items-center justify-between mb-3">
          <span className="text-[12px] font-medium text-plume-700 dark:text-plume-300">
            Preview: {actionLabel} on "{preview.column}"
          </span>
          <div className="flex gap-2">
            <button
              onClick={onCancel}
              className="px-3 py-1 text-[11px] rounded-[var(--radius-default)] border border-border text-text-secondary hover:bg-surface-alt transition-colors duration-200 cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={onApply}
              disabled={disabled}
              className="px-3 py-1 text-[11px] rounded-[var(--radius-default)] bg-plume-500 text-white hover:bg-plume-600 disabled:opacity-40 transition-colors duration-200 cursor-pointer"
            >
              Apply
            </button>
          </div>
        </div>

        {/* Stats comparison */}
        <div className="mb-3">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-text-tertiary">
                <th className="text-left font-normal pr-4 pb-1">Stat</th>
                <th className="text-right font-normal pr-4 pb-1">Before</th>
                <th className="text-right font-normal pb-1">After</th>
              </tr>
            </thead>
            <tbody>
              {statRows.map((row) => (
                <tr key={row.label}>
                  <td className="text-text-tertiary pr-4 py-0.5">{row.label}</td>
                  <td className="text-right text-text-secondary pr-4 py-0.5">{row.before}</td>
                  <td className={`text-right py-0.5 ${row.before !== row.after ? "text-plume-600 dark:text-plume-400 font-medium" : "text-text-secondary"}`}>
                    {row.after}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Sample values */}
        {sample_before.length > 0 && (
          <div>
            <p className="text-[10px] text-text-tertiary mb-1">Sample values (first {sample_before.length} rows)</p>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-text-tertiary">
                    <th className="text-left font-normal pr-4 pb-1 w-8">#</th>
                    <th className="text-left font-normal pr-4 pb-1">Before</th>
                    <th className="text-left font-normal pb-1">After</th>
                  </tr>
                </thead>
                <tbody>
                  {sample_before.map((val, i) => (
                    <tr key={i}>
                      <td className="text-text-tertiary pr-4 py-0.5">{i + 1}</td>
                      <td className="text-text-secondary pr-4 py-0.5 font-mono">{val}</td>
                      <td className={`py-0.5 font-mono ${val !== sample_after[i] ? "text-plume-600 dark:text-plume-400 font-medium" : "text-text-secondary"}`}>
                        {sample_after[i]}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

function ActionButton({
  label,
  onClick,
  disabled,
  warn,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  warn?: boolean;
}) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      disabled={disabled}
      className={`
        px-3 py-1.5 text-[11px] rounded-[var(--radius-default)] border
        transition-colors duration-200 disabled:opacity-40 cursor-pointer
        ${warn
          ? "border-red-200 dark:border-red-900 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20"
          : "border-border text-text-secondary hover:bg-surface-alt hover:text-text-primary"
        }
      `}
    >
      {label}
    </button>
  );
}
