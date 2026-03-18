import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
import type { ColumnProfile, DataSummary, TablePage, TypeRecommendation } from "../types/data";

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

  const handleFillMissing = useCallback(async (column: string, strategy: string) => {
    setWorking(true);
    try {
      await invoke("fill_missing", { column, strategy });
      await updateAfterTransform();
    } catch (err) {
      console.error("Fill missing failed:", err);
      setWorking(false);
    }
  }, [updateAfterTransform]);

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
    setWorking(true);
    try {
      await invoke("cast_column", { column, targetType });
      await updateAfterTransform();
    } catch (err) {
      console.error("Cast failed:", err);
      setWorking(false);
    }
  }, [updateAfterTransform]);

  const handleFillMode = useCallback(async (column: string) => {
    setWorking(true);
    try {
      await invoke("fill_mode", { column });
      await updateAfterTransform();
    } catch (err) {
      console.error("Fill mode failed:", err);
      setWorking(false);
    }
  }, [updateAfterTransform]);

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
        <div className="flex items-center justify-between px-6 py-2.5 bg-blue-50 dark:bg-blue-500/10 border-b border-blue-200 dark:border-blue-800">
          <span className="text-[12px] text-blue-700 dark:text-blue-300">
            {typeRecs.length} {typeRecs.length === 1 ? "column looks" : "columns look"} like{" "}
            {typeRecs.every((r) => r.recommended_type === typeRecs[0].recommended_type)
              ? typeRecs[0].recommended_type === "f64" ? "numbers stored as text" : typeRecs[0].recommended_type
              : "a different type"
            }
          </span>
          <button
            onClick={handleApplyAllRecommendations}
            disabled={working}
            className="px-3 py-1 text-[11px] rounded-[var(--radius-default)] bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 transition-colors duration-200 cursor-pointer"
          >
            Convert all {typeRecs.length}
          </button>
        </div>
      )}

      {/* Column list */}
      <div className="flex-1 overflow-auto">
        <div className="max-w-[720px] mx-auto py-4 px-6">
          {profiles.map((profile) => (
            <ColumnRow
              key={profile.name}
              profile={profile}
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
          ))}
        </div>
      </div>
    </div>
  );
}

function ColumnRow({
  profile,
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
