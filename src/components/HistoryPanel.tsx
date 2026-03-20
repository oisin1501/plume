import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
import type { DataSummary, TablePage } from "../types/data";

export function HistoryPanel() {
  const historyOpen = useAppStore((s) => s.historyOpen);
  const setHistoryOpen = useAppStore((s) => s.setHistoryOpen);
  const transformLog = useAppStore((s) => s.transformLog);
  const fetchTransformLog = useAppStore((s) => s.fetchTransformLog);
  const fileName = useAppStore((s) => s.fileName);
  const [restoring, setRestoring] = useState<number | null>(null);

  useEffect(() => {
    if (historyOpen) fetchTransformLog();
  }, [historyOpen]);

  const handleRestore = async (step: number) => {
    setRestoring(step);
    try {
      const summary = await invoke<DataSummary>("restore_to_step", { step });
      const page = await invoke<TablePage>("get_table_page", { offset: 0, limit: 100 });
      useAppStore.setState({ summary, tablePage: page });
      await fetchTransformLog();
    } catch (err) {
      console.error("Restore failed:", err);
    } finally {
      setRestoring(null);
    }
  };

  return (
    <AnimatePresence>
      {historyOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-40 bg-black/20"
            onClick={() => setHistoryOpen(false)}
          />
          {/* Panel */}
          <motion.div
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: "spring", stiffness: 400, damping: 35 }}
            className="fixed left-[140px] top-0 bottom-0 z-50 w-[280px] bg-surface border-r border-border shadow-lg flex flex-col"
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <h2 className="text-[13px] font-medium text-text-primary">History</h2>
              <button
                onClick={() => setHistoryOpen(false)}
                className="text-[12px] text-text-tertiary hover:text-text-primary cursor-pointer"
              >
                Close
              </button>
            </div>

            <div className="flex-1 overflow-auto px-4 py-3">
              {/* Timeline */}
              <div className="flex flex-col">
                {/* Step 0: Raw data */}
                <div className="flex items-start gap-3 group">
                  <div className="flex flex-col items-center">
                    <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                      transformLog.length === 0
                        ? "bg-plume-500 ring-2 ring-plume-200 dark:ring-plume-500/30"
                        : "bg-border"
                    }`} />
                    {transformLog.length > 0 && (
                      <div className="w-px h-6 bg-border" />
                    )}
                  </div>
                  <div className="flex-1 pb-4">
                    <p className={`text-[12px] font-medium ${
                      transformLog.length === 0 ? "text-text-primary" : "text-text-secondary"
                    }`}>
                      {fileName ?? "Raw data"}
                    </p>
                    <p className="text-[10px] text-text-tertiary">Original dataset</p>
                    {transformLog.length > 0 && (
                      <button
                        onClick={() => handleRestore(0)}
                        disabled={restoring !== null}
                        className="mt-1 text-[10px] text-plume-600 dark:text-plume-400 hover:text-plume-700 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity duration-150"
                      >
                        {restoring === 0 ? "Restoring..." : "Restore to this point"}
                      </button>
                    )}
                  </div>
                </div>

                {/* Transform steps */}
                {transformLog.map((entry, i) => {
                  const isLast = i === transformLog.length - 1;
                  const isCurrent = isLast;
                  return (
                    <div key={i} className="flex items-start gap-3 group">
                      <div className="flex flex-col items-center">
                        <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                          isCurrent
                            ? "bg-plume-500 ring-2 ring-plume-200 dark:ring-plume-500/30"
                            : "bg-border"
                        }`} />
                        {!isLast && (
                          <div className="w-px h-6 bg-border" />
                        )}
                      </div>
                      <div className="flex-1 pb-4">
                        <p className={`text-[12px] font-medium ${
                          isCurrent ? "text-text-primary" : "text-text-secondary"
                        }`}>
                          {entry.action}
                        </p>
                        <p className="text-[10px] text-text-tertiary">
                          {entry.column && <span>{entry.column}</span>}
                          {entry.column && entry.detail && <span> · </span>}
                          {entry.detail && <span>{entry.detail}</span>}
                        </p>
                        {isCurrent && (
                          <span className="inline-block mt-1 text-[9px] text-plume-600 dark:text-plume-400 bg-plume-50 dark:bg-plume-500/10 px-1.5 py-0.5 rounded">
                            Current
                          </span>
                        )}
                        {!isCurrent && (
                          <button
                            onClick={() => handleRestore(i + 1)}
                            disabled={restoring !== null}
                            className="mt-1 text-[10px] text-plume-600 dark:text-plume-400 hover:text-plume-700 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity duration-150"
                          >
                            {restoring === i + 1 ? "Restoring..." : "Restore to this point"}
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {transformLog.length === 0 && (
                <p className="text-[11px] text-text-tertiary mt-2">
                  No transforms applied yet. Changes you make in the Shape tab will appear here.
                </p>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
