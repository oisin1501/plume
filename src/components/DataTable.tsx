import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { useAppStore } from "../stores/appStore";
import type { ColumnProfile, ColumnDistribution, TablePage } from "../types/data";

const PAGE_SIZE = 100;

export function DataTable() {
  const tablePage = useAppStore((s) => s.tablePage);
  const [offset, setOffset] = useState(0);
  const [selectedCol, setSelectedCol] = useState<string | null>(null);
  const [profile, setProfile] = useState<ColumnProfile | null>(null);
  const [distribution, setDistribution] = useState<ColumnDistribution | null>(null);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDesc, setSortDesc] = useState(false);
  const [filterCol, setFilterCol] = useState<string | null>(null);
  const [filterQuery, setFilterQuery] = useState("");
  const tableRef = useRef<HTMLDivElement>(null);

  const [localPage, setLocalPage] = useState<TablePage | null>(null);
  const activePage = localPage ?? tablePage;

  // Reset local state when the store's tablePage changes (e.g. new file loaded)
  useEffect(() => {
    setLocalPage(null);
    setOffset(0);
    setSelectedCol(null);
    setProfile(null);
    setDistribution(null);
  }, [tablePage]);

  const loadPage = useCallback(
    async (newOffset: number) => {
      const page = await invoke<TablePage>("get_table_page", {
        offset: newOffset,
        limit: PAGE_SIZE,
      });
      setLocalPage(page);
      setOffset(newOffset);
      tableRef.current?.scrollTo(0, 0);
    },
    []
  );

  const handleSort = useCallback(async (colName: string) => {
    const desc = sortCol === colName ? !sortDesc : false;
    setSortCol(colName);
    setSortDesc(desc);
    try {
      const summary = await invoke<import("../types/data").DataSummary>("sort_column", { column: colName, descending: desc });
      const page = await invoke<TablePage>("get_table_page", { offset: 0, limit: PAGE_SIZE });
      useAppStore.setState({ summary, tablePage: page });
      setOffset(0);
    } catch (err) {
      console.error("Sort failed:", err);
    }
  }, [sortCol, sortDesc]);

  const handleFilter = useCallback(async () => {
    if (!filterCol || !filterQuery.trim()) return;
    try {
      const summary = await invoke<import("../types/data").DataSummary>("filter_rows", { column: filterCol, query: filterQuery.trim() });
      const page = await invoke<TablePage>("get_table_page", { offset: 0, limit: PAGE_SIZE });
      useAppStore.setState({ summary, tablePage: page });
      setOffset(0);
      setFilterCol(null);
      setFilterQuery("");
    } catch (err) {
      console.error("Filter failed:", err);
    }
  }, [filterCol, filterQuery]);

  const handleColumnClick = useCallback(async (colName: string) => {
    if (selectedCol === colName) {
      setSelectedCol(null);
      setProfile(null);
      setDistribution(null);
      return;
    }
    setSelectedCol(colName);
    setDistribution(null);
    try {
      const [p, dist] = await Promise.all([
        invoke<ColumnProfile>("get_column_profile", { columnName: colName }),
        invoke<ColumnDistribution>("get_column_distribution", { columnName: colName }),
      ]);
      setProfile(p);
      setDistribution(dist);
    } catch (err) {
      console.error("Failed to profile column:", err);
    }
  }, [selectedCol]);

  useEffect(() => {
    setSelectedCol(null);
    setProfile(null);
    setDistribution(null);
  }, [activePage?.columns]);

  if (!activePage) return null;

  const totalPages = Math.ceil(activePage.total_rows / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Column profile dropdown */}
      <AnimatePresence>
        {profile && selectedCol && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden border-b border-border"
          >
            <div className="px-6 py-4 bg-surface-alt">
              <div className="flex items-baseline gap-4">
                <span className="text-[14px] font-semibold">{profile.name}</span>
                <span className="text-[12px] text-text-tertiary">{profile.dtype}</span>
              </div>
              <div className="grid grid-cols-4 gap-x-8 gap-y-1 mt-3 text-[12px]">
                <Stat label="Count" value={profile.count.toLocaleString()} />
                <Stat label="Missing" value={`${profile.null_count} (${profile.null_percent.toFixed(1)}%)`} warn={profile.null_count > 0} />
                {profile.unique_count != null && <Stat label="Unique" value={profile.unique_count.toLocaleString()} />}
                {profile.mean != null && <Stat label="Mean" value={profile.mean.toFixed(2)} />}
                {profile.std != null && <Stat label="Std" value={profile.std.toFixed(2)} />}
                {profile.min != null && <Stat label="Min" value={profile.min} />}
                {profile.max != null && <Stat label="Max" value={profile.max} />}
              </div>
              {distribution && distribution.counts.length > 0 && (
                <MiniChart distribution={distribution} />
              )}
              <div className="flex items-center gap-2 mt-3">
                <input
                  type="text"
                  placeholder={`Filter by ${profile.name}...`}
                  value={filterCol === profile.name ? filterQuery : ""}
                  onChange={(e) => { setFilterCol(profile.name); setFilterQuery(e.target.value); }}
                  onKeyDown={(e) => { if (e.key === "Enter") handleFilter(); }}
                  className="flex-1 px-2 py-1 text-[11px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500"
                />
                <button
                  onClick={() => { setFilterCol(profile.name); handleFilter(); }}
                  disabled={!filterQuery.trim() || filterCol !== profile.name}
                  className="px-2 py-1 text-[10px] rounded-[var(--radius-default)] border border-border text-text-secondary hover:bg-surface-alt disabled:opacity-30 transition-colors duration-200 cursor-pointer"
                >
                  Filter
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Table */}
      <div ref={tableRef} className="flex-1 overflow-auto">
        <table className="w-full text-[13px] border-collapse">
          <thead className="sticky top-0 z-10">
            <tr className="bg-surface-alt">
              <th className="px-3 py-2 text-left text-[11px] font-medium text-text-tertiary border-b border-border w-[60px]">
                #
              </th>
              {activePage.columns.map((col, i) => (
                <th
                  key={col}
                  onClick={() => handleColumnClick(col)}
                  className={`
                    px-3 py-2 text-left font-medium border-b border-border cursor-pointer
                    transition-colors duration-200 ease-out whitespace-nowrap
                    ${selectedCol === col
                      ? "text-plume-600 dark:text-plume-500 bg-plume-50 dark:bg-plume-500/10"
                      : "text-text-primary hover:text-plume-600 dark:hover:text-plume-500"
                    }
                  `}
                >
                  <div className="flex items-center gap-1">
                    <div className="flex flex-col gap-0.5">
                      <span className="text-[12px]">{col}</span>
                      <span className="text-[10px] font-normal text-text-tertiary">
                        {activePage.column_types[i]}
                      </span>
                    </div>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleSort(col); }}
                      className="text-[9px] text-text-tertiary hover:text-text-primary ml-1 opacity-0 group-hover:opacity-100 transition-opacity"
                      title={`Sort by ${col}`}
                    >
                      {sortCol === col ? (sortDesc ? "▼" : "▲") : "⇅"}
                    </button>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {activePage.rows.map((row, rowIdx) => (
              <tr
                key={rowIdx}
                className="hover:bg-surface-alt transition-colors duration-100"
              >
                <td className="px-3 py-1.5 text-text-tertiary text-[11px] border-b border-border/50">
                  {offset + rowIdx + 1}
                </td>
                {row.map((cell, cellIdx) => (
                  <td
                    key={cellIdx}
                    className={`
                      px-3 py-1.5 border-b border-border/50 tabular-nums
                      ${cell === null ? "text-text-tertiary italic" : "text-text-primary"}
                      ${selectedCol === activePage.columns[cellIdx] ? "bg-plume-50/50 dark:bg-plume-500/5" : ""}
                    `}
                  >
                    {cell === null ? "—" : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer bar */}
      <div className="flex items-center justify-between px-6 py-3 border-t border-border text-[12px] text-text-secondary">
        {/* Column scroll (left side) */}
        <div className="flex items-center gap-2">
          <span className="text-text-tertiary">{activePage.columns.length} cols</span>
          <button
            onClick={() => tableRef.current?.scrollTo({ left: 0, behavior: "smooth" })}
            className="px-2 py-1 rounded-[var(--radius-default)] border border-border hover:bg-surface-alt transition-colors duration-200"
            title="Scroll to first column"
          >
            ← Start
          </button>
          <button
            onClick={() => {
              const el = tableRef.current;
              if (el) el.scrollTo({ left: el.scrollWidth, behavior: "smooth" });
            }}
            className="px-2 py-1 rounded-[var(--radius-default)] border border-border hover:bg-surface-alt transition-colors duration-200"
            title="Scroll to last column"
          >
            End →
          </button>
        </div>

        {/* Row pagination (right side) */}
        {totalPages > 1 && (
          <div className="flex items-center gap-2">
            <span>
              {offset + 1}–{Math.min(offset + PAGE_SIZE, activePage.total_rows)} of{" "}
              {activePage.total_rows.toLocaleString()}
            </span>
            <button
              onClick={() => loadPage(0)}
              disabled={offset === 0}
              className="px-2 py-1 rounded-[var(--radius-default)] border border-border hover:bg-surface-alt disabled:opacity-30 disabled:cursor-default transition-colors duration-200"
              title="First page"
            >
              First
            </button>
            <button
              onClick={() => loadPage(Math.max(0, offset - PAGE_SIZE))}
              disabled={offset === 0}
              className="px-3 py-1 rounded-[var(--radius-default)] border border-border hover:bg-surface-alt disabled:opacity-30 disabled:cursor-default transition-colors duration-200"
            >
              Previous
            </button>
            <button
              onClick={() => loadPage(offset + PAGE_SIZE)}
              disabled={currentPage >= totalPages}
              className="px-3 py-1 rounded-[var(--radius-default)] border border-border hover:bg-surface-alt disabled:opacity-30 disabled:cursor-default transition-colors duration-200"
            >
              Next
            </button>
            <button
              onClick={() => loadPage((totalPages - 1) * PAGE_SIZE)}
              disabled={currentPage >= totalPages}
              className="px-2 py-1 rounded-[var(--radius-default)] border border-border hover:bg-surface-alt disabled:opacity-30 disabled:cursor-default transition-colors duration-200"
              title="Last page"
            >
              Last
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex gap-2">
      <span className="text-text-tertiary">{label}</span>
      <span className={warn ? "text-amber-600 dark:text-amber-400" : "text-text-primary"}>
        {value}
      </span>
    </div>
  );
}

const CHART_HEIGHT = 48;

function MiniChart({ distribution }: { distribution: ColumnDistribution }) {
  const maxCount = Math.max(...distribution.counts);
  const isHistogram = distribution.kind === "histogram";

  return (
    <div className="mt-3">
      <p className="text-[10px] text-text-tertiary mb-1.5">
        {isHistogram ? "Distribution" : "Top values"}
      </p>
      <div
        className={`flex items-end ${isHistogram ? "gap-px" : "gap-1"}`}
        style={{ height: CHART_HEIGHT }}
      >
        {distribution.counts.map((count, i) => {
          const ratio = maxCount > 0 ? count / maxCount : 0;
          const barHeight = Math.max(Math.round(ratio * CHART_HEIGHT), 1);
          return (
            <div
              key={i}
              className="flex flex-col items-center justify-end"
              style={{ flex: isHistogram ? 1 : undefined, minWidth: isHistogram ? 0 : undefined, height: CHART_HEIGHT }}
            >
              <div
                className="w-full bg-plume-400/60 dark:bg-plume-500/40 rounded-sm"
                style={{
                  height: barHeight,
                  minWidth: isHistogram ? undefined : 24,
                  maxWidth: isHistogram ? undefined : 48,
                }}
                title={`${distribution.labels[i]}: ${count.toLocaleString()}`}
              />
            </div>
          );
        })}
      </div>
      {!isHistogram && (
        <div className={`flex ${isHistogram ? "gap-px" : "gap-1"} mt-1`}>
          {distribution.labels.map((label, i) => (
            <span
              key={i}
              className="text-[8px] text-text-tertiary truncate text-center"
              style={{ minWidth: 24, maxWidth: 48 }}
              title={label}
            >
              {label}
            </span>
          ))}
        </div>
      )}
      {isHistogram && (
        <div className="flex justify-between mt-1">
          <span className="text-[8px] text-text-tertiary">{distribution.labels[0]}</span>
          <span className="text-[8px] text-text-tertiary">{distribution.labels[distribution.labels.length - 1]}</span>
        </div>
      )}
    </div>
  );
}
