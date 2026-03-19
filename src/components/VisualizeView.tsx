import { useState, useEffect, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ScatterChart, Scatter, ResponsiveContainer,
} from "recharts";
import { useAppStore } from "../stores/appStore";
import type { ColumnDistribution, CorrelationMatrix, ScatterData, BoxPlotGroup } from "../types/data";
import { ChartSizeToggle, useChartSize } from "./ChartSizeToggle";

type VisTab = "histogram" | "scatter" | "correlation" | "boxplot";

const NUMERIC_TYPES = new Set(["f64", "i64", "i32", "f32", "u8", "u16", "u32", "u64"]);

export function VisualizeView() {
  const summary = useAppStore((s) => s.summary);
  const [activeTab, setActiveTab] = useState<VisTab>("histogram");

  const numericColumns = useMemo(() => {
    if (!summary) return [];
    return summary.column_names.filter((_, i) => NUMERIC_TYPES.has(summary.column_types[i]));
  }, [summary]);

  const allColumns = useMemo(() => summary?.column_names ?? [], [summary]);

  if (!summary) return null;

  const tabs: { id: VisTab; label: string }[] = [
    { id: "histogram", label: "Histogram" },
    { id: "scatter", label: "Scatter Plot" },
    { id: "correlation", label: "Correlation Heatmap" },
    { id: "boxplot", label: "Box Plot" },
  ];

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Tab bar */}
      <div className="flex items-center gap-1 px-6 py-3 border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              px-3 py-1.5 text-[12px] font-medium rounded-[var(--radius-default)]
              transition-colors duration-200 cursor-pointer
              ${activeTab === tab.id
                ? "bg-plume-500 text-white"
                : "text-text-secondary hover:text-text-primary hover:bg-surface-alt"
              }
            `}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto p-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
          >
            {activeTab === "histogram" && (
              <HistogramTab numericColumns={numericColumns} allColumns={allColumns} />
            )}
            {activeTab === "scatter" && (
              <ScatterTab numericColumns={numericColumns} />
            )}
            {activeTab === "correlation" && (
              <CorrelationTab numericColumns={numericColumns} />
            )}
            {activeTab === "boxplot" && (
              <BoxPlotTab numericColumns={numericColumns} allColumns={allColumns} />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Histogram Tab                                                     */
/* ------------------------------------------------------------------ */

function HistogramTab({ numericColumns, allColumns }: { numericColumns: string[]; allColumns: string[] }) {
  const [selectedCol, setSelectedCol] = useState<string>("");
  const [bins, setBins] = useState(20);
  const [distribution, setDistribution] = useState<ColumnDistribution | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartHeight, chartSize, setChartSize] = useChartSize("M");

  useEffect(() => {
    if (!selectedCol) {
      setDistribution(null);
      return;
    }
    setLoading(true);
    invoke<ColumnDistribution>("get_column_distribution", { columnName: selectedCol })
      .then(setDistribution)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedCol, bins]);

  const chartData = useMemo(() => {
    if (!distribution) return [];
    return distribution.labels.map((label, i) => ({
      label,
      count: distribution.counts[i],
    }));
  }, [distribution]);

  return (
    <div className="max-w-[780px] mx-auto">
      <div className="flex items-center gap-4 mb-6">
        <div className="flex flex-col gap-1">
          <label className="text-[11px] text-text-tertiary">Column</label>
          <select
            value={selectedCol}
            onChange={(e) => setSelectedCol(e.target.value)}
            className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200 min-w-[160px]"
          >
            <option value="">Select a column</option>
            <optgroup label="Numeric">
              {numericColumns.map((col) => (
                <option key={col} value={col}>{col}</option>
              ))}
            </optgroup>
            {allColumns.filter((c) => !numericColumns.includes(c)).length > 0 && (
              <optgroup label="Categorical">
                {allColumns.filter((c) => !numericColumns.includes(c)).map((col) => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </optgroup>
            )}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-[11px] text-text-tertiary">Bins: {bins}</label>
          <input
            type="range"
            min={5}
            max={50}
            value={bins}
            onChange={(e) => setBins(parseInt(e.target.value))}
            className="w-[120px] accent-plume-500"
          />
        </div>
      </div>

      {!selectedCol && (
        <p className="text-[13px] text-text-tertiary">Select a column to visualize</p>
      )}

      {loading && (
        <div className="flex items-center gap-2">
          <div className="w-[80px] h-[2px] bg-border rounded-full overflow-hidden">
            <motion.div
              className="h-full w-[40%] bg-plume-500 rounded-full"
              animate={{ x: ["-100%", "350%"] }}
              transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
        </div>
      )}

      {distribution && !loading && (
        <div className="bg-white dark:bg-neutral-800 rounded-xl shadow-sm p-4">
          <div className="flex justify-end mb-2">
            <ChartSizeToggle size={chartSize} onChange={setChartSize} />
          </div>
          <ResponsiveContainer width="100%" height={chartHeight}>
            <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 40, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                angle={-45}
                textAnchor="end"
                height={70}
                interval={chartData.length > 30 ? Math.floor(chartData.length / 15) : 0}
                tickFormatter={(v: string) => v.length > 12 ? v.slice(0, 11) + "\u2026" : v}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                label={{ value: "Count", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-text-tertiary)" }}
              />
              <Tooltip
                contentStyle={{ fontSize: 11, background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 6 }}
              />
              <Bar dataKey="count" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Scatter Plot Tab                                                  */
/* ------------------------------------------------------------------ */

function ScatterTab({ numericColumns }: { numericColumns: string[] }) {
  const [xCol, setXCol] = useState<string>("");
  const [yCol, setYCol] = useState<string>("");
  const [scatterData, setScatterData] = useState<ScatterData | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartHeight, chartSize, setChartSize] = useChartSize("M");

  useEffect(() => {
    if (!xCol || !yCol) {
      setScatterData(null);
      return;
    }
    setLoading(true);
    invoke<ScatterData>("get_scatter_data", { xCol, yCol })
      .then(setScatterData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [xCol, yCol]);

  const chartData = useMemo(() => {
    if (!scatterData) return [];
    return scatterData.x.map((xv, i) => ({ x: xv, y: scatterData.y[i] }));
  }, [scatterData]);

  return (
    <div className="max-w-[780px] mx-auto">
      <div className="flex items-center gap-4 mb-6">
        <div className="flex flex-col gap-1">
          <label className="text-[11px] text-text-tertiary">X axis</label>
          <select
            value={xCol}
            onChange={(e) => setXCol(e.target.value)}
            className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200 min-w-[160px]"
          >
            <option value="">Select column</option>
            {numericColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-[11px] text-text-tertiary">Y axis</label>
          <select
            value={yCol}
            onChange={(e) => setYCol(e.target.value)}
            className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200 min-w-[160px]"
          >
            <option value="">Select column</option>
            {numericColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      </div>

      {(!xCol || !yCol) && (
        <p className="text-[13px] text-text-tertiary">Select columns for both axes to visualize</p>
      )}

      {loading && (
        <div className="flex items-center gap-2">
          <div className="w-[80px] h-[2px] bg-border rounded-full overflow-hidden">
            <motion.div
              className="h-full w-[40%] bg-plume-500 rounded-full"
              animate={{ x: ["-100%", "350%"] }}
              transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
        </div>
      )}

      {scatterData && !loading && (
        <div className="bg-white dark:bg-neutral-800 rounded-xl shadow-sm p-4">
          <div className="flex justify-end mb-2">
            <ChartSizeToggle size={chartSize} onChange={setChartSize} />
          </div>
          <ResponsiveContainer width="100%" height={chartHeight}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 40, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="x"
                type="number"
                name={scatterData.x_label}
                tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                label={{
                  value: scatterData.x_label.length > 40 ? scatterData.x_label.slice(0, 39) + "\u2026" : scatterData.x_label,
                  position: "bottom",
                  offset: 10,
                  fontSize: 10,
                  fill: "var(--color-text-tertiary)",
                }}
              />
              <YAxis
                dataKey="y"
                type="number"
                name={scatterData.y_label}
                tick={{ fontSize: 10, fill: "var(--color-text-tertiary)" }}
                label={{
                  value: scatterData.y_label.length > 40 ? scatterData.y_label.slice(0, 39) + "\u2026" : scatterData.y_label,
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 10,
                  fill: "var(--color-text-tertiary)",
                }}
              />
              <Tooltip
                contentStyle={{ fontSize: 11, background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 6 }}
                formatter={(v) => Number(v).toFixed(3)}
              />
              <Scatter data={chartData} fill="#8b5cf6" fillOpacity={0.6} r={3} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Correlation Heatmap Tab                                           */
/* ------------------------------------------------------------------ */

function CorrelationTab({ numericColumns }: { numericColumns: string[] }) {
  const [matrix, setMatrix] = useState<CorrelationMatrix | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (numericColumns.length < 2) return;
    setLoading(true);
    setError(null);
    invoke<CorrelationMatrix>("get_correlation_matrix")
      .then(setMatrix)
      .catch((err) => {
        console.error("Correlation matrix failed:", err);
        setError(String(err));
      })
      .finally(() => setLoading(false));
  }, [numericColumns]);

  const getColor = useCallback((value: number): string => {
    // blue (-1) -> white (0) -> red (+1)
    if (value >= 0) {
      const intensity = Math.min(value, 1);
      const r = 255;
      const g = Math.round(255 - intensity * 120);
      const b = Math.round(255 - intensity * 120);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const intensity = Math.min(Math.abs(value), 1);
      const r = Math.round(255 - intensity * 120);
      const g = Math.round(255 - intensity * 120);
      const b = 255;
      return `rgb(${r}, ${g}, ${b})`;
    }
  }, []);

  if (numericColumns.length < 2) {
    return (
      <p className="text-[13px] text-text-tertiary">
        Need at least 2 numeric columns for a correlation matrix
      </p>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2">
        <div className="w-[80px] h-[2px] bg-border rounded-full overflow-hidden">
          <motion.div
            className="h-full w-[40%] bg-plume-500 rounded-full"
            animate={{ x: ["-100%", "350%"] }}
            transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
          />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-[12px] text-red-500">{error}</p>
    );
  }

  if (!matrix) return null;

  const n = matrix.columns.length;
  const cellSize = Math.max(40, Math.min(56, 600 / n));

  return (
    <div className="max-w-full mx-auto">
      {n > 15 && (
        <p className="text-[11px] text-text-tertiary mb-3">
          Showing all {n} numeric columns
        </p>
      )}

      <div className="bg-white dark:bg-neutral-800 rounded-xl shadow-sm p-4 overflow-auto">
        <div className="inline-block">
          {/* Top header row */}
          <div className="flex" style={{ marginLeft: cellSize + 4 }}>
            {matrix.columns.map((col) => (
              <div
                key={col}
                style={{ width: cellSize, height: cellSize }}
                className="flex items-end justify-center"
              >
                <span
                  className="text-[10px] text-text-tertiary whitespace-nowrap origin-bottom-left"
                  style={{ transform: "rotate(-45deg)", transformOrigin: "center", maxWidth: cellSize * 1.4 }}
                  title={col}
                >
                  {col.length > 8 ? col.slice(0, 7) + "\u2026" : col}
                </span>
              </div>
            ))}
          </div>

          {/* Grid rows */}
          {matrix.values.map((row, i) => (
            <div key={i} className="flex items-center">
              {/* Row label */}
              <div
                style={{ width: cellSize, minWidth: cellSize }}
                className="text-[10px] text-text-tertiary text-right pr-1 truncate"
                title={matrix.columns[i]}
              >
                {matrix.columns[i].length > 8 ? matrix.columns[i].slice(0, 7) + "\u2026" : matrix.columns[i]}
              </div>

              {/* Cells */}
              {row.map((value, j) => (
                <div
                  key={j}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: getColor(value),
                  }}
                  className="flex items-center justify-center border border-white/20 dark:border-neutral-700/30"
                  title={`${matrix.columns[i]} vs ${matrix.columns[j]}: ${value.toFixed(4)}`}
                >
                  <span className="text-[10px] font-medium" style={{
                    color: Math.abs(value) > 0.6 ? "white" : "var(--color-text-secondary)",
                  }}>
                    {value.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-2 mt-4">
          <span className="text-[10px] text-text-tertiary">-1</span>
          <div
            className="h-3 flex-1 max-w-[200px] rounded"
            style={{
              background: "linear-gradient(to right, rgb(135,135,255), rgb(255,255,255), rgb(255,135,135))",
            }}
          />
          <span className="text-[10px] text-text-tertiary">+1</span>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Box Plot Tab                                                      */
/* ------------------------------------------------------------------ */

function BoxPlotTab({ numericColumns, allColumns }: { numericColumns: string[]; allColumns: string[] }) {
  const [numericCol, setNumericCol] = useState<string>("");
  const [groupCol, setGroupCol] = useState<string>("");
  const [boxData, setBoxData] = useState<BoxPlotGroup[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartHeight, chartSize, setChartSize] = useChartSize("M");

  useEffect(() => {
    if (!numericCol || !groupCol) {
      setBoxData(null);
      return;
    }
    setLoading(true);
    invoke<BoxPlotGroup[]>("get_box_plot_data", { numericCol, groupCol })
      .then(setBoxData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [numericCol, groupCol]);

  return (
    <div className="max-w-[780px] mx-auto">
      <div className="flex items-center gap-4 mb-6">
        <div className="flex flex-col gap-1">
          <label className="text-[11px] text-text-tertiary">Numeric column</label>
          <select
            value={numericCol}
            onChange={(e) => setNumericCol(e.target.value)}
            className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200 min-w-[160px]"
          >
            <option value="">Select column</option>
            {numericColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-[11px] text-text-tertiary">Group by</label>
          <select
            value={groupCol}
            onChange={(e) => setGroupCol(e.target.value)}
            className="px-3 py-1.5 text-[12px] border border-border rounded-[var(--radius-default)] bg-surface text-text-primary outline-none focus:border-plume-500 transition-colors duration-200 min-w-[160px]"
          >
            <option value="">Select column</option>
            {allColumns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      </div>

      {(!numericCol || !groupCol) && (
        <p className="text-[13px] text-text-tertiary">Select a numeric column and a grouping column to visualize</p>
      )}

      {loading && (
        <div className="flex items-center gap-2">
          <div className="w-[80px] h-[2px] bg-border rounded-full overflow-hidden">
            <motion.div
              className="h-full w-[40%] bg-plume-500 rounded-full"
              animate={{ x: ["-100%", "350%"] }}
              transition={{ duration: 0.8, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
        </div>
      )}

      {boxData && !loading && (
        <div className="bg-white dark:bg-neutral-800 rounded-xl shadow-sm p-4">
          <div className="flex justify-end mb-2">
            <ChartSizeToggle size={chartSize} onChange={setChartSize} />
          </div>
          <BoxPlotSVG data={boxData} label={numericCol} height={chartHeight} />
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Custom SVG Box Plot                                               */
/* ------------------------------------------------------------------ */

function BoxPlotSVG({ data, label, height = 360 }: { data: BoxPlotGroup[]; label: string; height?: number }) {
  const width = 700;
  const marginTop = 20;
  const marginRight = 20;
  const marginBottom = 70;
  const marginLeft = 60;

  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;

  // Compute Y scale from all data points
  const allValues = data.flatMap((d) => [d.min, d.max, ...d.outliers]);
  const yMin = Math.min(...allValues);
  const yMax = Math.max(...allValues);
  const yPadding = (yMax - yMin) * 0.1 || 1;
  const scaleYMin = yMin - yPadding;
  const scaleYMax = yMax + yPadding;

  const toY = (v: number) => {
    return marginTop + plotHeight - ((v - scaleYMin) / (scaleYMax - scaleYMin)) * plotHeight;
  };

  const boxWidth = Math.min(48, plotWidth / data.length * 0.6);
  const groupSpacing = plotWidth / data.length;

  // Generate Y axis ticks
  const tickCount = 6;
  const yTicks = Array.from({ length: tickCount }, (_, i) => {
    return scaleYMin + ((scaleYMax - scaleYMin) * i) / (tickCount - 1);
  });

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
      {/* Y axis */}
      <line
        x1={marginLeft}
        y1={marginTop}
        x2={marginLeft}
        y2={marginTop + plotHeight}
        stroke="var(--color-border)"
        strokeWidth={1}
      />
      {yTicks.map((tick, i) => (
        <g key={i}>
          <line
            x1={marginLeft - 4}
            y1={toY(tick)}
            x2={marginLeft}
            y2={toY(tick)}
            stroke="var(--color-border)"
            strokeWidth={1}
          />
          <text
            x={marginLeft - 8}
            y={toY(tick)}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize={10}
            fill="var(--color-text-tertiary)"
          >
            {tick.toFixed(tick % 1 === 0 ? 0 : 1)}
          </text>
          {/* Grid lines */}
          <line
            x1={marginLeft}
            y1={toY(tick)}
            x2={marginLeft + plotWidth}
            y2={toY(tick)}
            stroke="var(--color-border)"
            strokeWidth={0.5}
            strokeDasharray="3 3"
          />
        </g>
      ))}

      {/* Y axis label */}
      <text
        x={14}
        y={marginTop + plotHeight / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={10}
        fill="var(--color-text-tertiary)"
        transform={`rotate(-90, 14, ${marginTop + plotHeight / 2})`}
      >
        {label}
      </text>

      {/* X axis */}
      <line
        x1={marginLeft}
        y1={marginTop + plotHeight}
        x2={marginLeft + plotWidth}
        y2={marginTop + plotHeight}
        stroke="var(--color-border)"
        strokeWidth={1}
      />

      {/* Box plots */}
      {data.map((group, i) => {
        const cx = marginLeft + groupSpacing * i + groupSpacing / 2;
        const halfBox = boxWidth / 2;

        return (
          <g key={group.group}>
            {/* Whisker line: min to max */}
            <line
              x1={cx}
              y1={toY(group.min)}
              x2={cx}
              y2={toY(group.max)}
              stroke="var(--color-text-tertiary)"
              strokeWidth={1}
            />

            {/* Min whisker cap */}
            <line
              x1={cx - halfBox / 2}
              y1={toY(group.min)}
              x2={cx + halfBox / 2}
              y2={toY(group.min)}
              stroke="var(--color-text-tertiary)"
              strokeWidth={1}
            />

            {/* Max whisker cap */}
            <line
              x1={cx - halfBox / 2}
              y1={toY(group.max)}
              x2={cx + halfBox / 2}
              y2={toY(group.max)}
              stroke="var(--color-text-tertiary)"
              strokeWidth={1}
            />

            {/* IQR box (Q1 to Q3) */}
            <rect
              x={cx - halfBox}
              y={toY(group.q3)}
              width={boxWidth}
              height={toY(group.q1) - toY(group.q3)}
              fill="#a78bfa"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              rx={2}
              opacity={0.8}
            />

            {/* Median line */}
            <line
              x1={cx - halfBox}
              y1={toY(group.median)}
              x2={cx + halfBox}
              y2={toY(group.median)}
              stroke="#6d28d9"
              strokeWidth={2}
            />

            {/* Outliers */}
            {group.outliers.map((o, j) => (
              <circle
                key={j}
                cx={cx}
                cy={toY(o)}
                r={2.5}
                fill="#8b5cf6"
                fillOpacity={0.5}
                stroke="#8b5cf6"
                strokeWidth={0.5}
              />
            ))}

            {/* Group label */}
            <text
              x={cx}
              y={marginTop + plotHeight + 12}
              textAnchor="end"
              fontSize={10}
              fill="var(--color-text-tertiary)"
              transform={`rotate(-45, ${cx}, ${marginTop + plotHeight + 12})`}
            >
              {group.group.length > 14 ? group.group.slice(0, 13) + "\u2026" : group.group}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
