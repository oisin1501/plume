import { useState } from "react";

export type ChartSize = "S" | "M" | "L";

const HEIGHTS: Record<ChartSize, number> = {
  S: 180,
  M: 300,
  L: 440,
};

export function useChartSize(defaultSize: ChartSize = "M"): [number, ChartSize, (s: ChartSize) => void] {
  const [size, setSize] = useState<ChartSize>(defaultSize);
  return [HEIGHTS[size], size, setSize];
}

export function ChartSizeToggle({
  size,
  onChange,
}: {
  size: ChartSize;
  onChange: (s: ChartSize) => void;
}) {
  const sizes: ChartSize[] = ["S", "M", "L"];
  return (
    <div className="flex items-center gap-0.5">
      {sizes.map((s) => (
        <button
          key={s}
          onClick={() => onChange(s)}
          className={`px-1.5 py-0.5 text-[10px] rounded transition-colors duration-150 cursor-pointer ${
            size === s
              ? "bg-plume-500 text-white"
              : "text-text-tertiary hover:text-text-secondary hover:bg-surface-alt"
          }`}
        >
          {s}
        </button>
      ))}
    </div>
  );
}
