import { motion } from "framer-motion";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../stores/appStore";
import type { View } from "../types/data";

const navItems: { id: View; label: string }[] = [
  { id: "data", label: "Data" },
  { id: "shape", label: "Shape" },
  { id: "visualize", label: "Visualize" },
  { id: "model", label: "Model" },
  { id: "results", label: "Results" },
];

export function Sidebar() {
  const view = useAppStore((s) => s.view);
  const setView = useAppStore((s) => s.setView);
  const summary = useAppStore((s) => s.summary);
  const loadFile = useAppStore((s) => s.loadFile);

  const handleNewFile = async () => {
    const file = await open({
      multiple: false,
      filters: [
        { name: "Data files", extensions: ["csv", "tsv", "parquet"] },
      ],
    });
    if (file) {
      await loadFile(file);
    }
  };

  return (
    <nav className="flex flex-col w-[140px] min-w-[140px] border-r border-border bg-surface-alt pt-12 pb-4 px-3">
      <div className="flex flex-col gap-1 mt-4">
        {navItems.map((item) => {
          const isActive = view === item.id;
          const isDisabled = !summary && item.id !== "data";

          return (
            <button
              key={item.id}
              onClick={() => !isDisabled && setView(item.id)}
              className={`
                relative text-left px-3 py-2 rounded-[var(--radius-default)] text-[14px] font-medium
                transition-colors duration-200 ease-out
                ${isDisabled ? "text-text-tertiary cursor-default" : "cursor-pointer"}
                ${!isActive && !isDisabled ? "text-text-secondary hover:text-text-primary hover:bg-surface" : ""}
              `}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute inset-0 bg-white dark:bg-neutral-800 rounded-[var(--radius-default)] shadow-sm"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <span className="relative z-10">{item.label}</span>
            </button>
          );
        })}
      </div>

      <div className="mt-auto px-3">
        {summary && (
          <div className="text-[11px] text-text-tertiary leading-relaxed mb-3">
            <div>{summary.rows.toLocaleString()} rows</div>
            <div>{summary.columns} columns</div>
          </div>
        )}
        {summary && (
          <button
            onClick={handleNewFile}
            className="text-[11px] text-text-tertiary hover:text-text-primary transition-colors duration-200 cursor-pointer"
          >
            Load new file
          </button>
        )}
      </div>
    </nav>
  );
}
