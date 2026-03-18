import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { useAppStore } from "../stores/appStore";

const VALID_EXTENSIONS = ["csv", "tsv", "parquet", "xlsx", "xls"];

export function DropZone() {
  const [isDragging, setIsDragging] = useState(false);
  const loadFile = useAppStore((s) => s.loadFile);
  const error = useAppStore((s) => s.error);
  const clearError = useAppStore((s) => s.clearError);

  // Use Tauri's native drag-drop event for reliable file path access
  useEffect(() => {
    const webview = getCurrentWebviewWindow();
    const unlisten = webview.onDragDropEvent((event) => {
      if (event.payload.type === "over") {
        setIsDragging(true);
      } else if (event.payload.type === "leave") {
        setIsDragging(false);
      } else if (event.payload.type === "drop") {
        setIsDragging(false);
        clearError();
        const paths = event.payload.paths;
        if (paths.length > 0) {
          const path = paths[0];
          const ext = path.split(".").pop()?.toLowerCase() ?? "";
          if (VALID_EXTENSIONS.includes(ext)) {
            loadFile(path);
          } else {
            useAppStore.setState({
              error: `Unsupported file type ".${ext}". Plume supports CSV, TSV, Parquet, and Excel files.`,
            });
          }
        }
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, [loadFile, clearError]);

  const handleClick = async () => {
    clearError();
    const file = await open({
      multiple: false,
      filters: [
        { name: "Data files", extensions: ["csv", "tsv", "parquet", "xlsx", "xls"] },
      ],
    });
    if (file) {
      await loadFile(file);
    }
  };

  return (
    <div className="flex-1 flex items-center justify-center">
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="flex flex-col items-center gap-6"
        >
          <motion.div
            animate={isDragging ? { scale: 1.02 } : { scale: 1 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            onClick={handleClick}
            className={`
              flex flex-col items-center justify-center
              w-[400px] h-[240px] rounded-[var(--radius-default)]
              border-2 border-dashed cursor-pointer
              transition-colors duration-200 ease-out
              ${isDragging
                ? "border-plume-500 bg-plume-50 dark:bg-plume-500/10"
                : "border-border hover:border-text-tertiary"
              }
            `}
          >
            <p className="text-[15px] text-text-secondary">
              Drop a file here, or{" "}
              <span className="text-plume-600 dark:text-plume-500">browse</span>
            </p>
            <p className="text-[12px] text-text-tertiary mt-2">
              CSV, TSV, Parquet, or Excel
            </p>
          </motion.div>

          {error && (
            <motion.p
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-[13px] text-red-500 max-w-[400px] text-center"
            >
              {error}
            </motion.p>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
