import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { useAppStore } from "./stores/appStore";
import { Sidebar } from "./components/Sidebar";
import { DropZone } from "./components/DropZone";
import { DataTable } from "./components/DataTable";
import { ShapeView } from "./components/ShapeView";
import { ModelView } from "./components/ModelView";
import { ResultsView } from "./components/ResultsView";
import { VisualizeView } from "./components/VisualizeView";

const VALID_EXTENSIONS = ["csv", "tsv", "parquet", "xlsx", "xls"];

function App() {
  const view = useAppStore((s) => s.view);
  const summary = useAppStore((s) => s.summary);
  const isLoading = useAppStore((s) => s.isLoading);
  const fileName = useAppStore((s) => s.fileName);

  // App-level drag-drop: works even when a dataset is already loaded
  useEffect(() => {
    const webview = getCurrentWebviewWindow();
    const unlisten = webview.onDragDropEvent((event) => {
      if (event.payload.type === "drop") {
        const paths = event.payload.paths;
        if (paths.length > 0) {
          const path = paths[0];
          const ext = path.split(".").pop()?.toLowerCase() ?? "";
          if (VALID_EXTENSIONS.includes(ext)) {
            useAppStore.getState().loadFile(path);
          } else {
            useAppStore.setState({
              error: `Unsupported file type ".${ext}". Plume supports CSV, TSV, Parquet, and Excel files.`,
            });
          }
        }
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden relative">
        {/* Header — draggable title bar region */}
        <div
          className="h-12 min-h-12 flex items-center justify-center border-b border-border gap-2"
          data-tauri-drag-region
        >
          <span className="text-[13px] font-medium text-text-tertiary">Plume</span>
          {fileName && (
            <>
              <span className="text-[11px] text-text-tertiary/40">·</span>
              <span className="text-[13px] text-text-primary">{fileName}</span>
              {summary && (
                <>
                  <span className="text-[11px] text-text-tertiary/40">·</span>
                  <span className="text-[11px] text-text-tertiary">{summary.rows.toLocaleString()} rows</span>
                </>
              )}
            </>
          )}
        </div>

        {/* Loading overlay */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-surface/80 backdrop-blur-sm"
            >
              <p className="text-[13px] text-text-secondary mb-6">
                Loading {fileName ?? "data"}
              </p>
              <div className="w-[240px] h-[3px] bg-border rounded-full overflow-hidden">
                <motion.div
                  className="h-full w-[40%] bg-plume-500 rounded-full"
                  initial={{ x: "-100%" }}
                  animate={{ x: "350%" }}
                  transition={{
                    duration: 1.2,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="flex-1 flex flex-col overflow-hidden"
          >
            {view === "data" && (summary ? <DataTable /> : <DropZone />)}
            {view === "shape" && <ShapeView />}
            {view === "visualize" && <VisualizeView />}
            {view === "model" && <ModelView />}
            {view === "results" && <ResultsView />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
