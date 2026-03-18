import { AnimatePresence, motion } from "framer-motion";
import { useAppStore } from "./stores/appStore";
import { Sidebar } from "./components/Sidebar";
import { DropZone } from "./components/DropZone";
import { DataTable } from "./components/DataTable";
import { ShapeView } from "./components/ShapeView";
import { ModelView } from "./components/ModelView";
import { ResultsView } from "./components/ResultsView";

function App() {
  const view = useAppStore((s) => s.view);
  const summary = useAppStore((s) => s.summary);
  const isLoading = useAppStore((s) => s.isLoading);
  const fileName = useAppStore((s) => s.fileName);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-surface">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden relative">
        {/* Header — draggable title bar region */}
        <div
          className="h-12 min-h-12 flex items-center justify-center border-b border-border"
          data-tauri-drag-region
        >
          <span className="text-[13px] text-text-secondary">
            {fileName ?? "Plume"}
          </span>
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
            {view === "model" && <ModelView />}
            {view === "results" && <ResultsView />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
