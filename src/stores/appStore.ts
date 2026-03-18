import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import type { DataSummary, TablePage, TrainResult, View } from "../types/data";

interface AppState {
  view: View;
  setView: (view: View) => void;

  summary: DataSummary | null;
  tablePage: TablePage | null;

  isLoading: boolean;
  filePath: string | null;
  fileName: string | null;
  error: string | null;

  trainingResults: TrainResult[];
  addTrainingResult: (result: TrainResult) => void;

  loadFile: (path: string) => Promise<void>;
  resetFile: () => void;
  clearError: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  view: "data",
  setView: (view) => set({ view }),

  summary: null,
  tablePage: null,
  isLoading: false,
  filePath: null,
  fileName: null,
  error: null,

  trainingResults: [],
  addTrainingResult: (result) =>
    set((s) => ({ trainingResults: [result, ...s.trainingResults] })),

  clearError: () => set({ error: null }),

  resetFile: () =>
    set({
      summary: null,
      tablePage: null,
      filePath: null,
      fileName: null,
      error: null,
      view: "data",
      trainingResults: [],
    }),

  loadFile: async (path: string) => {
    const name = path.split("/").pop()?.split("\\").pop() ?? "dataset";
    set({ isLoading: true, fileName: name, error: null });

    // Give React time to paint the loading UI
    await new Promise((r) => setTimeout(r, 100));

    try {
      const lowerPath = path.toLowerCase();
      const isParquet = lowerPath.endsWith(".parquet");
      const isExcel = lowerPath.endsWith(".xlsx") || lowerPath.endsWith(".xls");
      const summary = isParquet
        ? await invoke<DataSummary>("load_parquet", { path })
        : isExcel
          ? await invoke<DataSummary>("load_excel", { path })
          : await invoke<DataSummary>("load_csv", { path });

      const page = await invoke<TablePage>("get_table_page", {
        offset: 0,
        limit: 100,
      });

      set({
        summary,
        tablePage: page,
        filePath: path,
        isLoading: false,
        view: "data",
        trainingResults: [],
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("Failed to load file:", message);
      set({ isLoading: false, error: message, fileName: null });
    }
  },
}));
