export interface DataSummary {
  rows: number;
  columns: number;
  column_names: string[];
  column_types: string[];
}

export interface TablePage {
  rows: (string | number | boolean | null)[][];
  total_rows: number;
  columns: string[];
  column_types: string[];
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  count: number;
  null_count: number;
  null_percent: number;
  unique_count: number | null;
  mean: number | null;
  std: number | null;
  min: string | null;
  max: string | null;
}

export type View = "data" | "shape" | "visualize" | "model" | "results";

export interface CorrelationMatrix {
  columns: string[];
  values: number[][]; // NxN
}

export interface ScatterData {
  x: number[];
  y: number[];
  x_label: string;
  y_label: string;
}

export interface BoxPlotGroup {
  group: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers: number[];
}

export interface TransformStep {
  id: string;
  label: string;
  type: "fill_missing" | "drop_column" | "cast_column" | "one_hot_encode";
  params: Record<string, string>;
}

export interface AllColumnProfiles {
  profiles: ColumnProfile[];
}

export interface ColumnDistribution {
  labels: string[];
  counts: number[];
  kind: "histogram" | "frequency";
}

export interface TypeRecommendation {
  column: string;
  current_type: string;
  recommended_type: string;
  reason: string;
}

export interface TrainResult {
  status: string;
  task: string;
  algorithm: string;
  target?: string | null;
  hyperparams?: Record<string, any>;
  metrics: Record<string, any>;
  feature_importance?: { feature: string; importance: number }[];
  clusters?: { cluster: number; size: number; characteristics: string[] }[];
  features_used?: string[];
  train_size?: number;
  test_size?: number;
  cv_scores?: {
    scores: number[];
    mean: number;
    std: number;
    metric: string;
    folds: number;
  };
  roc_curve?: {
    fpr: number[];
    tpr: number[];
    auc: number | null;
  };
  residuals?: {
    y_true: number[];
    y_pred: number[];
  };
  scatter?: {
    x: number[];
    y: number[];
    labels: number[];
    x_label: string;
    y_label: string;
    explained_variance: number[];
  };
  predictions?: number[];
}
