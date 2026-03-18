use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

mod data;
mod ml;

pub struct AppState {
    pub dataframe: Mutex<Option<DataFrame>>,
    pub history: Mutex<Vec<DataFrame>>,
    pub file_path: Mutex<Option<String>>,
    pub transform_log: Mutex<Vec<TransformEntry>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TransformEntry {
    pub action: String,
    pub column: Option<String>,
    pub detail: String,
}

#[derive(Serialize, Deserialize)]
pub struct DataSummary {
    pub rows: usize,
    pub columns: usize,
    pub column_names: Vec<String>,
    pub column_types: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ColumnProfile {
    pub name: String,
    pub dtype: String,
    pub count: usize,
    pub null_count: usize,
    pub null_percent: f64,
    pub unique_count: Option<usize>,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<String>,
    pub max: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct TablePage {
    pub rows: Vec<Vec<serde_json::Value>>,
    pub total_rows: usize,
    pub columns: Vec<String>,
    pub column_types: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ColumnDistribution {
    pub labels: Vec<String>,
    pub counts: Vec<usize>,
    pub kind: String, // "histogram" or "frequency"
}

#[derive(Serialize, Deserialize)]
pub struct TypeRecommendation {
    pub column: String,
    pub current_type: String,
    pub recommended_type: String,
    pub reason: String,
}

#[derive(Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub columns: Vec<String>,
    pub values: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
pub struct ScatterData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub x_label: String,
    pub y_label: String,
}

#[derive(Serialize, Deserialize)]
pub struct BoxPlotGroup {
    pub group: String,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
}

#[tauri::command]
async fn load_csv(path: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let path_clone = path.clone();

    // Run heavy CSV parsing on a blocking thread so the UI stays responsive
    let (df, summary) = tokio::task::spawn_blocking(move || {
        let df = data::read_csv(&path_clone)?;
        if df.height() == 0 {
            return Err("File is empty or contains no data rows".to_string());
        }
        if df.width() == 0 {
            return Err("File contains no columns".to_string());
        }
        let summary = data::get_summary(&df);
        Ok::<_, String>((df, summary))
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))??;

    *state.dataframe.lock().unwrap_or_else(|p| p.into_inner()) = Some(df);
    *state.file_path.lock().unwrap_or_else(|p| p.into_inner()) = Some(path);
    state.history.lock().unwrap_or_else(|p| p.into_inner()).clear();
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).clear();
    Ok(summary)
}

#[tauri::command]
async fn get_table_page(offset: usize, limit: usize, state: State<'_, AppState>) -> Result<TablePage, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    data::get_page(df, offset, limit)
}

#[tauri::command]
async fn get_column_profile(column_name: String, state: State<'_, AppState>) -> Result<ColumnProfile, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    data::profile_column(df, &column_name)
}

#[tauri::command]
async fn get_summary(state: State<'_, AppState>) -> Result<DataSummary, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    Ok(data::get_summary(df))
}

#[tauri::command]
async fn get_all_column_profiles(state: State<'_, AppState>) -> Result<Vec<ColumnProfile>, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    data::profile_all_columns(df)
}

#[tauri::command]
async fn fill_missing(column: String, strategy: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;

    // Save current state to history
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());

    let new_df = data::fill_missing_values(df, &column, &strategy)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Fill missing".into(),
        column: Some(column),
        detail: strategy,
    });
    Ok(summary)
}

#[tauri::command]
async fn drop_column(column: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::drop_col(df, &column)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Drop column".into(),
        column: Some(column),
        detail: String::new(),
    });
    Ok(summary)
}

#[tauri::command]
async fn cast_column(column: String, target_type: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::cast_col(df, &column, &target_type)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Cast type".into(),
        column: Some(column),
        detail: target_type,
    });
    Ok(summary)
}

#[tauri::command]
async fn cast_columns_batch(
    casts: Vec<(String, String)>,
    state: State<'_, AppState>,
) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let mut new_df = df.clone();
    for (column, target_type) in &casts {
        new_df = data::cast_col(&new_df, column, target_type)?;
    }
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Batch cast".into(),
        column: None,
        detail: format!("{} columns", casts.len()),
    });
    Ok(summary)
}

#[tauri::command]
async fn fill_mode(column: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::fill_mode(df, &column)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Fill mode".into(),
        column: Some(column),
        detail: "most common".into(),
    });
    Ok(summary)
}

#[tauri::command]
async fn one_hot_encode(column: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::one_hot_encode(df, &column)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "One-hot encode".into(),
        column: Some(column),
        detail: String::new(),
    });
    Ok(summary)
}

#[tauri::command]
async fn rename_column(old_name: String, new_name: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::rename_col(df, &old_name, &new_name)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Rename".into(),
        column: Some(old_name),
        detail: new_name,
    });
    Ok(summary)
}

#[tauri::command]
async fn sort_column(column: String, descending: bool, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::sort_by_column(df, &column, descending)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Sort".into(),
        column: Some(column),
        detail: if descending { "desc".into() } else { "asc".into() },
    });
    Ok(summary)
}

#[tauri::command]
async fn filter_rows(column: String, query: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::filter_rows(df, &column, &query)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Filter".into(),
        column: Some(column),
        detail: query,
    });
    Ok(summary)
}

#[tauri::command]
async fn math_columns(col_a: String, col_b: String, op: String, new_name: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::math_columns(df, &col_a, &col_b, &op, &new_name)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Math columns".into(),
        column: Some(new_name),
        detail: format!("{} {} {}", col_a, op, col_b),
    });
    Ok(summary)
}

#[tauri::command]
async fn transform_column(column: String, transform: String, new_name: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::transform_column(df, &column, &transform, &new_name)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Transform".into(),
        column: Some(column),
        detail: format!("{} -> {}", transform, new_name),
    });
    Ok(summary)
}

#[tauri::command]
async fn bin_column(column: String, n_bins: usize, new_name: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    state.history.lock().unwrap_or_else(|p| p.into_inner()).push(df.clone());
    let new_df = data::bin_column(df, &column, n_bins, &new_name)?;
    let summary = data::get_summary(&new_df);
    *guard = Some(new_df);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).push(TransformEntry {
        action: "Bin column".into(),
        column: Some(column),
        detail: format!("{} bins -> {}", n_bins, new_name),
    });
    Ok(summary)
}

#[tauri::command]
async fn undo_transform(state: State<'_, AppState>) -> Result<DataSummary, String> {
    let mut history = state.history.lock().unwrap_or_else(|p| p.into_inner());
    let prev = history.pop().ok_or("Nothing to undo")?;
    let summary = data::get_summary(&prev);
    *state.dataframe.lock().unwrap_or_else(|p| p.into_inner()) = Some(prev);
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).pop();
    Ok(summary)
}

#[tauri::command]
async fn get_history_length(state: State<'_, AppState>) -> Result<usize, String> {
    Ok(state.history.lock().unwrap_or_else(|p| p.into_inner()).len())
}

#[tauri::command]
async fn get_transform_log(state: State<'_, AppState>) -> Result<Vec<TransformEntry>, String> {
    Ok(state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).clone())
}

#[tauri::command]
async fn load_parquet(path: String, state: State<'_, AppState>) -> Result<DataSummary, String> {
    let path_clone = path.clone();
    let df = tokio::task::spawn_blocking(move || data::read_parquet(&path_clone))
        .await.map_err(|e| format!("Task failed: {}", e))??;
    let summary = data::get_summary(&df);
    *state.dataframe.lock().unwrap_or_else(|p| p.into_inner()) = Some(df);
    *state.file_path.lock().unwrap_or_else(|p| p.into_inner()) = Some(path);
    state.history.lock().unwrap_or_else(|p| p.into_inner()).clear();
    state.transform_log.lock().unwrap_or_else(|p| p.into_inner()).clear();
    Ok(summary)
}

#[tauri::command]
async fn get_column_distribution(column_name: String, state: State<'_, AppState>) -> Result<ColumnDistribution, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    data::column_distribution(df, &column_name)
}

#[tauri::command]
async fn get_type_recommendations(state: State<'_, AppState>) -> Result<Vec<TypeRecommendation>, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    Ok(data::type_recommendations(df))
}

#[tauri::command]
async fn train_model(
    task: String,
    target: Option<String>,
    features: Vec<String>,
    algorithm: String,
    n_clusters: Option<usize>,
    hyperparams: Option<serde_json::Value>,
    use_cv: Option<bool>,
    cv_folds: Option<usize>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // Export current in-memory DataFrame to temp CSV so Python sees all transforms
    let temp_path = {
        let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
        let df = guard.as_mut().ok_or("No dataset loaded")?;
        data::export_to_temp_csv(df)?
    };

    let command = if task == "clustering" {
        serde_json::json!({
            "action": "train_clustering",
            "params": {
                "path": temp_path,
                "separator": ",",
                "features": features,
                "algorithm": algorithm,
                "n_clusters": n_clusters.unwrap_or(3),
            }
        })
    } else {
        serde_json::json!({
            "action": "train_supervised",
            "params": {
                "path": temp_path,
                "separator": ",",
                "target": target,
                "features": features,
                "task": task,
                "algorithm": algorithm,
                "hyperparams": hyperparams.unwrap_or(serde_json::json!({})),
                "use_cv": use_cv.unwrap_or(false),
                "cv_folds": cv_folds.unwrap_or(5),
            }
        })
    };

    tokio::task::spawn_blocking(move || {
        ml::run_ml_command(command)
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

#[tauri::command]
async fn export_model_pickle(
    task: String,
    target: Option<String>,
    features: Vec<String>,
    algorithm: String,
    hyperparams: Option<serde_json::Value>,
    output_path: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let temp_path = {
        let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
        let df = guard.as_mut().ok_or("No dataset loaded")?;
        data::export_to_temp_csv(df)?
    };

    let command = serde_json::json!({
        "action": "export_model",
        "params": {
            "path": temp_path,
            "separator": ",",
            "task": task,
            "target": target,
            "features": features,
            "algorithm": algorithm,
            "hyperparams": hyperparams.unwrap_or(serde_json::json!({})),
            "output_path": output_path,
        }
    });

    tokio::task::spawn_blocking(move || ml::run_ml_command(command))
        .await.map_err(|e| format!("Task failed: {}", e))?
}

#[tauri::command]
async fn generate_report(
    results: serde_json::Value,
    output_path: String,
) -> Result<serde_json::Value, String> {
    let command = serde_json::json!({
        "action": "generate_report",
        "params": {
            "results": results,
            "output_path": output_path,
        }
    });

    tokio::task::spawn_blocking(move || ml::run_ml_command(command))
        .await.map_err(|e| format!("Task failed: {}", e))?
}

#[tauri::command]
async fn compute_shap(
    task: String,
    target: Option<String>,
    features: Vec<String>,
    algorithm: String,
    hyperparams: Option<serde_json::Value>,
    n_samples: Option<usize>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let temp_path = {
        let mut guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
        let df = guard.as_mut().ok_or("No dataset loaded")?;
        data::export_to_temp_csv(df)?
    };

    let command = serde_json::json!({
        "action": "compute_shap",
        "params": {
            "path": temp_path,
            "separator": ",",
            "task": task,
            "target": target,
            "features": features,
            "algorithm": algorithm,
            "hyperparams": hyperparams.unwrap_or(serde_json::json!({})),
            "n_samples": n_samples.unwrap_or(5),
        }
    });

    tokio::task::spawn_blocking(move || ml::run_ml_command(command))
        .await.map_err(|e| format!("Task failed: {}", e))?
}

#[tauri::command]
async fn get_correlation_matrix(state: State<'_, AppState>) -> Result<CorrelationMatrix, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    data::correlation_matrix(df)
}

#[tauri::command]
async fn get_scatter_data(x_col: String, y_col: String, state: State<'_, AppState>) -> Result<ScatterData, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    data::scatter_data(df, &x_col, &y_col)
}

#[tauri::command]
async fn get_box_plot_data(numeric_col: String, group_col: String, state: State<'_, AppState>) -> Result<Vec<BoxPlotGroup>, String> {
    let guard = state.dataframe.lock().unwrap_or_else(|p| p.into_inner());
    let df = guard.as_ref().ok_or("No dataset loaded")?;
    data::box_plot_data(df, &numeric_col, &group_col)
}

#[tauri::command]
async fn save_text_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| format!("Failed to write file: {}", e))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            dataframe: Mutex::new(None),
            history: Mutex::new(Vec::new()),
            file_path: Mutex::new(None),
            transform_log: Mutex::new(Vec::new()),
        })
        .invoke_handler(tauri::generate_handler![
            load_csv,
            load_parquet,
            get_table_page,
            get_column_profile,
            get_summary,
            get_all_column_profiles,
            fill_missing,
            drop_column,
            cast_column,
            cast_columns_batch,
            fill_mode,
            one_hot_encode,
            rename_column,
            sort_column,
            filter_rows,
            math_columns,
            transform_column,
            bin_column,
            undo_transform,
            get_transform_log,
            get_history_length,
            get_column_distribution,
            get_type_recommendations,
            train_model,
            export_model_pickle,
            generate_report,
            compute_shap,
            get_correlation_matrix,
            get_scatter_data,
            get_box_plot_data,
            save_text_file,
        ])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
