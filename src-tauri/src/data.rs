use polars::prelude::*;
use polars::io::parquet::read::ParquetReader;
use crate::{BoxPlotGroup, ColumnProfile, CorrelationMatrix, DataSummary, ScatterData, TablePage};

/// Export the current in-memory DataFrame to a temp CSV file.
/// Returns the path to the temp file.
pub fn export_to_temp_csv(df: &mut DataFrame) -> Result<String, String> {
    let dir = std::env::temp_dir();
    let path = dir.join("plume_current_data.csv");
    let path_str = path.to_string_lossy().to_string();
    let file = std::fs::File::create(&path)
        .map_err(|e| format!("Failed to create temp file: {}", e))?;
    CsvWriter::new(file)
        .finish(df)
        .map_err(|e| format!("Failed to write temp CSV: {}", e))?;
    Ok(path_str)
}

pub fn read_csv(path: &str) -> Result<DataFrame, String> {
    // Detect separator by reading the first few lines
    let sample = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to open file: {}", e))?
        .chars()
        .take(8192)
        .collect::<String>();

    let separator = detect_separator(&sample);

    CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(10000))
        .map_parse_options(|opts| opts.with_separator(separator).with_try_parse_dates(true))
        .try_into_reader_with_file_path(Some(path.into()))
        .map_err(|e| format!("Failed to open file: {}", e))?
        .finish()
        .map_err(|e| format!("Failed to parse CSV: {}", e))
}

fn detect_separator(sample: &str) -> u8 {
    let first_line = sample.lines().next().unwrap_or("");

    // Count candidate separators in the header line
    let candidates: &[(u8, char)] = &[
        (b',', ','),
        (b'\t', '\t'),
        (b';', ';'),
        (b'|', '|'),
    ];

    let mut best = b',';
    let mut best_count = 0;

    for &(byte, ch) in candidates {
        let count = first_line.chars().filter(|&c| c == ch).count();
        if count > best_count {
            best_count = count;
            best = byte;
        }
    }

    best
}

pub fn get_summary(df: &DataFrame) -> DataSummary {
    DataSummary {
        rows: df.height(),
        columns: df.width(),
        column_names: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        column_types: df.dtypes().iter().map(|d| format!("{}", d)).collect(),
    }
}

pub fn get_page(df: &DataFrame, offset: usize, limit: usize) -> Result<TablePage, String> {
    let total = df.height();
    let actual_offset = offset.min(total);
    let actual_limit = limit.min(total.saturating_sub(actual_offset));

    let sliced = df.slice(actual_offset as i64, actual_limit);

    let mut rows: Vec<Vec<serde_json::Value>> = Vec::new();

    for i in 0..sliced.height() {
        let mut row = Vec::new();
        for col in sliced.columns() {
            let val = col.as_materialized_series().get(i).map_err(|e| format!("Error reading cell: {}", e))?;
            row.push(anyvalue_to_json(&val));
        }
        rows.push(row);
    }

    Ok(TablePage {
        rows,
        total_rows: total,
        columns: df.get_column_names().iter().map(|s| s.to_string()).collect(),
        column_types: df.dtypes().iter().map(|d| format!("{}", d)).collect(),
    })
}

pub fn profile_column(df: &DataFrame, name: &str) -> Result<ColumnProfile, String> {
    let col = df.column(name).map_err(|e| format!("Column not found: {}", e))?;
    let series = col.as_materialized_series();
    let dtype = format!("{}", series.dtype());
    let count = series.len();
    let null_count = series.null_count();
    let null_percent = if count > 0 {
        (null_count as f64 / count as f64) * 100.0
    } else {
        0.0
    };

    let unique_count = series.n_unique().ok();

    let (mean, std, min, max) = if series.dtype().is_numeric() {
        (
            series.mean(),
            series.std(1),
            series.min::<f64>().ok().flatten().map(|v| format!("{:.4}", v)),
            series.max::<f64>().ok().flatten().map(|v| format!("{:.4}", v)),
        )
    } else {
        (None, None, None, None)
    };

    Ok(ColumnProfile {
        name: name.to_string(),
        dtype,
        count,
        null_count,
        null_percent,
        unique_count,
        mean,
        std,
        min,
        max,
    })
}

pub fn profile_all_columns(df: &DataFrame) -> Result<Vec<crate::ColumnProfile>, String> {
    let mut profiles = Vec::new();
    for name in df.get_column_names() {
        profiles.push(profile_column(df, &name.to_string())?);
    }
    Ok(profiles)
}

pub fn fill_missing_values(df: &DataFrame, column: &str, strategy: &str) -> Result<DataFrame, String> {
    let col = df.column(column).map_err(|e| format!("{}", e))?;
    let series = col.as_materialized_series();

    let filled = match strategy {
        "mean" => {
            series.fill_null(FillNullStrategy::Mean)
                .map_err(|e| format!("Fill failed: {}", e))?
        }
        "median" => {
            let median_val = series.median().ok_or("Cannot compute median for this column")?;
            let fill_series = Series::new(series.name().clone(), &[median_val]);
            series.fill_null(FillNullStrategy::Zero)
                .and_then(|s| {
                    // Use a different approach: replace nulls with median
                    let mask = series.is_null();
                    let median_broadcast = fill_series.new_from_index(0, series.len());
                    s.zip_with(&mask, &median_broadcast)
                })
                .map_err(|e| format!("Fill failed: {}", e))?
        }
        "forward" => {
            series.fill_null(FillNullStrategy::Forward(None))
                .map_err(|e| format!("Fill failed: {}", e))?
        }
        "zero" => {
            series.fill_null(FillNullStrategy::Zero)
                .map_err(|e| format!("Fill failed: {}", e))?
        }
        "drop" => {
            // Drop rows with null in this column — handled differently
            let mask = col.as_materialized_series().is_not_null();
            return df.filter(&mask).map_err(|e| format!("Drop rows failed: {}", e));
        }
        _ => return Err(format!("Unknown strategy: {}", strategy)),
    };

    let mut new_df = df.clone();
    new_df.replace(column, filled.into()).map_err(|e| format!("Replace failed: {}", e))?;
    Ok(new_df)
}

pub fn drop_col(df: &DataFrame, column: &str) -> Result<DataFrame, String> {
    df.drop(column).map_err(|e| format!("Drop column failed: {}", e))
}

pub fn cast_col(df: &DataFrame, column: &str, target_type: &str) -> Result<DataFrame, String> {
    let dtype = match target_type {
        "f64" | "float" => DataType::Float64,
        "i64" | "integer" => DataType::Int64,
        "str" | "string" | "text" => DataType::String,
        "bool" | "boolean" => DataType::Boolean,
        "category" => DataType::String, // cast to string first; categorical encoding handled separately
        _ => return Err(format!("Unknown type: {}", target_type)),
    };

    let col = df.column(column).map_err(|e| format!("{}", e))?;
    let casted = col.as_materialized_series()
        .cast(&dtype)
        .map_err(|e| format!("Cast failed: {}", e))?;

    let mut new_df = df.clone();
    new_df.replace(column, casted.into()).map_err(|e| format!("Replace failed: {}", e))?;
    Ok(new_df)
}

pub fn column_distribution(df: &DataFrame, name: &str) -> Result<crate::ColumnDistribution, String> {
    let col = df.column(name).map_err(|e| format!("Column not found: {}", e))?;
    let series = col.as_materialized_series();

    if series.dtype().is_numeric() {
        // Histogram: compute bins
        let non_null = series.drop_nulls();
        if non_null.len() == 0 {
            return Ok(crate::ColumnDistribution {
                labels: vec![],
                counts: vec![],
                kind: "histogram".to_string(),
            });
        }

        let min_val = non_null.min::<f64>().ok().flatten().unwrap_or(0.0);
        let max_val = non_null.max::<f64>().ok().flatten().unwrap_or(0.0);

        let n_bins: usize = 20;

        if (max_val - min_val).abs() < f64::EPSILON {
            return Ok(crate::ColumnDistribution {
                labels: vec![format!("{:.2}", min_val)],
                counts: vec![non_null.len()],
                kind: "histogram".to_string(),
            });
        }

        let bin_width = (max_val - min_val) / n_bins as f64;
        let mut counts = vec![0usize; n_bins];
        let mut labels = Vec::with_capacity(n_bins);

        for i in 0..n_bins {
            let low = min_val + i as f64 * bin_width;
            labels.push(format!("{:.1}", low));
        }

        let casted = non_null.cast(&DataType::Float64).map_err(|e| format!("{}", e))?;
        let ca = casted.f64().map_err(|e| format!("{}", e))?;
        for val in ca.into_no_null_iter() {
            let mut bin = ((val - min_val) / bin_width) as usize;
            if bin >= n_bins {
                bin = n_bins - 1;
            }
            counts[bin] += 1;
        }

        Ok(crate::ColumnDistribution { labels, counts, kind: "histogram".to_string() })
    } else {
        // Frequency: top 15 value counts
        let non_null = series.drop_nulls();
        let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        if let Ok(str_col) = non_null.str() {
            for val in str_col.into_iter().flatten() {
                *freq.entry(val.to_string()).or_insert(0) += 1;
            }
        } else {
            for i in 0..non_null.len() {
                if let Ok(val) = non_null.get(i) {
                    let s = format!("{}", val);
                    *freq.entry(s).or_insert(0) += 1;
                }
            }
        }

        let mut pairs: Vec<(String, usize)> = freq.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(15);

        let labels = pairs.iter().map(|(k, _)| k.clone()).collect();
        let counts = pairs.iter().map(|(_, v)| *v).collect();

        Ok(crate::ColumnDistribution { labels, counts, kind: "frequency".to_string() })
    }
}

pub fn read_parquet(path: &str) -> Result<DataFrame, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    ParquetReader::new(file)
        .finish()
        .map_err(|e| format!("Failed to parse Parquet: {}", e))
}

pub fn sort_by_column(df: &DataFrame, column: &str, descending: bool) -> Result<DataFrame, String> {
    let options = SortMultipleOptions::new().with_order_descending(descending);
    df.sort([column], options)
        .map_err(|e| format!("Sort failed: {}", e))
}

pub fn filter_rows(df: &DataFrame, column: &str, query: &str) -> Result<DataFrame, String> {
    let col = df.column(column).map_err(|e| format!("{}", e))?;
    let series = col.as_materialized_series();

    let mask = if series.dtype().is_numeric() {
        if let Ok(val) = query.parse::<f64>() {
            series.equal(val).map_err(|e| format!("{}", e))?
        } else {
            // Fall back to string-based search
            let str_series = series.cast(&DataType::String).map_err(|e| format!("{}", e))?;
            let ca = str_series.str().map_err(|e| format!("{}", e))?;
            ca.into_iter()
                .map(|opt| opt.map_or(false, |s| s.contains(query)))
                .collect::<BooleanChunked>()
        }
    } else {
        let ca = series.str().map_err(|e| format!("{}", e))?;
        ca.into_iter()
            .map(|opt| opt.map_or(false, |s| s.contains(query)))
            .collect::<BooleanChunked>()
    };

    df.filter(&mask).map_err(|e| format!("Filter failed: {}", e))
}

pub fn fill_mode(df: &DataFrame, column: &str) -> Result<DataFrame, String> {
    let col = df.column(column).map_err(|e| format!("{}", e))?;
    let series = col.as_materialized_series();

    let non_null = series.drop_nulls();
    if non_null.len() == 0 {
        return Err("Cannot compute mode: all values are null".to_string());
    }

    // Find mode via string representation counting
    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for i in 0..non_null.len() {
        let val = non_null.get(i).map_err(|e| format!("{}", e))?;
        *freq.entry(format!("{}", val)).or_insert(0) += 1;
    }

    let mode_idx = freq.into_iter().max_by_key(|(_, count)| *count)
        .ok_or("Cannot compute mode")?.0;

    // Find the first occurrence of the mode value to get the proper AnyValue
    let mut mode_val = non_null.get(0).map_err(|e| format!("{}", e))?;
    for i in 0..non_null.len() {
        let val = non_null.get(i).map_err(|e| format!("{}", e))?;
        if format!("{}", val) == mode_idx {
            mode_val = val;
            break;
        }
    }

    let fill_series = Series::from_any_values(series.name().clone(), &[mode_val], false)
        .map_err(|e| format!("{}", e))?;
    let broadcast = fill_series.new_from_index(0, series.len());
    let mask = series.is_null();
    let filled = series.zip_with(&mask, &broadcast).map_err(|e| format!("{}", e))?;

    let mut new_df = df.clone();
    new_df.replace(column, filled.into()).map_err(|e| format!("{}", e))?;
    Ok(new_df)
}

pub fn one_hot_encode(df: &DataFrame, column: &str) -> Result<DataFrame, String> {
    let col = df.column(column).map_err(|e| format!("{}", e))?;
    let series = col.as_materialized_series();
    let str_series = series.cast(&DataType::String).map_err(|e| format!("{}", e))?;
    let ca = str_series.str().map_err(|e| format!("{}", e))?;

    // Get unique values
    let unique = ca.unique().map_err(|e| format!("{}", e))?;

    let mut new_df = df.clone();
    for val in unique.into_iter().flatten() {
        let col_name = format!("{}_{}", column, val);
        let mask = ca.equal(val);
        let encoded = mask.into_series().cast(&DataType::UInt8).map_err(|e| format!("{}", e))?;
        let renamed = encoded.with_name(col_name.into());
        new_df.with_column(renamed.into()).map_err(|e| format!("{}", e))?;
    }

    // Drop original column
    new_df = new_df.drop(column).map_err(|e| format!("{}", e))?;
    Ok(new_df)
}

pub fn rename_col(df: &DataFrame, old_name: &str, new_name: &str) -> Result<DataFrame, String> {
    let mut new_df = df.clone();
    new_df.rename(old_name, new_name.into()).map_err(|e| format!("Rename failed: {}", e))?;
    Ok(new_df)
}

pub fn type_recommendations(df: &DataFrame) -> Vec<crate::TypeRecommendation> {
    let mut recs = Vec::new();
    for col in df.columns() {
        let series = col.as_materialized_series();
        let dtype = series.dtype();
        let name = series.name().to_string();
        let non_null = series.len() - series.null_count();

        match dtype {
            DataType::String => {
                if non_null > 0 {
                    if let Ok(str_col) = series.str() {
                        let numeric_count = str_col
                            .into_iter()
                            .filter(|v| v.is_some_and(|s| s.trim().parse::<f64>().is_ok()))
                            .count();
                        let ratio = numeric_count as f64 / non_null as f64;
                        if ratio > 0.8 {
                            recs.push(crate::TypeRecommendation {
                                column: name,
                                current_type: "str".to_string(),
                                recommended_type: "f64".to_string(),
                                reason: format!(
                                    "{}% of values are numeric",
                                    (ratio * 100.0).round()
                                ),
                            });
                        }
                    }
                }
            }
            DataType::Float64 | DataType::Int64 | DataType::Int32 | DataType::Float32 => {
                if let Ok(unique) = series.n_unique() {
                    if unique == 2 {
                        let min = series.min::<f64>().ok().flatten();
                        let max = series.max::<f64>().ok().flatten();
                        if min == Some(0.0) && max == Some(1.0) {
                            recs.push(crate::TypeRecommendation {
                                column: name,
                                current_type: format!("{}", dtype),
                                recommended_type: "bool".to_string(),
                                reason: "Column contains only 0 and 1".to_string(),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    recs
}

pub fn correlation_matrix(df: &DataFrame) -> Result<CorrelationMatrix, String> {
    // Collect numeric column names
    let numeric_cols: Vec<String> = df
        .columns()
        .iter()
        .filter(|c| c.as_materialized_series().dtype().is_numeric())
        .map(|c| c.as_materialized_series().name().to_string())
        .collect();

    if numeric_cols.is_empty() {
        return Err("No numeric columns found".to_string());
    }

    let n = numeric_cols.len();
    let mut values = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        values[i][i] = 1.0;
        for j in (i + 1)..n {
            let corr = pearson_corr(df, &numeric_cols[i], &numeric_cols[j]);
            values[i][j] = corr;
            values[j][i] = corr;
        }
    }

    Ok(CorrelationMatrix {
        columns: numeric_cols,
        values,
    })
}

fn pearson_corr(df: &DataFrame, col_a: &str, col_b: &str) -> f64 {
    let a = match df.column(col_a) {
        Ok(c) => c.as_materialized_series().cast(&DataType::Float64).ok(),
        Err(_) => None,
    };
    let b = match df.column(col_b) {
        Ok(c) => c.as_materialized_series().cast(&DataType::Float64).ok(),
        Err(_) => None,
    };

    let (a, b) = match (a, b) {
        (Some(a), Some(b)) => (a, b),
        _ => return 0.0,
    };

    let ca = match a.f64() {
        Ok(c) => c,
        Err(_) => return 0.0,
    };
    let cb = match b.f64() {
        Ok(c) => c,
        Err(_) => return 0.0,
    };

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;
    let mut count = 0u64;

    for (av, bv) in ca.into_iter().zip(cb.into_iter()) {
        if let (Some(x), Some(y)) = (av, bv) {
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            count += 1;
        }
    }

    if count < 2 {
        return 0.0;
    }

    let n = count as f64;
    let numerator = n * sum_xy - sum_x * sum_y;
    let denom_a = (n * sum_x2 - sum_x * sum_x).sqrt();
    let denom_b = (n * sum_y2 - sum_y * sum_y).sqrt();

    if denom_a < f64::EPSILON || denom_b < f64::EPSILON {
        return 0.0;
    }

    let r = numerator / (denom_a * denom_b);
    r.clamp(-1.0, 1.0)
}

pub fn scatter_data(df: &DataFrame, x_col: &str, y_col: &str) -> Result<ScatterData, String> {
    let x_series = df
        .column(x_col)
        .map_err(|e| format!("Column '{}' not found: {}", x_col, e))?
        .as_materialized_series()
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot cast '{}' to f64: {}", x_col, e))?;
    let y_series = df
        .column(y_col)
        .map_err(|e| format!("Column '{}' not found: {}", y_col, e))?
        .as_materialized_series()
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot cast '{}' to f64: {}", y_col, e))?;

    let x_ca = x_series.f64().map_err(|e| format!("{}", e))?;
    let y_ca = y_series.f64().map_err(|e| format!("{}", e))?;

    // Collect non-null pairs
    let mut points: Vec<(f64, f64)> = Vec::new();
    for (xv, yv) in x_ca.into_iter().zip(y_ca.into_iter()) {
        if let (Some(x), Some(y)) = (xv, yv) {
            points.push((x, y));
        }
    }

    // Sample if more than 2000 rows
    if points.len() > 2000 {
        let step = points.len() as f64 / 2000.0;
        let sampled: Vec<(f64, f64)> = (0..2000)
            .map(|i| points[(i as f64 * step) as usize])
            .collect();
        points = sampled;
    }

    let x: Vec<f64> = points.iter().map(|(x, _)| *x).collect();
    let y: Vec<f64> = points.iter().map(|(_, y)| *y).collect();

    Ok(ScatterData {
        x,
        y,
        x_label: x_col.to_string(),
        y_label: y_col.to_string(),
    })
}

pub fn box_plot_data(
    df: &DataFrame,
    numeric_col: &str,
    group_col: &str,
) -> Result<Vec<BoxPlotGroup>, String> {
    let num_series = df
        .column(numeric_col)
        .map_err(|e| format!("Column '{}' not found: {}", numeric_col, e))?
        .as_materialized_series()
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot cast '{}' to f64: {}", numeric_col, e))?;
    let grp_series = df
        .column(group_col)
        .map_err(|e| format!("Column '{}' not found: {}", group_col, e))?
        .as_materialized_series()
        .cast(&DataType::String)
        .map_err(|e| format!("Cannot cast '{}' to string: {}", group_col, e))?;

    let num_ca = num_series.f64().map_err(|e| format!("{}", e))?;
    let grp_ca = grp_series.str().map_err(|e| format!("{}", e))?;

    // Group values
    let mut groups: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();

    for (nv, gv) in num_ca.into_iter().zip(grp_ca.into_iter()) {
        if let (Some(val), Some(group)) = (nv, gv) {
            groups
                .entry(group.to_string())
                .or_insert_with(Vec::new)
                .push(val);
        }
    }

    // Sort groups by size descending and limit to top 20
    let mut group_list: Vec<(String, Vec<f64>)> = groups.into_iter().collect();
    group_list.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    group_list.truncate(20);

    let mut results: Vec<BoxPlotGroup> = Vec::new();

    for (group_name, mut vals) in group_list {
        if vals.is_empty() {
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = vals.len();
        let min = vals[0];
        let max = vals[n - 1];
        let median = percentile(&vals, 50.0);
        let q1 = percentile(&vals, 25.0);
        let q3 = percentile(&vals, 75.0);
        let iqr = q3 - q1;
        let lower_fence = q1 - 1.5 * iqr;
        let upper_fence = q3 + 1.5 * iqr;

        let mut outliers: Vec<f64> = vals
            .iter()
            .filter(|&&v| v < lower_fence || v > upper_fence)
            .copied()
            .collect();
        outliers.truncate(50);

        results.push(BoxPlotGroup {
            group: group_name,
            min,
            q1,
            median,
            q3,
            max,
            outliers,
        });
    }

    Ok(results)
}

fn percentile(sorted_vals: &[f64], pct: f64) -> f64 {
    if sorted_vals.len() == 1 {
        return sorted_vals[0];
    }
    let rank = (pct / 100.0) * (sorted_vals.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted_vals[lower]
    } else {
        let frac = rank - lower as f64;
        sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac
    }
}

pub fn math_columns(df: &DataFrame, col_a: &str, col_b: &str, op: &str, new_name: &str) -> Result<DataFrame, String> {
    let a = df.column(col_a).map_err(|e| format!("{}", e))?
        .as_materialized_series()
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot cast '{}' to f64: {}", col_a, e))?;
    let b = df.column(col_b).map_err(|e| format!("{}", e))?
        .as_materialized_series()
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot cast '{}' to f64: {}", col_b, e))?;

    let ca = a.f64().map_err(|e| format!("{}", e))?;
    let cb = b.f64().map_err(|e| format!("{}", e))?;

    let result: Float64Chunked = match op {
        "add" => ca + cb,
        "subtract" => ca - cb,
        "multiply" => ca * cb,
        "divide" => {
            ca.into_iter()
                .zip(cb.into_iter())
                .map(|(a_val, b_val)| match (a_val, b_val) {
                    (Some(a), Some(b)) if b != 0.0 => Some(a / b),
                    _ => None,
                })
                .collect::<Float64Chunked>()
        }
        _ => return Err(format!("Unknown operation: {}. Use add, subtract, multiply, or divide", op)),
    };

    let new_series = result.with_name(new_name.into()).into_series();
    let mut new_df = df.clone();
    new_df.with_column(new_series.into()).map_err(|e| format!("Failed to add column: {}", e))?;
    Ok(new_df)
}

pub fn transform_column(df: &DataFrame, column: &str, transform: &str, new_name: &str) -> Result<DataFrame, String> {
    let col = df.column(column).map_err(|e| format!("{}", e))?;
    let casted = col.as_materialized_series()
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot cast '{}' to f64: {}", column, e))?;
    let ca = casted.f64().map_err(|e| format!("{}", e))?;

    let result: Float64Chunked = match transform {
        "log" => {
            ca.into_iter()
                .map(|v| v.and_then(|x| if x > 0.0 { Some(x.ln()) } else { None }))
                .collect()
        }
        "log10" => {
            ca.into_iter()
                .map(|v| v.and_then(|x| if x > 0.0 { Some(x.log10()) } else { None }))
                .collect()
        }
        "sqrt" => {
            ca.into_iter()
                .map(|v| v.and_then(|x| if x >= 0.0 { Some(x.sqrt()) } else { None }))
                .collect()
        }
        "square" => {
            ca.into_iter()
                .map(|v| v.map(|x| x * x))
                .collect()
        }
        "abs" => {
            ca.into_iter()
                .map(|v| v.map(|x| x.abs()))
                .collect()
        }
        "standardize" => {
            let mean = ca.mean().ok_or("Cannot compute mean for standardization")?;
            let std = ca.std(1).ok_or("Cannot compute std for standardization")?;
            if std < f64::EPSILON {
                return Err("Standard deviation is zero; cannot standardize".to_string());
            }
            ca.into_iter()
                .map(|v| v.map(|x| (x - mean) / std))
                .collect()
        }
        "normalize" => {
            let min = ca.min().ok_or("Cannot compute min for normalization")?;
            let max = ca.max().ok_or("Cannot compute max for normalization")?;
            let range = max - min;
            if range < f64::EPSILON {
                return Err("Min equals max; cannot normalize".to_string());
            }
            ca.into_iter()
                .map(|v| v.map(|x| (x - min) / range))
                .collect()
        }
        _ => return Err(format!("Unknown transform: {}. Use log, log10, sqrt, square, abs, standardize, or normalize", transform)),
    };

    let new_series = result.with_name(new_name.into()).into_series();
    let mut new_df = df.clone();
    new_df.with_column(new_series.into()).map_err(|e| format!("Failed to add column: {}", e))?;
    Ok(new_df)
}

pub fn bin_column(df: &DataFrame, column: &str, n_bins: usize, new_name: &str) -> Result<DataFrame, String> {
    if n_bins == 0 {
        return Err("Number of bins must be greater than zero".to_string());
    }

    let col = df.column(column).map_err(|e| format!("{}", e))?;
    let casted = col.as_materialized_series()
        .cast(&DataType::Float64)
        .map_err(|e| format!("Cannot cast '{}' to f64: {}", column, e))?;
    let ca = casted.f64().map_err(|e| format!("{}", e))?;

    let min_val = ca.min().ok_or("Cannot compute min for binning")?;
    let max_val = ca.max().ok_or("Cannot compute max for binning")?;

    let bin_width = if (max_val - min_val).abs() < f64::EPSILON {
        1.0
    } else {
        (max_val - min_val) / n_bins as f64
    };

    // Build bin edges and labels
    let mut labels: Vec<String> = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let low = min_val + i as f64 * bin_width;
        let high = low + bin_width;
        labels.push(format!("{:.1}-{:.1}", low, high));
    }

    let binned: StringChunked = ca.into_iter()
        .map(|v| {
            v.map(|x| {
                let mut bin = ((x - min_val) / bin_width) as usize;
                if bin >= n_bins {
                    bin = n_bins - 1;
                }
                labels[bin].clone()
            })
        })
        .collect::<StringChunked>();

    let new_series = binned.with_name(new_name.into()).into_series();
    let mut new_df = df.clone();
    new_df.with_column(new_series.into()).map_err(|e| format!("Failed to add column: {}", e))?;
    Ok(new_df)
}

fn anyvalue_to_json(val: &AnyValue) -> serde_json::Value {
    match val {
        AnyValue::Null => serde_json::Value::Null,
        AnyValue::Boolean(b) => serde_json::Value::Bool(*b),
        AnyValue::Int8(v) => serde_json::json!(*v),
        AnyValue::Int16(v) => serde_json::json!(*v),
        AnyValue::Int32(v) => serde_json::json!(*v),
        AnyValue::Int64(v) => serde_json::json!(*v),
        AnyValue::UInt8(v) => serde_json::json!(*v),
        AnyValue::UInt16(v) => serde_json::json!(*v),
        AnyValue::UInt32(v) => serde_json::json!(*v),
        AnyValue::UInt64(v) => serde_json::json!(*v),
        AnyValue::Float32(v) => serde_json::json!(*v),
        AnyValue::Float64(v) => serde_json::json!(*v),
        AnyValue::String(s) => serde_json::Value::String(s.to_string()),
        _ => serde_json::Value::String(format!("{}", val)),
    }
}
