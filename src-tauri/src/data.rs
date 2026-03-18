use polars::prelude::*;
use polars::io::parquet::read::ParquetReader;
use crate::{ColumnProfile, DataSummary, TablePage};

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

pub fn detect_separator_pub(sample: &str) -> u8 {
    detect_separator(sample)
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
