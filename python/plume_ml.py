"""
Plume ML sidecar — receives JSON commands via stdin, returns JSON results via stdout.
"""
import sys
import json
import io
import base64
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score,
    confusion_matrix, roc_curve, auc,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

warnings.filterwarnings("ignore")


def respond(result):
    print(json.dumps(result), flush=True)


def respond_error(msg):
    respond({"error": str(msg)})


def load_data(params):
    path = params["path"]
    separator = params.get("separator", ",")
    df = pd.read_csv(path, sep=separator)
    return df


def prepare_features(df, features, target=None):
    """Prepare feature matrix: encode categoricals, handle missing values."""
    X = df[features].copy()

    # Track encoders for later use
    encoders = {}

    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            le = LabelEncoder()
            X[col] = X[col].fillna("__missing__")
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            col_median = X[col].median()
            if pd.isna(col_median):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(col_median)

    y = None
    target_encoder = None
    if target:
        y = df[target].copy()
        if y.dtype == "object" or y.dtype.name == "category":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        else:
            y_median = y.median()
            if pd.isna(y_median):
                y = y.fillna(0)
            else:
                y = y.fillna(y_median)

    # Replace inf/-inf with NaN, then fill remaining NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().any():
            col_median = X[col].median()
            X[col] = X[col].fillna(0 if pd.isna(col_median) else col_median)

    if y is not None and hasattr(y, 'replace'):
        y = pd.Series(y).replace([np.inf, -np.inf], np.nan)
        y_median = y.median()
        y = y.fillna(0 if pd.isna(y_median) else y_median)
        y = y.values

    return X, y, encoders, target_encoder


def get_algorithm(task, algo_name, hyperparams=None):
    hp = hyperparams or {}

    # Convert max_depth=0 to None (meaning unlimited)
    if "max_depth" in hp and hp["max_depth"] == 0:
        hp["max_depth"] = None

    if algo_name == "random_forest":
        kw = {"random_state": 42}
        kw["n_estimators"] = hp.get("n_estimators", 100)
        if "max_depth" in hp:
            kw["max_depth"] = hp["max_depth"]
        if "min_samples_split" in hp:
            kw["min_samples_split"] = hp["min_samples_split"]
        if task == "classification":
            return RandomForestClassifier(**kw)
        else:
            return RandomForestRegressor(**kw)

    if algo_name == "logistic_regression":
        return LogisticRegression(
            C=hp.get("C", 1.0),
            max_iter=hp.get("max_iter", 1000),
            random_state=42,
        )

    if algo_name == "linear_regression":
        return LinearRegression()

    if algo_name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            raise ValueError("XGBoost is not installed. Run: pip install xgboost")
        kw = {"random_state": 42, "verbosity": 0}
        kw["n_estimators"] = hp.get("n_estimators", 100)
        kw["learning_rate"] = hp.get("learning_rate", 0.1)
        if "max_depth" in hp:
            kw["max_depth"] = hp["max_depth"] if hp["max_depth"] is not None else 6
        if task == "classification":
            kw["eval_metric"] = "logloss"
            return xgb.XGBClassifier(**kw)
        else:
            return xgb.XGBRegressor(**kw)

    if algo_name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ValueError("LightGBM is not installed. Run: pip install lightgbm")
        kw = {"random_state": 42, "verbosity": -1}
        kw["n_estimators"] = hp.get("n_estimators", 100)
        kw["learning_rate"] = hp.get("learning_rate", 0.1)
        if "max_depth" in hp:
            kw["max_depth"] = hp["max_depth"] if hp["max_depth"] is not None else -1
        if "num_leaves" in hp:
            kw["num_leaves"] = hp["num_leaves"]
        if task == "classification":
            return lgb.LGBMClassifier(**kw)
        else:
            return lgb.LGBMRegressor(**kw)

    raise ValueError(f"Unknown algorithm: {algo_name}")


def train_supervised(params):
    try:
        df = load_data(params)
        target = params["target"]
        features = params["features"]
        task = params["task"]  # "classification" or "regression"
        algo_name = params.get("algorithm", "random_forest")
        test_size = params.get("test_size", 0.2)
        hyperparams = params.get("hyperparams", {})
        use_cv = params.get("use_cv", False)
        cv_folds = params.get("cv_folds", 5)

        if len(df) < 2:
            respond_error("Dataset has fewer than 2 rows. Cannot train a model.")
            return

        if len(features) == 0:
            respond_error("No features selected.")
            return

        X, y, encoders, target_encoder = prepare_features(df, features, target)

        if task == "classification":
            n_classes = len(set(v for v in y if not (isinstance(v, float) and np.isnan(v))))
            if n_classes < 2:
                respond_error(f"Target column has only {n_classes} unique value(s). Classification requires at least 2.")
                return

        if len(X) < 5:
            respond_error("Too few valid rows after preprocessing. Need at least 5.")
            return

        model = get_algorithm(task, algo_name, hyperparams)

        # Cross-validation
        cv_scores = None
        if use_cv:
            scoring = "accuracy" if task == "classification" else "r2"
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            cv_scores = {
                "scores": [round(float(s), 4) for s in scores],
                "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
                "metric": scoring,
                "folds": cv_folds,
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        metrics = {}
        if task == "classification":
            metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            avg = "weighted" if len(set(y_test)) > 2 else "binary"
            metrics["precision"] = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
            metrics["recall"] = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
            metrics["f1"] = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            labels = target_encoder.classes_.tolist() if target_encoder else sorted(list(set(y_test)))
            metrics["confusion_matrix"] = {
                "matrix": cm.tolist(),
                "labels": [str(l) for l in labels],
            }
        else:
            metrics["r2"] = round(r2_score(y_test, y_pred), 4)
            metrics["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
            metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

        # Feature importance
        importance = []
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            for fname, fval in sorted(zip(features, imp), key=lambda x: -x[1]):
                importance.append({"feature": fname, "importance": round(float(fval), 4)})
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
            for fname, fval in sorted(zip(features, coef), key=lambda x: -x[1]):
                importance.append({"feature": fname, "importance": round(float(fval), 4)})

        # ROC curve data (classification only)
        roc_data = None
        if task == "classification" and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                classes = list(range(y_proba.shape[1]))
                if len(classes) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    # Downsample to max 200 points for JSON size
                    step = max(1, len(fpr) // 200)
                    roc_data = {
                        "fpr": [round(float(v), 4) for v in fpr[::step]],
                        "tpr": [round(float(v), 4) for v in tpr[::step]],
                        "auc": round(float(roc_auc), 4),
                    }
                else:
                    # Multiclass: use macro average with one-vs-rest
                    from sklearn.preprocessing import label_binarize
                    from sklearn.metrics import roc_auc_score
                    y_test_bin = label_binarize(y_test, classes=classes)
                    fpr_all, tpr_all = [], []
                    for ci in range(len(classes)):
                        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, ci], y_proba[:, ci])
                        fpr_all.extend(fpr_i)
                        tpr_all.extend(tpr_i)
                    # Sort and downsample
                    pairs = sorted(zip(fpr_all, tpr_all))
                    step = max(1, len(pairs) // 200)
                    try:
                        macro_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
                    except Exception:
                        macro_auc = None
                    roc_data = {
                        "fpr": [round(float(p[0]), 4) for p in pairs[::step]],
                        "tpr": [round(float(p[1]), 4) for p in pairs[::step]],
                        "auc": round(float(macro_auc), 4) if macro_auc is not None else None,
                    }
            except Exception:
                pass  # Some models don't support predict_proba well

        # Residual data (regression only)
        residuals = None
        if task == "regression":
            # Downsample to max 500 points
            indices = list(range(len(y_test)))
            if len(indices) > 500:
                step = len(indices) // 500
                indices = indices[::step]
            residuals = {
                "y_true": [round(float(y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]), 4) for i in indices],
                "y_pred": [round(float(y_pred[i]), 4) for i in indices],
            }

        # Predictions for export
        predictions = [round(float(v), 6) for v in y_pred]

        result = {
            "status": "success",
            "task": task,
            "algorithm": algo_name,
            "metrics": metrics,
            "feature_importance": importance,
            "features_used": features,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "predictions": predictions,
        }
        if cv_scores:
            result["cv_scores"] = cv_scores
        if roc_data:
            result["roc_curve"] = roc_data
        if residuals:
            result["residuals"] = residuals

        respond(result)

    except Exception as e:
        respond_error(e)


def train_clustering(params):
    try:
        df = load_data(params)
        features = params["features"]
        algo_name = params.get("algorithm", "kmeans")
        n_clusters = params.get("n_clusters", 3)

        X, _, encoders, _ = prepare_features(df, features)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if algo_name == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algo_name == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        elif algo_name == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            respond_error(f"Unknown algorithm: {algo_name}")
            return

        labels = model.fit_predict(X_scaled)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)

        metrics = {"n_clusters": n_found}
        if n_found > 1 and n_found < len(X_scaled):
            metrics["silhouette"] = round(silhouette_score(X_scaled, labels), 4)

        # Cluster summaries
        cluster_summaries = []
        df_result = df[features].copy()
        df_result["__cluster__"] = labels
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            cluster_df = df_result[df_result["__cluster__"] == cid]
            summary = {"cluster": int(cid), "size": len(cluster_df), "characteristics": []}
            for feat in features[:5]:
                col = cluster_df[feat]
                if col.dtype in ["object", "category"]:
                    mode = col.mode().iloc[0] if len(col.mode()) > 0 else "N/A"
                    summary["characteristics"].append(f"{feat}: mostly {mode}")
                else:
                    summary["characteristics"].append(f"{feat}: avg {col.mean():.2f}")
            cluster_summaries.append(summary)

        # Scatter plot data via PCA to 2D
        scatter_data = None
        try:
            from sklearn.decomposition import PCA
            if X_scaled.shape[1] >= 2:
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X_scaled)
                # Downsample to max 1000 points
                indices = list(range(len(coords)))
                if len(indices) > 1000:
                    step = len(indices) // 1000
                    indices = indices[::step]
                scatter_data = {
                    "x": [round(float(coords[i, 0]), 4) for i in indices],
                    "y": [round(float(coords[i, 1]), 4) for i in indices],
                    "labels": [int(labels[i]) for i in indices],
                    "x_label": "PC1",
                    "y_label": "PC2",
                    "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
                }
        except Exception:
            pass

        result = {
            "status": "success",
            "task": "clustering",
            "algorithm": algo_name,
            "metrics": metrics,
            "clusters": cluster_summaries,
        }
        if scatter_data:
            result["scatter"] = scatter_data

        respond(result)

    except Exception as e:
        respond_error(e)


def export_pickle(params):
    """Retrain model and save as pickle file."""
    try:
        import pickle
        df = load_data(params)
        target = params.get("target")
        features = params["features"]
        task = params["task"]
        algo_name = params.get("algorithm", "random_forest")
        hyperparams = params.get("hyperparams", {})
        output_path = params["output_path"]

        X, y, encoders, target_encoder = prepare_features(df, features, target)
        model = get_algorithm(task, algo_name, hyperparams)
        model.fit(X, y)

        bundle = {
            "model": model,
            "features": features,
            "target": target,
            "task": task,
            "algorithm": algo_name,
            "encoders": encoders,
            "target_encoder": target_encoder,
        }

        with open(output_path, "wb") as f:
            pickle.dump(bundle, f)

        respond({"status": "success", "path": output_path})

    except Exception as e:
        respond_error(e)


def generate_report(params):
    """Generate an HTML summary report from training results."""
    try:
        output_path = params["output_path"]
        results = params["results"]  # list of TrainResult dicts

        html_parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Plume Model Report</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;color:#333;background:#fafafa}",
            "h1{font-size:22px;font-weight:600;margin-bottom:4px}",
            "h2{font-size:16px;font-weight:600;margin-top:32px;border-bottom:1px solid #e5e5e5;padding-bottom:6px}",
            "h3{font-size:13px;font-weight:600;margin-top:20px}",
            "table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}",
            "th,td{padding:6px 12px;text-align:left;border-bottom:1px solid #eee}",
            "th{font-weight:500;color:#888;font-size:11px;text-transform:uppercase;letter-spacing:0.5px}",
            "td.num{text-align:right;font-variant-numeric:tabular-nums}",
            ".bar-wrap{background:#e5e5e5;border-radius:3px;height:6px;width:100%}",
            ".bar{background:#8b5cf6;height:6px;border-radius:3px}",
            ".meta{font-size:11px;color:#888;margin-top:24px}",
            ".best{background:#f5f3ff}",
            "</style></head><body>",
            "<h1>Plume Model Report</h1>",
            f"<p style='font-size:12px;color:#888'>Generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
        ]

        if len(results) > 1:
            # Comparison table
            html_parts.append("<h2>Model Comparison</h2><table><thead><tr><th>Algorithm</th>")
            task_type = results[0].get("task", "")
            if task_type == "classification":
                html_parts.append("<th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th>")
            elif task_type == "regression":
                html_parts.append("<th>R²</th><th>MAE</th><th>RMSE</th>")
            html_parts.append("</tr></thead><tbody>")

            for r in results:
                m = r.get("metrics", {})
                algo = r.get("algorithm", "")
                html_parts.append(f"<tr><td>{algo}</td>")
                if task_type == "classification":
                    html_parts.append(f"<td class='num'>{m.get('accuracy', 0)*100:.1f}%</td>")
                    html_parts.append(f"<td class='num'>{m.get('precision', 0)*100:.1f}%</td>")
                    html_parts.append(f"<td class='num'>{m.get('recall', 0)*100:.1f}%</td>")
                    html_parts.append(f"<td class='num'>{m.get('f1', 0)*100:.1f}%</td>")
                elif task_type == "regression":
                    html_parts.append(f"<td class='num'>{m.get('r2', 0):.4f}</td>")
                    html_parts.append(f"<td class='num'>{m.get('mae', 0):.4f}</td>")
                    html_parts.append(f"<td class='num'>{m.get('rmse', 0):.4f}</td>")
                html_parts.append("</tr>")
            html_parts.append("</tbody></table>")

        # Individual results
        for idx, r in enumerate(results):
            algo = r.get("algorithm", "unknown")
            m = r.get("metrics", {})
            html_parts.append(f"<h2>{algo}</h2>")

            # Metrics
            html_parts.append("<table><tbody>")
            for k, v in m.items():
                if k == "confusion_matrix":
                    continue
                if isinstance(v, float):
                    html_parts.append(f"<tr><td>{k}</td><td class='num'>{v:.4f}</td></tr>")
                else:
                    html_parts.append(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>")
            html_parts.append("</tbody></table>")

            # Feature importance
            fi = r.get("feature_importance", [])
            if fi:
                max_imp = fi[0]["importance"] if fi else 1
                html_parts.append("<h3>Feature Importance</h3><table><tbody>")
                for f in fi[:10]:
                    pct = (f["importance"] / max_imp * 100) if max_imp > 0 else 0
                    html_parts.append(
                        f"<tr><td>{f['feature']}</td>"
                        f"<td><div class='bar-wrap'><div class='bar' style='width:{pct}%'></div></div></td>"
                        f"<td class='num'>{f['importance']:.4f}</td></tr>"
                    )
                html_parts.append("</tbody></table>")

            if r.get("train_size"):
                html_parts.append(f"<p class='meta'>Trained on {r['train_size']} rows, tested on {r.get('test_size', '?')} rows.</p>")

        html_parts.append("</body></html>")

        html = "\n".join(html_parts)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        respond({"status": "success", "path": output_path})

    except Exception as e:
        respond_error(e)


def compute_shap(params):
    """Compute SHAP values for sample predictions."""
    try:
        df = load_data(params)
        target = params.get("target")
        features = params["features"]
        task = params["task"]
        algo_name = params.get("algorithm", "random_forest")
        hyperparams = params.get("hyperparams", {})
        n_samples = params.get("n_samples", 5)

        X, y, encoders, target_encoder = prepare_features(df, features, target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = get_algorithm(task, algo_name, hyperparams)
        model.fit(X_train, y_train)

        try:
            import shap
            # Use appropriate explainer
            if algo_name in ("random_forest", "xgboost", "lightgbm"):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_train)

            sample = X_test.iloc[:n_samples]
            shap_values = explainer.shap_values(sample)

            # For multiclass, shap_values is a list; take first class or average
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) == 2 else np.mean(shap_values, axis=0)

            predictions = model.predict(sample)

            explanations = []
            for i in range(len(sample)):
                row_shap = shap_values[i]
                contributions = []
                for fname, sv in sorted(zip(features, row_shap), key=lambda x: -abs(x[1])):
                    contributions.append({
                        "feature": fname,
                        "value": round(float(sample.iloc[i][fname]), 4),
                        "shap_value": round(float(sv), 4),
                    })
                pred_val = predictions[i]
                if target_encoder:
                    pred_label = target_encoder.inverse_transform([int(pred_val)])[0]
                else:
                    pred_label = str(round(float(pred_val), 4))
                explanations.append({
                    "prediction": pred_label,
                    "contributions": contributions[:10],
                })

            respond({"status": "success", "explanations": explanations})

        except ImportError:
            respond_error("SHAP library not installed. Run: pip install shap")

    except Exception as e:
        respond_error(e)


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            respond_error(f"Invalid JSON: {e}")
            continue

        action = cmd.get("action")
        params = cmd.get("params", {})

        if action == "train_supervised":
            train_supervised(params)
        elif action == "train_clustering":
            train_clustering(params)
        elif action == "export_model":
            export_pickle(params)
        elif action == "generate_report":
            generate_report(params)
        elif action == "compute_shap":
            compute_shap(params)
        elif action == "ping":
            respond({"status": "pong"})
        else:
            respond_error(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
