"""
模型 API - 预测结果、评估指标
读取 results/ 下的真实数据
"""
from fastapi import APIRouter, Query
import pandas as pd
import json
from pathlib import Path

router = APIRouter()

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

MODEL_CONFIGS = [
    {"name": "naive", "display": "Naive Baseline", "color": "#999999", "dir": "phase2_baselines/naive"},
    {"name": "rf", "display": "随机森林 (RF)", "color": "#5470c6", "dir": "phase2_baselines/rf"},
    {"name": "lstm", "display": "LSTM", "color": "#91cc75", "dir": "phase2_baselines/lstm"},
    {"name": "lstm_rf", "display": "LSTM-RF 混合", "color": "#fac858", "dir": "phase2_baselines/lstm_rf"},
    {"name": "cnn_lstm", "display": "CNN-LSTM", "color": "#ee6666", "dir": "phase3_cnn_lstm/cnn_lstm"},
]

_cache = {}


def load_model_data(model_name: str):
    if model_name in _cache:
        return _cache[model_name]

    config = next((m for m in MODEL_CONFIGS if m["name"] == model_name), None)
    if not config:
        return None

    base_dir = RESULTS_DIR / config["dir"]
    metrics_path = base_dir / "metrics.json"
    predictions_path = base_dir / "predictions_test.csv"

    if not metrics_path.exists() or not predictions_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    df = pd.read_csv(predictions_path, encoding="utf-8-sig")
    if "date" not in df.columns:
        df["date"] = range(len(df))

    result = {
        "config": config,
        "metrics": metrics,
        "dates": df["date"].tolist(),
        "y_true": [round(float(v), 4) if pd.notna(v) else None for v in df["y_true"].tolist()],
        "y_pred": [round(float(v), 4) if pd.notna(v) else None for v in df["y_pred"].tolist()],
    }

    _cache[model_name] = result
    return result


@router.get("/models")
async def get_models():
    models = []
    for m in MODEL_CONFIGS:
        data = load_model_data(m["name"])
        if data:
            models.append({
                "name": m["name"],
                "display": m["display"],
                "color": m["color"],
                "has_data": True,
            })
        else:
            models.append({
                "name": m["name"],
                "display": m["display"],
                "color": m["color"],
                "has_data": False,
            })
    return models


@router.get("/predictions/all")
async def get_all_predictions():
    result = {"dates": [], "y_true": [], "models": {}}
    first_data = None

    for m in MODEL_CONFIGS:
        data = load_model_data(m["name"])
        if data:
            if first_data is None:
                first_data = data
                result["dates"] = data["dates"]
                result["y_true"] = data["y_true"]

            result["models"][m["name"]] = {
                "display": m["display"],
                "color": m["color"],
                "y_pred": data["y_pred"],
            }

    return result


@router.get("/predictions/{model}")
async def get_predictions(
    model: str,
    start: str = Query(None),
    end: str = Query(None),
):
    data = load_model_data(model)
    if not data:
        return {"error": f"Model '{model}' not found or no data"}

    dates = data["dates"]
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    residual = [round(t - p, 4) if t is not None and p is not None else None
                for t, p in zip(y_true, y_pred)]

    return {
        "model": model,
        "display": data["config"]["display"],
        "dates": dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": residual,
    }


@router.get("/metrics")
async def get_metrics():
    metrics = {}
    for m in MODEL_CONFIGS:
        data = load_model_data(m["name"])
        if data:
            raw = data["metrics"]
            metrics[m["name"]] = {
                "RMSE": round(raw.get("RMSE", 0), 4),
                "MAE": round(raw.get("MAE", 0), 4),
                "MAPE": round(raw.get("MAPE", 0), 2),
                "R2": round(raw.get("R2", 0), 4),
            }
    return metrics


@router.get("/predictions/window/{window}")
async def get_window_predictions(window: int):
    """获取指定窗口大小的预测结果（用于窗口对比功能）"""
    window_dir = RESULTS_DIR / f"window_{window}"
    if not window_dir.exists():
        return {"error": f"Window {window} not found", "dates": [], "y_true": [], "models": {}}

    result = {"dates": [], "y_true": [], "models": {}}
    first_data = None

    for model_name in ["rf", "cnn_lstm"]:
        pred_file = window_dir / model_name / "predictions_test.csv"
        metrics_file = window_dir / model_name / "metrics.json"

        if pred_file.exists() and metrics_file.exists():
            df = pd.read_csv(pred_file, encoding="utf-8-sig")
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            if first_data is None:
                first_data = df
                result["dates"] = df["date"].tolist()
                result["y_true"] = [round(float(v), 4) if pd.notna(v) else None
                                   for v in df["y_true"].tolist()]

            display_names = {"cnn_lstm": "CNN-LSTM", "lstm_rf": "LSTM-RF", "rf": "随机森林"}
            colors = {"cnn_lstm": "#ee6666", "lstm_rf": "#fac858", "rf": "#5470c6"}
            display_name = display_names.get(model_name, model_name)
            color = colors.get(model_name, "#999999")

            result["models"][model_name] = {
                "display": display_name,
                "color": color,
                "y_pred": [round(float(v), 4) if pd.notna(v) else None
                          for v in df["y_pred"].tolist()],
                "metrics": {
                    "RMSE": round(metrics.get("RMSE", 0), 4),
                    "MAE": round(metrics.get("MAE", 0), 4),
                    "MAPE": round(metrics.get("MAPE", 0), 2),
                    "R2": round(metrics.get("R2", 0), 4),
                }
            }

    return result


@router.get("/windows")
async def get_windows():
    """获取所有可用的窗口大小"""
    windows = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("window_"):
            try:
                w = int(d.name.split("_")[1])
                windows.append(w)
            except (ValueError, IndexError):
                continue
    return {"windows": windows}


@router.get("/variables")
async def get_variables():
    """获取所有可用的变量组合"""
    variables = []
    var_configs = [
        {"name": "radon", "display": "水氡", "description": "单变量 - 水氡浓度"},
        {"name": "level", "display": "水位", "description": "单变量 - 动水位"},
        {"name": "radon_level", "display": "水氡+水位", "description": "双变量 - 水氡+水位"},
        {"name": "radon_level_weather", "display": "水氡+水位+气象", "description": "多变量 - 水氡+水位+降水+气温+气压"},
    ]
    for vc in var_configs:
        var_dir = RESULTS_DIR / vc["name"]
        if var_dir.exists() and any(var_dir.iterdir()):
            variables.append(vc)
    return {"variables": variables}


@router.get("/predictions/{var_name}/{window}")
async def get_var_predictions(var_name: str, window: int, model: str = "cnn_lstm"):
    """获取指定变量组合+窗口的预测结果"""
    pred_file = RESULTS_DIR / var_name / f"window_{window}" / model / "predictions_test.csv"
    metrics_file = RESULTS_DIR / var_name / f"window_{window}" / model / "metrics.json"

    if not pred_file.exists():
        return {"error": f"no predictions for var={var_name}, window={window}, model={model}",
                "dates": [], "y_true": [], "y_pred": []}

    df = pd.read_csv(pred_file, encoding="utf-8-sig")
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "var_name": var_name,
        "window": window,
        "model": model,
        "dates": df["date"].tolist(),
        "y_true": [round(float(v), 4) if pd.notna(v) else None for v in df["y_true"].tolist()],
        "y_pred": [round(float(v), 4) if pd.notna(v) else None for v in df["y_pred"].tolist()],
        "metrics": {
            "RMSE": round(metrics.get("RMSE", 0), 4),
            "MAE": round(metrics.get("MAE", 0), 4),
            "MAPE": round(metrics.get("MAPE", 0), 2),
            "R2": round(metrics.get("R2", 0), 4),
        }
    }


@router.get("/multi_var_comparison")
async def get_multi_var_comparison():
    """获取多变量对比汇总"""
    comparison_file = RESULTS_DIR / "multi_var_comparison.json"
    if not comparison_file.exists():
        return {"error": "comparison not found"}
    return json.loads(comparison_file.read_text(encoding="utf-8"))
