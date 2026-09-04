"""
异常检测 API - 读取 results/phase4_anomaly/ 下的完整结果
"""
from fastapi import APIRouter, Query
import pandas as pd
from pathlib import Path

router = APIRouter()

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
PHASE4_DIR = RESULTS_DIR / "phase4_anomaly"


def load_final_verdict():
    path = PHASE4_DIR / "final_verdict.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    events = []
    for _, row in df.iterrows():
        events.append({
            "id": int(row.get("event_id", 0)),
            "start": str(row.get("start", "")),
            "end": str(row.get("end", "")),
            "n_points": int(row.get("n_points", 0)),
            "peak_score": round(float(row.get("peak_score", 0)), 3),
            "direction": str(row.get("residual_direction", "")),
            "precip_mm": round(float(row.get("precip_sum_mm", 0)), 1),
            "attribution": str(row.get("attribution", "")),
            "eq_m4": int(row.get("eq_m4_count", 0)),
            "eq_m5": int(row.get("eq_m5_count", 0)),
            "eq_m5_details": str(row.get("eq_m5_details", "")),
            "correspondence": str(row.get("correspondence", "")),
        })
    return events


def load_sensitivity():
    path = PHASE4_DIR / "sensitivity.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    result = []
    for _, row in df.iterrows():
        result.append({
            "quantile": float(row.get("quantile", 0)),
            "n_anomalies": int(row.get("n_anomalies", 0)),
        })
    return result


def load_coincidence():
    path = PHASE4_DIR / "coincidence.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.to_dict(orient="records")


def load_scores_data():
    """加载异常得分数据用于可视化"""
    path = RESULTS_DIR / "phase3_cnn_lstm" / "cnn_lstm" / "predictions_test.csv"
    if not path.exists():
        return {"dates": [], "scores": [], "threshold": 0}

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "y_true" in df.columns and "y_pred" in df.columns:
        residual = abs(df["y_true"] - df["y_pred"])
        threshold = float(residual.quantile(0.98))
        dates = df["date"].tolist() if "date" in df.columns else list(range(len(df)))
        return {
            "dates": dates,
            "scores": [round(float(v), 4) for v in residual.tolist()],
            "threshold": round(threshold, 4),
        }
    return {"dates": [], "scores": [], "threshold": 0}


@router.get("/anomalies/events")
async def get_events():
    return load_final_verdict()


@router.get("/anomalies/sensitivity")
async def get_sensitivity():
    return load_sensitivity()


@router.get("/anomalies/coincidence")
async def get_coincidence():
    return load_coincidence()


@router.get("/anomalies/scores")
async def get_scores(
    quantile: float = Query(0.98),
):
    return load_scores_data()


@router.get("/anomalies/summary")
async def get_summary():
    events = load_final_verdict()
    return {
        "total_events": len(events),
        "verified_events": sum(1 for e in events if e["correspondence"] == "✓"),
        "total_m4_eq": sum(e["eq_m4"] for e in events),
        "total_m5_eq": sum(e["eq_m5"] for e in events),
        "events": events,
    }
