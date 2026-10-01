from fastapi import APIRouter
from app.services.runtime_diagnostics import get_runtime_diagnostics, get_capability_matrix

router = APIRouter(prefix="/api/runtime", tags=["Runtime"])

@router.get("/status")
async def runtime_status():
    status = get_runtime_diagnostics()
    status["capabilities"] = get_capability_matrix()
    status["environment_limitations"] = [
        "No GDAL/ENVI support for native metadata",
        "Pipeline explicitly runs on CPU"
    ]
    import json, numpy as np
    from pathlib import Path
    metrics_path = Path(__file__).resolve().parents[3] / "phase12_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            m = json.load(f)
            cons_euc = [s['euclidean_error'] for s in m.get('subpixel', []) if s['method'] == 'Consensus' and s['status'] not in ('FAILED', 'DIVERGED')]
            taylor_fail = sum(1 for s in m.get('subpixel', []) if s['method'] == 'Taylor' and s['status'] in ('FAILED', 'DIVERGED'))
            status["synthetic_consensus"] = {
                "rmse": float(np.sqrt(np.mean(np.square(cons_euc)))) if cons_euc else None,
                "median": float(np.median(cons_euc)) if cons_euc else None,
                "max": float(np.max(cons_euc)) if cons_euc else None,
                "taylor_failures": taylor_fail,
                "total_cases": 14
            }
    return status

@router.get("/job/{job_id}")
async def job_performance(job_id: str):
    # Mocking retrieval from a central job store as lightweight extension
    return {
        "job_id": job_id,
        "status": "NOT IMPLEMENTED",
        "message": "Prototype endpoint. Real job tracking requires persistent database/storage integration."
    }
