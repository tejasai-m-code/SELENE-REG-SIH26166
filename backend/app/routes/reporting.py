from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os

router = APIRouter(prefix="/api/report", tags=["Reporting"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"
JSON_PATH = REPORTS_DIR / "phase13_validation_report.json"
PDF_PATH = REPORTS_DIR / "SELENE_REG_X_Scientific_Report.pdf"

@router.get("/phase13")
async def get_phase13_report():
    if not JSON_PATH.exists():
        raise HTTPException(status_code=404, detail="Report JSON not found. Generate it first.")
    import json
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/phase13/pdf")
async def get_phase13_pdf():
    if not PDF_PATH.exists():
        raise HTTPException(status_code=404, detail="Report PDF not found. Generate it first.")
    return FileResponse(
        path=PDF_PATH,
        filename="SELENE_REG_X_Scientific_Report.pdf",
        media_type="application/pdf"
    )
