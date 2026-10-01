from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

sys.path.insert(0, str(BACKEND))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes.registration import router as registration_router
from app.routes.multi_registration import router as multi_registration_router
from app.routes.evaluation import router as evaluation_router
from app.routes.reporting import router as reporting_router
from app.routes.runtime import router as runtime_router

app = FastAPI(title="SELENE-REG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

(BACKEND / "outputs").mkdir(exist_ok=True)
(BACKEND / "uploads").mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(BACKEND / "outputs")), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(BACKEND / "uploads")), name="uploads")
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
app.include_router(registration_router, prefix="/api")
app.include_router(multi_registration_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
app.include_router(reporting_router)
app.include_router(runtime_router)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "SELENE-REG", "problem_statement": "SIH26166"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=True)
