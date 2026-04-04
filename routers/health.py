from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Monreale OS Python Intelligence API",
        "version": "1.0.0",
    }


@router.get("/dependencies")
async def check_dependencies():
    results = {}
    libs = [
        "fastapi", "yfinance", "pypfopt", "quantstats",
        "pyxirr", "ta", "pdfplumber", "openpyxl",
        "textblob", "holidays",
    ]
    for lib in libs:
        try:
            __import__(lib.replace("-", "_"))
            results[lib] = "installed"
        except ImportError:
            results[lib] = "missing"
    return {"dependencies": results}
