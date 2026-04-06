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
    # Map: display_name → actual_import_name
    libs = {
        "fastapi": "fastapi",
        "yfinance": "yfinance",
        "pypfopt": "pypfopt",
        "quantstats": "quantstats",
        "pyxirr": "pyxirr",
        "ta": "ta",
        "pdfplumber": "pdfplumber",
        "tabula": "tabula",
        "openpyxl": "openpyxl",
        "weasyprint": "weasyprint",
        "textblob": "textblob",
        "holidays": "holidays",
        "camelot": "camelot",
        "fredapi": "fredapi",
        "spacy": "spacy",
        "simfin": "simfin",
        "pandas_datareader": "pandas_datareader",
        "finvizfinance": "finvizfinance",
        "riskfolio": "riskfolio",
        "empyrical": "empyrical",
        "statsmodels": "statsmodels",
        "arch": "arch",
        "vectorbt": "vectorbt",
        "QuantLib": "QuantLib",
        "FinancePy": "financepy",
        "geopy": "geopy",
        "shapely": "shapely",
        "taxcalc": "taxcalc",
        "PIL": "PIL",
        "pytesseract": "pytesseract",
        "pdf2image": "pdf2image",
    }

    results = {}
    for display_name, import_name in libs.items():
        try:
            __import__(import_name)
            results[display_name] = "installed"
        except ImportError:
            results[display_name] = "missing"

    installed = [k for k, v in results.items() if v == "installed"]
    missing = [k for k, v in results.items() if v == "missing"]

    return {
        "dependencies": results,
        "installed_count": len(installed),
        "missing_count": len(missing),
        "missing": missing,
        "total": len(libs),
    }
