from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from routers import data, portfolio, technical, extract, generate, tax, health
from routers import nlp, macro, fundamentals, factors, simulation, valuation, screener, research
from routers import extract_trades, performance, bias, options

app = FastAPI(
    title="Monreale OS Python Intelligence API",
    description="Python financial intelligence microservice for Monreale OS",
    version="1.1.0",
    docs_url="/docs" if os.getenv("ENV") != "production" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://monreale-os-web.vercel.app",
        "http://localhost:3000",
        os.getenv("ALLOWED_ORIGIN", ""),
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])
app.include_router(technical.router, prefix="/technical", tags=["Technical"])
app.include_router(extract.router, prefix="/extract", tags=["Extract"])
app.include_router(extract_trades.router, prefix="/extract", tags=["Extract"])
app.include_router(performance.router, prefix="/performance", tags=["Performance"])
app.include_router(generate.router, prefix="/generate", tags=["Generate"])
app.include_router(tax.router, prefix="/tax", tags=["Tax"])
app.include_router(nlp.router, prefix="/nlp", tags=["NLP"])
app.include_router(macro.router, prefix="/macro", tags=["Macro"])
app.include_router(fundamentals.router, prefix="/fundamentals", tags=["Fundamentals"])
app.include_router(factors.router, prefix="/factors", tags=["Factors"])
app.include_router(simulation.router, prefix="/simulation", tags=["Simulation"])
app.include_router(valuation.router, prefix="/valuation", tags=["Valuation"])
app.include_router(screener.router, prefix="/screener", tags=["Screener"])
app.include_router(research.router, prefix="/research", tags=["Research"])
app.include_router(bias.router, prefix="/bias", tags=["Bias"])
app.include_router(options.router, prefix="/options", tags=["Options"])


@app.on_event("startup")
async def download_spacy_model():
    import subprocess
    import sys

    try:
        import spacy

        spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
        except Exception as e:
            print(f"spaCy model download failed (non-fatal): {e}")


@app.get("/")
async def root():
    return {"status": "Monreale OS Python Intelligence API", "version": "1.1.0"}


@app.get("/routes")
async def list_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            routes.append({"path": route.path, "methods": sorted(route.methods or [])})
    return {"routes": sorted(routes, key=lambda r: r["path"]), "count": len(routes)}
