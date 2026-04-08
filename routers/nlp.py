from fastapi import APIRouter, Depends, HTTPException, Body
from lib.auth import verify_api_key
import re

router = APIRouter()

FINANCIAL_POSITIVE = [
    "beat", "exceeded", "raised", "upgraded", "outperform", "growth", "record",
    "strong", "accelerating", "momentum", "margin expansion", "buyback",
    "dividend increase", "guidance raise", "above expectations", "surprise",
    "profitable", "recovery", "improving", "robust", "upside",
]

FINANCIAL_NEGATIVE = [
    "missed", "lowered", "downgraded", "underperform", "decline", "loss",
    "weak", "disappointing", "headwinds", "guidance cut", "impairment",
    "restructuring", "layoffs", "investigation", "lawsuit", "recall",
    "shortage", "miss", "below expectations", "warning", "default",
    "bankruptcy", "delisted", "fraud", "writedown",
]

FINANCIAL_PATTERNS = {
    "dollar_amounts": r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|K))?",
    "percentages": r"\d+(?:\.\d+)?%",
    "dates": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b",
    "cap_rates": r"(?:cap rate|capitalization rate)[:\s]+(\d+(?:\.\d+)?%)",
    "dscr": r"(?:DSCR|debt service coverage)[:\s]+(\d+(?:\.\d+)?x?)",
    "ltv": r"(?:LTV|loan.to.value)[:\s]+(\d+(?:\.\d+)?%)",
    "noi": r"(?:NOI|net operating income)[:\s]+\$?([\d,]+)",
    "sqft": r"([\d,]+)\s*(?:square feet|sq\.?\s*ft\.?|SF)",
    "units": r"([\d,]+)\s*(?:units?|apartments?|beds?)",
    "loan_amount": r"(?:loan amount|loan size)[:\s]+\$?([\d,]+(?:\.\d+)?(?:\s*(?:million|M))?)",
    "interest_rate": r"(?:interest rate|note rate)[:\s]+(\d+(?:\.\d+)?%)",
}


@router.post("/extract-entities", dependencies=[Depends(verify_api_key)])
async def extract_entities(text: str = Body(..., media_type="text/plain")):
    """Extract financial entities from text using regex patterns."""
    try:
        results = {}
        for name, pattern in FINANCIAL_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results[name] = matches[:20]

        # Also try spaCy if available (non-blocking)
        spacy_entities = []
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:10000])
            spacy_entities = [
                {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                for ent in doc.ents
                if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "DATE", "PERCENT", "CARDINAL"]
            ]
        except Exception:
            pass

        return {
            "patterns": results,
            "entities": spacy_entities,
            "money_mentions": results.get("dollar_amounts", []),
            "percentages": results.get("percentages", []),
            "entity_count": len(spacy_entities),
            "pattern_count": sum(len(v) for v in results.values()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment", dependencies=[Depends(verify_api_key)])
async def analyze_sentiment(text: str = Body(..., media_type="text/plain")):
    """Analyze financial sentiment using TextBlob + financial keyword lexicon."""
    try:
        from textblob import TextBlob

        text_lower = text.lower()

        pos_signals = [w for w in FINANCIAL_POSITIVE if w in text_lower]
        neg_signals = [w for w in FINANCIAL_NEGATIVE if w in text_lower]
        pos_count = len(pos_signals)
        neg_count = len(neg_signals)

        tb = TextBlob(text[:5000])
        tb_polarity = tb.sentiment.polarity
        subjectivity = tb.sentiment.subjectivity

        keyword_score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        combined = (tb_polarity * 0.4) + (keyword_score * 0.6)

        if combined > 0.1:
            label = "positive"
        elif combined < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": round(combined, 3),
            "polarity": round(tb_polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "positive_signals": pos_signals,
            "negative_signals": neg_signals,
            "confidence": "high" if abs(combined) > 0.3 else "medium" if abs(combined) > 0.1 else "low",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
