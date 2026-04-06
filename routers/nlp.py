from fastapi import APIRouter, Depends, HTTPException, Body
from lib.auth import verify_api_key
import re

router = APIRouter()


@router.post("/extract-entities", dependencies=[Depends(verify_api_key)])
async def extract_entities(text: str = Body(..., media_type="text/plain")):
    """Extract financial entities from text using spaCy."""
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys

            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            nlp = spacy.load("en_core_web_sm")

        doc = nlp(text[:10000])

        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
            if ent.label_
            in ["ORG", "PERSON", "GPE", "MONEY", "DATE", "PERCENT", "CARDINAL"]
        ]

        # Extract financial figures
        money_pattern = r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|K))?"
        money_matches = re.findall(money_pattern, text, re.IGNORECASE)

        # Extract percentages
        pct_pattern = r"\d+(?:\.\d+)?%"
        pct_matches = re.findall(pct_pattern, text)

        return {
            "entities": entities,
            "money_mentions": money_matches[:20],
            "percentages": pct_matches[:20],
            "entity_count": len(entities),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment", dependencies=[Depends(verify_api_key)])
async def analyze_sentiment(text: str = Body(..., media_type="text/plain")):
    """Analyze financial sentiment using TextBlob."""
    try:
        from textblob import TextBlob

        blob = TextBlob(text[:5000])
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "confidence": abs(polarity),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
