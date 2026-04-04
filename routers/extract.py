from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from lib.auth import verify_api_key
import tempfile
import os

router = APIRouter()


@router.post("/pdf-text", dependencies=[Depends(verify_api_key)])
async def extract_pdf_text(file: UploadFile = File(...)):
    try:
        import pdfplumber

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            pages_text = []
            with pdfplumber.open(tmp_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_text.append({"page": i + 1, "text": text, "char_count": len(text)})

            full_text = "\n\n".join([p["text"] for p in pages_text])
            return {"filename": file.filename, "pages": len(pages_text), "full_text": full_text, "page_texts": pages_text}
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/excel", dependencies=[Depends(verify_api_key)])
async def extract_excel(file: UploadFile = File(...)):
    try:
        import pandas as pd

        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            xl = pd.ExcelFile(tmp_path)
            sheets = {}
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(tmp_path, sheet_name=sheet_name)
                if not df.empty:
                    sheets[sheet_name] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "data": df.fillna("").to_dict("records"),
                    }
            return {"filename": file.filename, "sheet_names": xl.sheet_names, "sheets": sheets}
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
