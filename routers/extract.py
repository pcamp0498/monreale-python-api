from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from lib.auth import verify_api_key
import tempfile
import os

router = APIRouter()


@router.post("/pdf-text", dependencies=[Depends(verify_api_key)])
async def extract_pdf_text(file: UploadFile = File(...)):
    """Extract text from PDF — digital or scanned (OCR fallback)."""
    try:
        import pdfplumber
        import io

        content = await file.read()

        # Step 1: Try pdfplumber (fast, accurate for digital PDFs)
        extracted_text = ""
        page_texts = []
        extraction_method = "digital"

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                page_texts.append({
                    "page": i + 1,
                    "text": text,
                    "char_count": len(text),
                })
                extracted_text += text + "\n\n"

        # Step 2: If text is sparse (<100 chars total), try OCR
        if len(extracted_text.strip()) < 100:
            extraction_method = "ocr"
            extracted_text = ""
            page_texts = []

            try:
                import pytesseract
                from pdf2image import convert_from_bytes

                images = convert_from_bytes(content, dpi=300)

                for i, image in enumerate(images):
                    gray = image.convert("L")
                    text = pytesseract.image_to_string(gray, config="--psm 6")
                    page_texts.append({
                        "page": i + 1,
                        "text": text,
                        "char_count": len(text),
                    })
                    extracted_text += text + "\n\n"
            except Exception as ocr_error:
                print(f"OCR failed: {ocr_error}")
                extraction_method = "failed"

        full_text = extracted_text.strip()

        return {
            "filename": file.filename,
            "pages": len(page_texts),
            "full_text": full_text,
            "page_texts": page_texts,
            "extraction_method": extraction_method,
            "char_count": len(full_text),
            "success": len(full_text) > 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf-table", dependencies=[Depends(verify_api_key)])
async def extract_pdf_tables(file: UploadFile = File(...)):
    """Extract tables from PDF using tabula."""
    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        tables = []

        # Try tabula
        try:
            import tabula

            dfs = tabula.read_pdf(
                tmp_path,
                pages="all",
                multiple_tables=True,
                pandas_options={"header": 0},
                silent=True,
            )
            for i, df in enumerate(dfs):
                if not df.empty and len(df.columns) > 1:
                    df.columns = [str(c).strip() for c in df.columns]
                    tables.append({
                        "table_index": i,
                        "rows": len(df),
                        "columns": list(df.columns),
                        "data": df.fillna("").to_dict("records"),
                        "source": "tabula",
                    })
        except Exception as tabula_error:
            print(f"Tabula failed: {tabula_error}")

        # If tabula found nothing, try pdfplumber table extraction
        if not tables:
            try:
                import pdfplumber
                import io

                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        for t_idx, table in enumerate(page.extract_tables()):
                            if table and len(table) > 1:
                                headers = [
                                    str(c).strip() if c else f"col_{j}"
                                    for j, c in enumerate(table[0])
                                ]
                                data = []
                                for row in table[1:]:
                                    data.append(
                                        {
                                            headers[j]: str(v).strip() if v else ""
                                            for j, v in enumerate(row)
                                            if j < len(headers)
                                        }
                                    )
                                if data:
                                    tables.append({
                                        "table_index": len(tables),
                                        "rows": len(data),
                                        "columns": headers,
                                        "data": data,
                                        "source": "pdfplumber",
                                    })
            except Exception as plumber_error:
                print(f"pdfplumber table extraction failed: {plumber_error}")

        return {
            "filename": file.filename,
            "tables": tables,
            "table_count": len(tables),
        }
    finally:
        os.unlink(tmp_path)


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
