from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from lib.auth import verify_api_key
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class PDFRequest(BaseModel):
    html_content: str
    title: Optional[str] = "Monreale Report"


@router.post("/pdf", dependencies=[Depends(verify_api_key)])
async def generate_pdf(request: PDFRequest):
    try:
        from weasyprint import HTML, CSS

        base_css = CSS(
            string="""
            @page { size: letter; margin: 0.75in; }
            body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; color: #0A0A0A; line-height: 1.5; }
            h1 { font-size: 18pt; color: #1B2A4A; }
            h2 { font-size: 14pt; color: #1B2A4A; border-bottom: 1pt solid #C9A84C; }
            table { width: 100%; border-collapse: collapse; font-size: 9pt; }
            th { background: #1B2A4A; color: white; padding: 6pt 8pt; text-align: left; }
            td { padding: 5pt 8pt; border-bottom: 0.5pt solid #E5E7EB; }
            .gold { color: #C9A84C; }
            .confidential { color: #EF4444; font-weight: bold; font-size: 8pt; }
        """
        )

        html = HTML(string=request.html_content)
        pdf_bytes = html.write_pdf(stylesheets=[base_css])

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{request.title}.pdf"'},
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="weasyprint not available — PDF generation disabled")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
