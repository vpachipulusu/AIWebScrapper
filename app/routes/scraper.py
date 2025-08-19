from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from app.services.playwright_service import scrape_page
from app.services.scraper_service import run_scraper

router = APIRouter()


class ScrapeRequest(BaseModel):
    url: HttpUrl


@router.post("/beautiful-soup")
async def scrape_and_analyze(request: ScrapeRequest):
    try:
        scraped_data = run_scraper(str(request.url))
        if not scraped_data:
            raise HTTPException(status_code=404, detail="No data extracted from page")

        # analyzed_data = await analyze_with_scrapegraph(scraped_data)
        # return {"scraped": scraped_data, "analyzed": analyzed_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/playwright-scrape")
async def scrape_endpoint(url: str = Query(..., description="URL to scrape")):
    data = await scrape_page(url)
    return data