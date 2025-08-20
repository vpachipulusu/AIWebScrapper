from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from app.services.playwright_service import (
    ContentType,
    ScrapeRequest,
    scrape_specific_page_content,
)
from app.services.scraper_service import run_scraper
from app.services.selectolax_service import scrape_page as selectolax_scrape_page
from urllib.parse import urljoin

router = APIRouter()


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


@router.post("/selectolax-scrape")
async def selectolax_scrape_endpoint(
    url: str = Query(..., description="URL to scrape"),
):
    data = await selectolax_scrape_page(url)
    return data


@router.post("/playwright-scrape")
async def scrape_website(request: ScrapeRequest):
    """
    Scrape a website with user-selectable content extraction options
    """
    try:
        result = await scrape_specific_page_content(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/playwright-scrape")
async def scrape_website_get(
    url: str,
    content_types: str = Query(
        "tables,lists,articles,metadata",
        description="Comma-separated list of content types to extract",
    ),
    timeout: int = Query(30, description="Timeout in seconds"),
    wait_after_load: int = Query(2, description="Seconds to wait after page load"),
):
    """
    Scrape a website with user-selectable content extraction options (GET version)
    """
    try:
        # Parse content types
        types_list = [
            ContentType(t.strip()) for t in content_types.split(",") if t.strip()
        ]

        request = ScrapeRequest(
            url=url,
            content_types=types_list,
            timeout=timeout,
            wait_after_load=wait_after_load,
        )

        result = await scrape_specific_page_content(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
