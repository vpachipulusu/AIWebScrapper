from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from app.services.playwright_service import (
    ContentType,
    CustomSelector,
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
    custom_selectors: str = Query(None, description="JSON string of custom selectors"),
    timeout: int = Query(30, description="Timeout in seconds"),
    wait_after_load: int = Query(2, description="Seconds to wait after page load"),
    return_html: bool = Query(False, description="Whether to return HTML for analysis"),
):
    """
    Scrape a website with user-selectable content extraction options (GET version)
    """
    try:
        # Parse content types
        types_list = [
            ContentType(t.strip()) for t in content_types.split(",") if t.strip()
        ]

        # Parse custom selectors if provided
        custom_selectors_list = []
        if custom_selectors is not None and custom_selectors.strip():
            try:
                import json
                import ast

                # Clean the input
                custom_selectors_clean = custom_selectors.strip()

                # Try to parse as JSON first
                try:
                    custom_data = json.loads(custom_selectors_clean)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to parse as Python literal
                    try:
                        custom_data = ast.literal_eval(custom_selectors_clean)
                    except (ValueError, SyntaxError):
                        # If both fail, try to manually parse a simple format
                        custom_data = parse_simple_selector_format(
                            custom_selectors_clean
                        )

                # Convert to list of CustomSelector objects
                if isinstance(custom_data, list):
                    custom_selectors_list = [CustomSelector(**s) for s in custom_data]
                elif isinstance(custom_data, dict):
                    custom_selectors_list = [CustomSelector(**custom_data)]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="custom_selectors must be a list of objects or a single object",
                    )

            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Error parsing custom_selectors: {str(e)}"
                )

        request = ScrapeRequest(
            url=url,
            content_types=types_list,
            custom_selectors=custom_selectors_list,
            timeout=timeout,
            wait_after_load=wait_after_load,
            return_html=return_html,
        )

        result = await scrape_specific_page_content(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_simple_selector_format(selector_str: str):
    """
    Parse a simple format for selectors: name:selector:type:extract[:attribute]
    Multiple selectors can be separated by pipes (|)
    """
    selectors = []
    for selector in selector_str.split("|"):
        parts = selector.split(":")
        if len(parts) >= 4:
            selector_obj = {
                "name": parts[0],
                "selector": parts[1],
                "selector_type": parts[2],
                "extract": parts[3],
            }
            if len(parts) > 4:
                selector_obj["attribute"] = parts[4]
            selectors.append(selector_obj)

    return selectors