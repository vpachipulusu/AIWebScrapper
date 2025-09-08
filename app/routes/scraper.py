from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from app.services.playwright_service import (
    ContentType,
    CustomSelector,
    ScrapeRequest,
    scrape_specific_page_content,
)
from app.services.beautifulsoup_service import sync_scrape_with_beautifulsoup
from app.services.scraper_service import run_scraper, run_scraper_with_pandas
from app.services.selectolax_service import scrape_page as selectolax_scrape_page
from app.services.proxy_service import (
    refresh_proxies,
    get_proxy_statistics,
    get_proxy_for_request,
)
from app.services.data_processor import process_web_data
from urllib.parse import urljoin
import asyncio
from concurrent.futures import ThreadPoolExecutor

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


@router.post("/beautifulsoup-scrape")
async def scrape_with_beautifulsoup_endpoint(
    request: ScrapeRequest,
    use_proxy: bool = Query(True, description="Whether to use proxy rotation"),
):
    """
    Scrape a website using BeautifulSoup with user-selectable content extraction options.
    More reliable on Windows and for static content. Includes proxy support.
    """
    try:
        # Run in thread pool since sync_scrape_with_beautifulsoup is synchronous
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, sync_scrape_with_beautifulsoup, request, use_proxy
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/beautifulsoup-scrape")
async def scrape_with_beautifulsoup_get(
    url: str,
    content_types: str = Query(
        "tables,lists,articles,metadata",
        description="Comma-separated list of content types to extract",
    ),
    custom_selectors: str = Query(None, description="JSON string of custom selectors"),
    timeout: int = Query(30, description="Timeout in seconds"),
    wait_after_load: int = Query(2, description="Seconds to wait after page load"),
    return_html: bool = Query(False, description="Whether to return HTML for analysis"),
    use_proxy: bool = Query(True, description="Whether to use proxy rotation"),
):
    """
    Scrape a website using BeautifulSoup with user-selectable content extraction options (GET version).
    More reliable on Windows and for static content.
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

        # Run in thread pool since sync_scrape_with_beautifulsoup is synchronous
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, sync_scrape_with_beautifulsoup, request, use_proxy
            )
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


# Proxy management endpoints
@router.get("/proxy/stats")
async def get_proxy_stats():
    """Get current proxy pool statistics."""
    try:
        stats = get_proxy_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proxy/refresh")
async def refresh_proxy_pool(
    force: bool = Query(False, description="Force refresh even if recently refreshed"),
):
    """Refresh the proxy pool with new working proxies."""
    try:
        # Run refresh in thread pool since it's synchronous
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, refresh_proxies, force)
        return {"message": "Proxy pool refresh initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/proxy/current")
async def get_current_proxy():
    """Get a proxy from the current pool for testing."""
    try:
        proxy = get_proxy_for_request()
        if proxy:
            return {"proxy": proxy}
        else:
            return {"proxy": None, "message": "No working proxies available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Pandas-enhanced scraping endpoints
@router.post("/beautifulsoup-pandas")
async def scrape_with_beautifulsoup_pandas(
    request: ScrapeRequest,
    use_proxy: bool = Query(True, description="Whether to use proxy rotation"),
):
    """
    Enhanced BeautifulSoup scraping with comprehensive pandas data processing,
    cleaning, and analysis.
    """
    try:
        # Get basic scraped data using BeautifulSoup with pandas
        from app.services.scraper_service import run_scraper_with_pandas

        enhanced_data = run_scraper_with_pandas(str(request.url))

        return {
            "status": "success",
            "enhanced_with_pandas": True,
            "data": enhanced_data,
            "features": [
                "Table extraction and cleaning",
                "List analysis and patterns",
                "Text readability analysis",
                "Data quality assessment",
                "Export-ready formats",
                "Statistical insights",
            ],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Enhanced scraping failed: {str(e)}"
        )


@router.post("/playwright-pandas")
async def scrape_with_playwright_pandas(
    url: HttpUrl,
    content_types: str = Query(
        "tables,lists,articles", description="Comma-separated content types to extract"
    ),
    timeout: int = Query(30, description="Timeout in seconds"),
    wait_after_load: int = Query(2, description="Seconds to wait after page load"),
    use_proxy: bool = Query(True, description="Whether to use proxy rotation"),
):
    """
    Enhanced Playwright scraping with comprehensive pandas data processing.
    Extracts structured data and provides advanced analysis.
    """
    try:
        # Parse content types
        types_list = [
            ContentType(t.strip()) for t in content_types.split(",") if t.strip()
        ]

        # Create request object
        request = ScrapeRequest(
            url=url,
            content_types=types_list,
            timeout=timeout,
            wait_after_load=wait_after_load,
        )

        # Get data using Playwright
        playwright_data = await scrape_specific_page_content(request)

        # Process with pandas for enhanced analysis
        enhanced_data = process_web_data(playwright_data, str(url))

        return {
            "status": "success",
            "enhanced_with_pandas": True,
            "extraction_method": "playwright",
            "data": enhanced_data,
            "features": [
                "Advanced table processing with statistics",
                "List pattern detection and classification",
                "Content structure analysis",
                "Data quality metrics",
                "Multiple export formats",
                "Comprehensive insights",
            ],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Enhanced Playwright scraping failed: {str(e)}"
        )


@router.post("/comprehensive-analysis")
async def comprehensive_data_analysis(
    url: HttpUrl,
    include_beautifulsoup: bool = Query(
        True, description="Include BeautifulSoup extraction"
    ),
    include_playwright: bool = Query(True, description="Include Playwright extraction"),
    use_proxy: bool = Query(True, description="Whether to use proxy rotation"),
):
    """
    Comprehensive web data analysis using both BeautifulSoup and Playwright
    with advanced pandas processing for maximum data extraction and insights.
    """
    try:
        results = {
            "url": str(url),
            "analysis_timestamp": None,
            "extraction_methods": [],
            "combined_insights": {},
            "data_sources": {},
        }

        # BeautifulSoup extraction
        if include_beautifulsoup:
            try:
                bs_data = await sync_scrape_with_beautifulsoup(str(url), use_proxy)
                bs_enhanced = process_web_data(bs_data, str(url))
                results["data_sources"]["beautifulsoup"] = bs_enhanced
                results["extraction_methods"].append("beautifulsoup")
            except Exception as e:
                results["data_sources"]["beautifulsoup"] = {"error": str(e)}

        # Playwright extraction
        if include_playwright:
            try:
                request = ScrapeRequest(
                    url=url,
                    content_types=[
                        ContentType.TABLES,
                        ContentType.LISTS,
                        ContentType.ARTICLES,
                    ],
                    timeout=30,
                    wait_after_load=2,
                )
                pw_data = await scrape_specific_page_content(request)
                pw_enhanced = process_web_data(pw_data, str(url))
                results["data_sources"]["playwright"] = pw_enhanced
                results["extraction_methods"].append("playwright")
            except Exception as e:
                results["data_sources"]["playwright"] = {"error": str(e)}

        # Generate combined insights
        results["combined_insights"] = generate_combined_insights(
            results["data_sources"]
        )
        results["analysis_timestamp"] = (
            results["data_sources"]
            .get("beautifulsoup", results["data_sources"].get("playwright", {}))
            .get("processed_at")
        )

        return {
            "status": "success",
            "comprehensive_analysis": True,
            "data": results,
            "summary": {
                "total_extraction_methods": len(results["extraction_methods"]),
                "successful_extractions": len(
                    [
                        m
                        for m in results["extraction_methods"]
                        if "error" not in results["data_sources"].get(m, {})
                    ]
                ),
                "features": [
                    "Multi-method extraction comparison",
                    "Cross-validation of extracted data",
                    "Comprehensive data quality analysis",
                    "Combined statistical insights",
                    "Method-specific strengths identification",
                ],
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Comprehensive analysis failed: {str(e)}"
        )


def generate_combined_insights(data_sources):
    """Generate insights by comparing data from multiple extraction methods."""
    insights = {"data_consistency": {}, "method_comparison": {}, "recommendations": []}

    # Compare table extraction
    bs_tables = len(
        data_sources.get("beautifulsoup", {}).get("enhanced_data", {}).get("tables", [])
    )
    pw_tables = len(
        data_sources.get("playwright", {}).get("enhanced_data", {}).get("tables", [])
    )

    insights["method_comparison"]["tables_found"] = {
        "beautifulsoup": bs_tables,
        "playwright": pw_tables,
        "consistency": "high" if abs(bs_tables - pw_tables) <= 1 else "low",
    }

    # Compare list extraction
    bs_lists = len(
        data_sources.get("beautifulsoup", {}).get("enhanced_data", {}).get("lists", [])
    )
    pw_lists = len(
        data_sources.get("playwright", {}).get("enhanced_data", {}).get("lists", [])
    )

    insights["method_comparison"]["lists_found"] = {
        "beautifulsoup": bs_lists,
        "playwright": pw_lists,
        "consistency": "high" if abs(bs_lists - pw_lists) <= 2 else "low",
    }

    # Generate recommendations
    if pw_tables > bs_tables:
        insights["recommendations"].append(
            "Playwright found more tables - prefer for table extraction"
        )
    elif bs_tables > pw_tables:
        insights["recommendations"].append(
            "BeautifulSoup found more tables - prefer for table extraction"
        )

    if pw_lists > bs_lists:
        insights["recommendations"].append(
            "Playwright found more lists - prefer for list extraction"
        )
    elif bs_lists > pw_lists:
        insights["recommendations"].append(
            "BeautifulSoup found more lists - prefer for list extraction"
        )

    return insights
