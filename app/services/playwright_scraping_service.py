"""
Playwright-based scraping service that inherits from BaseScrapingService.
"""

import asyncio
import logging
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from bs4 import BeautifulSoup
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .base_scraping_service import BaseScrapingService
from .data_processor import convert_numpy_types

logger = logging.getLogger(__name__)


# Define content types that can be extracted
class ContentType(str, Enum):
    TABLES = "tables"
    LISTS = "lists"
    ARTICLES = "articles"
    KEY_POINTS = "key_points"
    EXPLANATIONS = "explanations"
    LINKS = "links"
    METADATA = "metadata"
    CUSTOM = "custom"


# Define selector types
class SelectorType(str, Enum):
    CSS = "css"
    XPATH = "xpath"


# Model for custom selectors
class CustomSelector(BaseModel):
    name: str = Field(..., description="Name for this selector (e.g., 'product_title')")
    selector: str = Field(..., description="The CSS selector or XPath expression")
    selector_type: SelectorType = Field(
        SelectorType.CSS, description="Whether this is a CSS selector or XPath"
    )
    extract_attribute: Optional[str] = Field(
        None, description="HTML attribute to extract (e.g., 'href', 'src')"
    )


# Main request model
class ScrapeRequest(BaseModel):
    url: str
    content_types: List[ContentType] = Field(
        default=[
            ContentType.TABLES,
            ContentType.LISTS,
            ContentType.ARTICLES,
            ContentType.METADATA,
        ],
        description="Types of content to extract",
    )
    custom_selectors: Optional[List[CustomSelector]] = Field(
        None, description="Custom CSS selectors or XPath expressions to extract data"
    )
    timeout: int = Field(30, description="Timeout in seconds")
    wait_after_load: int = Field(
        2, description="Seconds to wait after page load for dynamic content"
    )
    return_html: bool = Field(
        False, description="Whether to return raw HTML for analysis"
    )


class PlaywrightScrapingService(BaseScrapingService):
    """Playwright-based implementation of web scraping."""

    def __init__(self):
        super().__init__()

    def install_playwright_browsers(self):
        """Install Playwright browsers if not already installed."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.warning(f"Browser installation warning: {result.stderr}")
        except Exception as e:
            logger.warning(f"Could not install browsers: {e}")

    def fetch_page_content(self, url: str, **kwargs) -> str:
        """Fetch HTML content using Playwright."""
        # Install browsers if needed
        self.install_playwright_browsers()

        try:
            from playwright.sync_api import (
                sync_playwright,
                TimeoutError as PlaywrightTimeoutError,
            )
        except ImportError as e:
            raise Exception(
                "Playwright is not installed. Please run 'pip install playwright'"
            )

        timeout = kwargs.get("timeout", 30)
        wait_after_load = kwargs.get("wait_after_load", 2)

        try:
            with sync_playwright() as p:
                # Try different browsers
                browsers_to_try = ["chromium", "firefox"]
                browser = None
                errors = []

                for browser_type in browsers_to_try:
                    try:
                        logger.info(f"Trying to launch {browser_type}...")
                        if browser_type == "chromium":
                            browser = p.chromium.launch(headless=True)
                        else:
                            browser = p.firefox.launch(headless=True)
                        break
                    except Exception as e:
                        error_msg = str(e)
                        errors.append(f"{browser_type}: {error_msg}")
                        logger.warning(f"Failed to launch {browser_type}: {error_msg}")
                        continue

                if not browser:
                    raise Exception(
                        f"Could not launch any browser. Errors: {'; '.join(errors)}"
                    )

                try:
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                    )
                    page = context.new_page()

                    logger.info(f"Navigating to {url}...")
                    page.goto(
                        url, timeout=timeout * 1000, wait_until="domcontentloaded"
                    )

                    # Wait for dynamic content
                    if wait_after_load > 0:
                        logger.info(
                            f"Waiting {wait_after_load} seconds for dynamic content..."
                        )
                        page.wait_for_timeout(wait_after_load * 1000)

                    # Get page content
                    content = page.content()
                    logger.info("Successfully retrieved page content")

                    return content

                finally:
                    browser.close()

        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            raise Exception(f"Failed to fetch content with Playwright: {e}")

    def extract_raw_data(self, content: str, url: str, **kwargs) -> Dict[str, Any]:
        """Extract raw data from HTML content using BeautifulSoup."""
        try:
            soup = BeautifulSoup(content, "html.parser")

            # Use base class methods for structured data extraction
            structured_data = self.extract_structured_data(soup, url)

            result = {"url": url, "success": True, **structured_data}

            # Add custom selector extraction if requested
            custom_selectors = kwargs.get("custom_selectors")
            if custom_selectors:
                # This would require additional implementation for custom selectors
                pass

            # Add HTML if requested
            if kwargs.get("return_html"):
                result["html"] = content

            return result

        except Exception as e:
            logger.error(f"Error extracting data from {url}: {e}")
            return {"url": url, "error": str(e), "success": False}


# Create global instance
playwright_service = PlaywrightScrapingService()


# Main synchronous scraping function for backward compatibility
def sync_scrape(request: ScrapeRequest) -> dict:
    """Synchronous scraping function using Playwright service."""
    try:
        # Fetch content
        content = playwright_service.fetch_page_content(
            request.url,
            timeout=request.timeout,
            wait_after_load=request.wait_after_load,
        )

        # Extract data
        result = playwright_service.extract_raw_data(
            content,
            request.url,
            custom_selectors=request.custom_selectors,
            return_html=request.return_html,
        )

        # Convert NumPy types for JSON serialization
        result = convert_numpy_types(result)

        return result

    except Exception as e:
        logger.error(f"Error in sync_scrape: {str(e)}")
        raise Exception(str(e))


# Async wrapper
async def scrape_specific_page_content(request: ScrapeRequest) -> dict:
    """Async wrapper for the synchronous scraping function."""
    try:
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, sync_scrape, request)
            return result

    except Exception as e:
        error_msg = str(e) if str(e) else "Unknown error occurred during scraping"
        logger.error(f"Error in scrape_page: {error_msg}")
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail=error_msg)
