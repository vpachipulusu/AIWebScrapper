"""
BeautifulSoup-based scraping service that inherits from BaseScrapingService.
"""

import random
import time
import re
from typing import Any, Dict, List, Optional
import requests
from lxml import html
from requests.exceptions import RequestException
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import json
import logging

from .base_scraping_service import BaseScrapingService

logger = logging.getLogger(__name__)

# Rotating user agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Linux; Android 13; SM-S901B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]

# Browser-like headers template
HEADERS_TEMPLATE = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


class BeautifulSoupScrapingService(BaseScrapingService):
    """BeautifulSoup-based implementation of web scraping."""

    def __init__(self):
        super().__init__()
        self.session = requests.Session()

    def generate_headers(self) -> dict:
        """Generate browser-like headers with random user agent."""
        headers = HEADERS_TEMPLATE.copy()
        headers["User-Agent"] = random.choice(USER_AGENTS)
        return headers

    def detect_anti_bot(self, response: requests.Response) -> bool:
        """Detect potential anti-bot protection."""
        if response.status_code in (403, 429, 503):
            return True

        patterns = [
            r"captcha",
            r"robot",
            r"access denied",
            r"blocked",
            r"security check",
            r"cloudflare",
        ]

        content_lower = response.text.lower()
        return any(re.search(pattern, content_lower) for pattern in patterns)

    def check_known_bot_protected_domains(self, url: str) -> bool:
        """Check if URL belongs to known bot-protected domains."""
        protected_domains = [
            "amazon.com",
            "amazon.co.uk",
            "amazon.de",
            "amazon.fr",
            "tesco.com",
            "sainsburys.co.uk",
            "asda.com",
            "booking.com",
            "expedia.com",
            "linkedin.com",
            "facebook.com",
            "twitter.com",
            "indeed.com",
            "glassdoor.com",
        ]

        return any(domain in url.lower() for domain in protected_domains)

    def fetch_page_content(self, url: str, **kwargs) -> str:
        """Fetch HTML content using requests and BeautifulSoup."""
        headers = self.generate_headers()

        try:
            response = self.session.get(url, headers=headers, timeout=(5, 15))

            logger.info(f"Status: {response.status_code}")
            logger.debug(f"First 500 chars of response: {response.text[:500]}")

            if self.detect_anti_bot(response):
                logger.warning("⚠️ Possible bot detection")
                if self.check_known_bot_protected_domains(url):
                    raise Exception(
                        f"Bot protection detected on {url}. "
                        "Consider using Playwright for this domain."
                    )

            return response.text

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise Exception(f"Failed to fetch content: {e}")

    def extract_raw_data(self, content: str, url: str, **kwargs) -> Dict[str, Any]:
        """Extract raw data from HTML content using BeautifulSoup."""
        try:
            soup = BeautifulSoup(content, "html.parser")

            # Use base class methods for structured data extraction
            structured_data = self.extract_structured_data(soup, url)

            return {
                "url": url,
                "structured_data": structured_data,
                "raw_html": content[:1000],  # First 1000 chars for debugging
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error extracting data from {url}: {e}")
            return {"url": url, "error": str(e), "success": False}


# Create global instance
beautifulsoup_service = BeautifulSoupScrapingService()


# Legacy functions for backward compatibility
def run_scraper_with_pandas(url: str) -> Dict[str, Any]:
    """Enhanced scraper with comprehensive pandas data processing."""
    return beautifulsoup_service.scrape_and_process(url)


def run_scraper(url: str) -> Dict[str, Any]:
    """Basic scraper function."""
    try:
        content = beautifulsoup_service.fetch_page_content(url)
        return beautifulsoup_service.extract_raw_data(content, url)
    except Exception as e:
        return {"url": url, "error": str(e), "success": False}


# Keep original functions for any existing dependencies
def generate_headers() -> dict:
    """Generate browser-like headers."""
    return beautifulsoup_service.generate_headers()


def detect_anti_bot(response: requests.Response) -> bool:
    """Detect anti-bot protection."""
    return beautifulsoup_service.detect_anti_bot(response)


def parse_page(content: str, url: str) -> Dict[str, Any]:
    """Legacy function for parsing page content."""
    return beautifulsoup_service.extract_raw_data(content, url)
