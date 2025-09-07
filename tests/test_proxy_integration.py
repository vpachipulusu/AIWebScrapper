#!/usr/bin/env python3
"""
Test proxy integration directly with the scraping service
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.beautifulsoup_service import sync_scrape_with_beautifulsoup
from app.services.playwright_service import ScrapeRequest, ContentType
from app.services.proxy_service import get_proxy_statistics, refresh_proxies
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_proxy_integration():
    """Test proxy integration with the scraping service."""
    print("Testing proxy integration...")

    # Test 1: Proxy statistics
    print("\n1. Getting proxy statistics...")
    stats = get_proxy_statistics()
    print(f"Initial proxy stats: {stats}")

    # Test 2: Refresh proxies
    print("\n2. Refreshing proxies...")
    refresh_proxies(force=True)
    stats = get_proxy_statistics()
    print(f"Stats after refresh: {stats}")

    # Test 3: Scrape with proxy enabled
    print("\n3. Testing scraping with proxy enabled...")
    request = ScrapeRequest(
        url="https://httpbin.org/html",
        content_types=[ContentType.METADATA, ContentType.ARTICLES],
        timeout=30,
    )

    try:
        result = sync_scrape_with_beautifulsoup(request, use_proxy=True)
        print("Scraping with proxy successful!")
        print(f"Metadata: {result.get('metadata', {})}")
        if "article" in result:
            print(
                f"Article sections found: {len(result['article'].get('sections', []))}"
            )
    except Exception as e:
        print(f"Scraping with proxy failed: {e}")

    # Test 4: Scrape without proxy
    print("\n4. Testing scraping without proxy...")
    try:
        result = sync_scrape_with_beautifulsoup(request, use_proxy=False)
        print("Scraping without proxy successful!")
        print(f"Metadata: {result.get('metadata', {})}")
    except Exception as e:
        print(f"Scraping without proxy failed: {e}")

    # Final stats
    final_stats = get_proxy_statistics()
    print(f"\nFinal proxy stats: {final_stats}")


if __name__ == "__main__":
    test_proxy_integration()
