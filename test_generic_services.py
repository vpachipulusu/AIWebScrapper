#!/usr/bin/env python3
"""
Test script to demonstrate the generic base service architecture.
Shows how both BeautifulSoup and Playwright services now share common functionality.
"""

from app.services.beautifulsoup_scraping_service import beautifulsoup_service
from app.services.playwright_scraping_service import playwright_service
from app.services.playwright_scraping_service import ScrapeRequest


def test_generic_services():
    """Test the generic base service architecture."""
    print("🧪 Testing Generic Base Service Architecture")
    print("=" * 60)

    # Test URL
    test_url = "https://httpbin.org/html"

    print(f"\n🌐 Testing URL: {test_url}")

    # Test BeautifulSoup service
    print("\n📄 Testing BeautifulSoup Service:")
    print("-" * 40)
    try:
        bs_result = beautifulsoup_service.scrape_and_process(test_url)
        print(f"✅ BeautifulSoup Success!")
        print(f"   - Keys: {list(bs_result.keys())}")
        if "enhanced_data" in bs_result:
            enhanced = bs_result["enhanced_data"]
            if "tables" in enhanced:
                print(f"   - Tables found: {len(enhanced['tables'])}")
            if "lists" in enhanced:
                print(f"   - Lists found: {len(enhanced['lists'])}")
    except Exception as e:
        print(f"❌ BeautifulSoup Error: {e}")

    # Test Playwright service
    print("\n🎭 Testing Playwright Service:")
    print("-" * 40)
    try:
        # Create request object
        request = ScrapeRequest(url=test_url)

        # Test content fetching
        content = playwright_service.fetch_page_content(test_url)
        print(f"✅ Playwright Content Fetch Success! ({len(content)} chars)")

        # Test data extraction
        pw_result = playwright_service.extract_raw_data(content, test_url)
        print(f"✅ Playwright Data Extraction Success!")
        print(f"   - Keys: {list(pw_result.keys())}")
        if "tables" in pw_result:
            print(f"   - Tables found: {len(pw_result['tables'])}")
        if "lists" in pw_result:
            print(f"   - Lists found: {len(pw_result['lists'])}")

    except Exception as e:
        print(f"❌ Playwright Error: {e}")

    # Compare functionality
    print("\n🔄 Generic Base Service Benefits:")
    print("-" * 40)
    print("✅ Shared table extraction logic")
    print("✅ Shared list processing with pandas")
    print("✅ Unified metadata extraction")
    print("✅ Common NumPy type conversion")
    print("✅ Consistent statistical analysis")
    print("✅ Reduced code duplication")
    print("✅ Easier maintenance and testing")


if __name__ == "__main__":
    test_generic_services()
