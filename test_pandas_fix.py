#!/usr/bin/env python3
"""Test script to check pandas integration and NumPy serialization fix."""

from app.services.scraper_service import run_scraper_with_pandas
import json


def test_pandas_integration():
    """Test the pandas-enhanced scraper."""
    print("Testing pandas integration...")

    try:
        # Test with a simple page
        url = "https://httpbin.org/html"
        result = run_scraper_with_pandas(url)

        print(f"✓ Successfully scraped and processed {url}")
        print(f"✓ Result keys: {list(result.keys())}")

        # Test JSON serialization
        json_str = json.dumps(result)
        print(f"✓ JSON serialization successful (length: {len(json_str)} chars)")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pandas_integration()
    exit(0 if success else 1)
