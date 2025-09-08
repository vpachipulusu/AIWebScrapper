#!/usr/bin/env python3
"""Test script to check the Playwright NumPy fix."""

from app.routes.scraper import router
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Create minimal app
app = FastAPI()
app.include_router(router, prefix="/scrape")

# Create test client
client = TestClient(app)


def test_playwright_endpoint():
    """Test the Playwright endpoint with NumPy conversion."""
    print("Testing Playwright endpoint...")

    try:
        # Test with a simpler URL first
        response = client.get(
            "/scrape/playwright-scrape",
            params={
                "url": "https://httpbin.org/html",
                "content_types": "tables,lists,articles,metadata",
                "timeout": 30,
                "wait_after_load": 2,
                "return_html": False,
            },
        )

        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        else:
            data = response.json()
            print(f"✓ Success! Keys: {list(data.keys())}")

            # Check if there are tables/lists that might contain NumPy types
            if "tables" in data:
                print(f"  - Found {len(data['tables'])} tables")
            if "lists" in data:
                print(f"  - Found {len(data['lists'])} lists")

    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_playwright_endpoint()
