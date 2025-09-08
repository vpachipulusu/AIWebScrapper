#!/usr/bin/env python3
"""Minimal test to isolate the pandas API issue."""

from app.routes.scraper import router
from app.services.playwright_service import ScrapeRequest
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Create minimal app
app = FastAPI()
app.include_router(router, prefix="/scrape")

# Create test client
client = TestClient(app)


def test_endpoint():
    """Test the pandas endpoint directly."""
    print("Testing pandas endpoint...")

    try:
        response = client.post(
            "/scrape/beautifulsoup-pandas", json={"url": "https://httpbin.org/html"}
        )

        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        else:
            data = response.json()
            print(f"✓ Success! Keys: {list(data.keys())}")

    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_endpoint()
