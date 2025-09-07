#!/usr/bin/env python3
"""
Test the proxy-enabled scraping endpoints
"""

import requests
import json


def test_proxy_scraping():
    """Test the scraping endpoints with proxy support."""
    base_url = "http://127.0.0.1:8001"

    # Test 1: Proxy statistics
    print("Testing proxy statistics endpoint...")
    try:
        response = requests.get(f"{base_url}/scrape/proxy/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"Proxy stats: {json.dumps(stats, indent=2)}")
        else:
            print(f"Proxy stats failed: {response.status_code}")
    except Exception as e:
        print(f"Error getting proxy stats: {e}")

    # Test 2: Refresh proxies
    print("\nRefreshing proxies...")
    try:
        response = requests.post(
            f"{base_url}/scrape/proxy/refresh", params={"force": True}
        )
        if response.status_code == 200:
            print(f"Proxy refresh: {response.json()}")
        else:
            print(f"Proxy refresh failed: {response.status_code}")
    except Exception as e:
        print(f"Error refreshing proxies: {e}")

    # Test 3: Get current proxy
    print("\nGetting current proxy...")
    try:
        response = requests.get(f"{base_url}/scrape/proxy/current")
        if response.status_code == 200:
            proxy_info = response.json()
            print(f"Current proxy: {json.dumps(proxy_info, indent=2)}")
        else:
            print(f"Get proxy failed: {response.status_code}")
    except Exception as e:
        print(f"Error getting proxy: {e}")

    # Test 4: Scrape with proxy (POST)
    print("\nTesting BeautifulSoup scraping with proxy (POST)...")
    scrape_request = {
        "url": "https://httpbin.org/html",
        "content_types": ["metadata", "articles"],
        "timeout": 30,
    }

    try:
        response = requests.post(
            f"{base_url}/scrape/beautifulsoup-scrape",
            json=scrape_request,
            params={"use_proxy": True},
        )
        if response.status_code == 200:
            result = response.json()
            print(f"Scraping with proxy successful!")
            print(f"Metadata: {result.get('metadata', {})}")
            if "article" in result:
                print(f"Article sections: {len(result['article'].get('sections', []))}")
        else:
            print(f"Scraping failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error scraping with proxy: {e}")

    # Test 5: Scrape without proxy for comparison
    print("\nTesting BeautifulSoup scraping without proxy (GET)...")
    try:
        response = requests.get(
            f"{base_url}/scrape/beautifulsoup-scrape",
            params={
                "url": "https://httpbin.org/html",
                "content_types": "metadata,articles",
                "use_proxy": False,
            },
        )
        if response.status_code == 200:
            result = response.json()
            print(f"Scraping without proxy successful!")
            print(f"Metadata: {result.get('metadata', {})}")
        else:
            print(f"Scraping without proxy failed: {response.status_code}")
    except Exception as e:
        print(f"Error scraping without proxy: {e}")


if __name__ == "__main__":
    test_proxy_scraping()
