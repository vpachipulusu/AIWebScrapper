#!/usr/bin/env python3
"""Test script to test the API endpoint directly."""

import requests
import json


def test_api_endpoint():
    """Test the pandas-enhanced API endpoint."""
    print("Testing API endpoint...")

    try:
        url = "http://localhost:8000/scrape/beautifulsoup-pandas"
        payload = {"url": "https://httpbin.org/html"}
        headers = {"Content-Type": "application/json"}

        print(f"Making request to: {url}")
        print(f"Payload: {payload}")

        response = requests.post(url, json=payload, headers=headers, timeout=30)

        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success! Keys: {list(data.keys())}")
        else:
            print(f"✗ Error: {response.status_code}")

    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_api_endpoint()
