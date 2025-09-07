#!/usr/bin/env python3
"""
Test script for proxy service
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.proxy_service import ProxyManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_proxy_service():
    """Test the proxy service functionality."""
    print("Testing proxy service...")

    # Create a proxy manager with shorter timeout for testing
    manager = ProxyManager(max_proxies=5, test_timeout=8)

    # Try to refresh the proxy pool
    print("Refreshing proxy pool...")
    manager.refresh_proxy_pool(force=True)

    # Get stats
    stats = manager.get_proxy_stats()
    print(f"Proxy stats: {stats}")

    # Try to get a proxy
    proxy = manager.get_proxy()
    if proxy:
        print(f"Got proxy: {proxy}")

        # Test the proxy with a simple request
        import requests

        try:
            response = requests.get("http://httpbin.org/ip", proxies=proxy, timeout=10)
            if response.status_code == 200:
                print(f"Proxy test successful! Response: {response.text[:100]}")
                manager.report_proxy_result(proxy, True)
            else:
                print(f"Proxy test failed with status: {response.status_code}")
                manager.report_proxy_result(proxy, False)
        except Exception as e:
            print(f"Proxy test failed with error: {e}")
            manager.report_proxy_result(proxy, False)
    else:
        print("No proxy available")

    # Final stats
    final_stats = manager.get_proxy_stats()
    print(f"Final proxy stats: {final_stats}")


if __name__ == "__main__":
    test_proxy_service()
