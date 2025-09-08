"""
Proxy service for managing free proxy servers to avoid scraping limitations.
"""

import logging
import time
import requests
from typing import Dict, List, Optional, Any
from fp.fp import FreeProxy  # type: ignore
from threading import Lock
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProxyInfo:
    """Information about a proxy server."""

    proxy_url: str
    country: Optional[str] = None
    anonymity: Optional[str] = None
    https: bool = False
    response_time: Optional[float] = None
    last_used: Optional[float] = None
    success_count: int = 0
    failure_count: int = 0
    is_working: Optional[bool] = (
        True  # None = untested, True = working, False = not working
    )


class ProxyManager:
    """Manages a pool of free proxy servers with rotation and health checking."""

    def __init__(self, max_proxies: int = 10, test_timeout: int = 5):
        self.max_proxies = max_proxies
        self.test_timeout = test_timeout
        self.proxies: List[ProxyInfo] = []
        self.current_index = 0
        self.lock = Lock()
        self._last_refresh: float = 0
        self.refresh_interval = 600  # Refresh every 10 minutes (increased)
        self._cached_proxy_urls: List[str] = []  # Cache fetched but untested proxies
        self._last_fetch: float = 0
        self.fetch_interval = 300  # Fetch new proxy URLs every 5 minutes

    def fetch_proxy_urls_lazily(self) -> List[str]:
        """Fetch proxy URLs and cache them without testing."""
        current_time = time.time()

        # Check if we need to refresh cached URLs
        if (
            current_time - self._last_fetch
        ) > self.fetch_interval or not self._cached_proxy_urls:
            logger.info("Fetching fresh proxy URLs...")

            # Try free-proxy library first
            try:
                proxy_list = FreeProxy().get_proxy_list(repeat=1)  # Get 300 proxies
                if proxy_list:
                    # Limit to 20 for faster processing
                    self._cached_proxy_urls = [
                        f"http://{proxy}" for proxy in proxy_list[:20]
                    ]
                    logger.info(
                        f"Fetched {len(self._cached_proxy_urls)} proxy URLs from free-proxy"
                    )
                else:
                    raise Exception("No proxies from free-proxy")
            except Exception as e:
                logger.warning(f"Free-proxy failed: {e}, falling back to web sources")
                # Fallback to web sources
                web_proxies = self.get_proxies_from_web()
                self._cached_proxy_urls = [f"http://{proxy}" for proxy in web_proxies]
                logger.info(
                    f"Fetched {len(self._cached_proxy_urls)} proxy URLs from web sources"
                )

            self._last_fetch = current_time

        return self._cached_proxy_urls.copy()

    def get_proxies_from_web(self) -> List[str]:
        """Get proxies by scraping public proxy lists."""
        proxies: List[str] = []

        try:
            # Simple public proxy list - this is just an example
            # In production, you might want to use paid proxy services
            public_proxies = [
                "8.210.83.33:80",
                "47.74.152.29:8888",
                "43.134.68.153:3128",
                "103.186.1.200:3128",
                "190.61.88.147:8080",
                "103.127.1.130:80",
                "138.197.102.119:80",
                "165.227.71.60:80",
                "159.89.195.14:80",
                "134.209.29.120:8080",
            ]

            proxies.extend(public_proxies)
            logger.info(f"Got {len(proxies)} proxies from web sources")

        except Exception as e:
            logger.error(f"Error getting proxies from web: {e}")

        return proxies

    def get_fresh_proxies(self, count: int = 10) -> List[str]:
        """Get fresh proxy servers from free-proxy sources."""
        fresh_proxies: List[str] = []

        try:
            # Get proxies with different criteria - simplified approach
            proxy_sources = [
                {"https": True},
                {"country_id": ["US"], "https": True},
                {"country_id": ["GB"], "https": True},
                {"elite": True},
                {"anonymity": True},  # Fixed: was 'anonymous'
                {},  # Default settings
            ]

            for source_config in proxy_sources:
                if len(fresh_proxies) >= count:
                    break

                try:
                    fp = FreeProxy(**source_config)
                    proxy = fp.get()
                    if proxy and proxy not in fresh_proxies:
                        fresh_proxies.append(proxy)
                        logger.info(f"Found proxy from FreeProxy: {proxy}")
                except Exception as e:
                    logger.debug(
                        f"Error getting proxy with config {source_config}: {e}"
                    )
                    continue

            # If we still don't have enough, try without any filters
            if len(fresh_proxies) < count:
                for _ in range(count - len(fresh_proxies)):
                    try:
                        fp = FreeProxy()  # No filters
                        proxy = fp.get()
                        if proxy and proxy not in fresh_proxies:
                            fresh_proxies.append(proxy)
                            logger.info(
                                f"Found proxy from FreeProxy (no filter): {proxy}"
                            )
                    except Exception as e:
                        logger.debug(f"Error getting random proxy: {e}")
                        break

        except Exception as e:
            logger.error(f"Error fetching fresh proxies: {e}")

        # If still no proxies, add some manual fallback proxies (public proxies)
        if len(fresh_proxies) < count:
            logger.info("Getting proxies from web sources as fallback")
            web_proxies = self.get_proxies_from_web()
            for proxy in web_proxies:
                if len(fresh_proxies) >= count:
                    break
                if proxy not in fresh_proxies:
                    fresh_proxies.append(proxy)
                    logger.info(f"Added proxy from web source: {proxy}")

        return fresh_proxies

    def test_proxy(self, proxy_url: str) -> ProxyInfo:
        """Test a proxy server and return its info."""
        proxy_info = ProxyInfo(proxy_url=proxy_url)

        try:
            # Ensure proxy URL has protocol
            if not proxy_url.startswith(("http://", "https://")):
                proxy_url = f"http://{proxy_url}"
                proxy_info.proxy_url = proxy_url

            # Set up proxy configuration
            proxies = {"http": proxy_url, "https": proxy_url}

            # Test with a simple request - use multiple test URLs
            test_urls = [
                "http://httpbin.org/ip",
                "http://ipinfo.io/json",
                "http://ip-api.com/json",
            ]

            for test_url in test_urls:
                try:
                    start_time = time.time()
                    response = requests.get(
                        test_url,
                        proxies=proxies,
                        timeout=min(self.test_timeout, 15),  # Cap at 15 seconds
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        },
                    )

                    response_time = time.time() - start_time

                    if response.status_code == 200:
                        proxy_info.response_time = response_time
                        proxy_info.https = test_url.startswith("https")
                        proxy_info.is_working = True
                        proxy_info.success_count = 1

                        # Try to get more info about the proxy
                        try:
                            data = response.json()
                            ip_info = data.get(
                                "origin", data.get("ip", data.get("query", "unknown"))
                            )
                            logger.info(
                                f"Proxy {proxy_url} working, IP: {ip_info}, Response time: {response_time:.2f}s"
                            )
                        except Exception:
                            logger.info(
                                f"Proxy {proxy_url} working, Response time: {response_time:.2f}s"
                            )

                        return proxy_info  # Return on first success

                except requests.exceptions.Timeout:
                    logger.debug(f"Proxy {proxy_url} timeout on {test_url}")
                    continue
                except requests.exceptions.ConnectionError:
                    logger.debug(f"Proxy {proxy_url} connection error on {test_url}")
                    continue
                except Exception as e:
                    logger.debug(f"Proxy {proxy_url} error on {test_url}: {e}")
                    continue

            # If we reach here, all test URLs failed
            proxy_info.is_working = False
            proxy_info.failure_count = 1

        except Exception as e:
            logger.warning(f"Proxy {proxy_url} failed test: {e}")
            proxy_info.is_working = False
            proxy_info.failure_count = 1

        return proxy_info

    def refresh_proxy_pool(self, force: bool = False) -> None:
        """Refresh the proxy pool with lazy loading - only fetch URLs, test on-demand."""
        current_time = time.time()

        if not force and current_time - self._last_refresh < self.refresh_interval:
            return

        with self.lock:
            logger.info("Refreshing proxy pool (lazy loading)...")

            # Get cached proxy URLs without testing them
            fresh_proxy_urls = self.fetch_proxy_urls_lazily()

            if not fresh_proxy_urls:
                logger.warning("No fresh proxy URLs found")
                return

            # Keep existing working proxies
            existing_good_proxies = [
                p
                for p in self.proxies
                if p.is_working
                and p.failure_count < 3
                and p.success_count > p.failure_count
            ]

            # Add new proxy URLs as untested proxies (lazy approach)
            new_proxies: List[ProxyInfo] = []
            for proxy_url in fresh_proxy_urls:
                if len(existing_good_proxies) + len(new_proxies) >= self.max_proxies:
                    break

                # Check if we already have this proxy
                if not any(p.proxy_url == proxy_url for p in existing_good_proxies):
                    proxy_info = ProxyInfo(proxy_url=proxy_url)
                    proxy_info.is_working = None  # Mark as untested
                    new_proxies.append(proxy_info)

            # Combine existing good proxies with new untested ones
            self.proxies = existing_good_proxies + new_proxies
            self.current_index = 0
            self._last_refresh = current_time

            logger.info(
                f"Proxy pool refreshed with {len(existing_good_proxies)} working proxies "
                f"and {len(new_proxies)} untested proxies (lazy loading)"
            )

    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next proxy in rotation, testing on-demand."""
        # Refresh if needed
        self.refresh_proxy_pool()

        if not self.proxies:
            logger.warning("No proxies available")
            return None

        # First, try to find an already working proxy
        with self.lock:
            for proxy_info in self.proxies:
                if proxy_info.is_working is True:
                    proxy_info.last_used = time.time()
                    return {"http": proxy_info.proxy_url, "https": proxy_info.proxy_url}

        # If no working proxies, test untested ones on-demand
        max_attempts = min(len(self.proxies), 5)  # Test up to 5 proxies
        attempts = 0

        while attempts < max_attempts:
            proxy_to_test = None

            # Find next untested proxy
            with self.lock:
                for proxy_info in self.proxies:
                    if proxy_info.is_working is None:  # Untested
                        proxy_to_test = proxy_info
                        break

            if not proxy_to_test:
                break  # No more untested proxies

            attempts += 1
            logger.info(
                f"Testing proxy on-demand ({attempts}/{max_attempts}): {proxy_to_test.proxy_url}"
            )
            tested_info = self.test_proxy(proxy_to_test.proxy_url)

            # Update the proxy info with test results
            with self.lock:
                for p in self.proxies:
                    if p.proxy_url == proxy_to_test.proxy_url:
                        p.is_working = tested_info.is_working
                        p.response_time = tested_info.response_time
                        p.success_count = tested_info.success_count
                        p.failure_count = tested_info.failure_count

                        if p.is_working:
                            p.last_used = time.time()
                            logger.info(f"Found working proxy: {p.proxy_url}")
                            return {"http": p.proxy_url, "https": p.proxy_url}
                        break

        logger.warning(f"No working proxies found after testing {attempts} proxies")
        return None

    def report_proxy_result(self, proxy_dict: Dict[str, str], success: bool) -> None:
        """Report the result of using a proxy."""
        if not proxy_dict:
            return

        proxy_url = proxy_dict.get("http", proxy_dict.get("https"))
        if not proxy_url:
            return

        with self.lock:
            for proxy_info in self.proxies:
                if proxy_info.proxy_url == proxy_url:
                    if success:
                        proxy_info.success_count += 1
                    else:
                        proxy_info.failure_count += 1
                        # Mark as not working if too many failures
                        if proxy_info.failure_count > 3:
                            proxy_info.is_working = False
                    break

    def get_proxy_stats(self) -> Dict[str, Any]:
        """Get statistics about the proxy pool."""
        working_count = sum(1 for p in self.proxies if p.is_working)
        total_success = sum(p.success_count for p in self.proxies)
        total_failure = sum(p.failure_count for p in self.proxies)

        return {
            "total_proxies": len(self.proxies),
            "working_proxies": working_count,
            "total_success": total_success,
            "total_failure": total_failure,
            "success_rate": total_success / max(total_success + total_failure, 1),
            "last_refresh": self._last_refresh,
        }


# Global proxy manager instance
proxy_manager = ProxyManager()


def get_proxy_for_request() -> Optional[Dict[str, str]]:
    """Get a proxy for making requests."""
    return proxy_manager.get_proxy()


def report_proxy_success(proxy_dict: Dict[str, str]) -> None:
    """Report successful use of a proxy."""
    proxy_manager.report_proxy_result(proxy_dict, success=True)


def report_proxy_failure(proxy_dict: Dict[str, str]) -> None:
    """Report failed use of a proxy."""
    proxy_manager.report_proxy_result(proxy_dict, success=False)


def refresh_proxies(force: bool = False) -> None:
    """Refresh the proxy pool."""
    proxy_manager.refresh_proxy_pool(force=force)


def get_proxy_statistics() -> Dict[str, Any]:
    """Get proxy pool statistics."""
    return proxy_manager.get_proxy_stats()
