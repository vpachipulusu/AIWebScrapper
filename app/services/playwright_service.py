from fastapi import HTTPException
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sync_scrape(url: str) -> dict:
    """
    Synchronous scraping function with enhanced error handling for page navigation
    """
    # Try to import Playwright only when needed
    try:
        from playwright.sync_api import (
            sync_playwright,
            TimeoutError as PlaywrightTimeoutError,
        )
    except ImportError as e:
        logger.error(f"Playwright not installed: {e}")
        raise Exception(
            "Playwright is not installed. Please run 'pip install playwright' and 'playwright install'"
        )

    browser = None
    try:
        logger.info(f"Starting sync scrape for URL: {url}")

        # Set environment variables to help with Windows compatibility
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
        os.environ["PLAYWRIGHT_DOWNLOAD_HOST"] = ""

        with sync_playwright() as p:
            logger.info("Launching browser...")

            # Try different browsers in order
            browsers_to_try = ["chromium", "firefox", "webkit"]
            browser = None

            for browser_type in browsers_to_try:
                try:
                    if browser_type == "chromium":
                        browser = p.chromium.launch(
                            headless=True,
                            args=[
                                "--disable-gpu",
                                "--disable-dev-shm-usage",
                                "--disable-setuid-sandbox",
                                "--no-sandbox",
                                "--disable-web-security",
                            ],
                        )
                    elif browser_type == "firefox":
                        browser = p.firefox.launch(
                            headless=True,
                            args=["--disable-gpu", "--disable-dev-shm-usage"],
                        )
                    elif browser_type == "webkit":
                        browser = p.webkit.launch(
                            headless=True,
                            args=["--disable-gpu", "--disable-dev-shm-usage"],
                        )
                    logger.info(f"Successfully launched {browser_type}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to launch {browser_type}: {e}")
                    continue

            if not browser:
                raise Exception("All browser types failed to launch")

            logger.info("Creating new page...")
            page = browser.new_page()

            # Set a user agent to avoid blocking
            page.set_extra_http_headers(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
            )

            # Set viewport size
            page.set_viewport_size({"width": 1366, "height": 768})

            logger.info(f"Navigating to URL: {url}")

            # Try multiple navigation strategies
            navigation_success = False
            for wait_until in ["domcontentloaded", "load", "networkidle"]:
                try:
                    logger.info(f"Trying navigation with wait_until: {wait_until}")
                    page.goto(
                        url,
                        timeout=45000,  # Slightly shorter timeout
                        wait_until=wait_until,
                    )
                    navigation_success = True
                    logger.info(f"Navigation successful with {wait_until}")
                    break
                except PlaywrightTimeoutError:
                    logger.warning(f"Timeout with wait_until: {wait_until}")
                    continue
                except Exception as e:
                    logger.warning(f"Error with wait_until {wait_until}: {e}")
                    continue

            if not navigation_success:
                # Final attempt with no specific wait condition
                try:
                    logger.info("Trying navigation with no wait_until condition")
                    page.goto(url, timeout=30000)
                    navigation_success = True
                    logger.info("Navigation successful with no wait_until")
                except Exception as e:
                    logger.error(f"All navigation attempts failed: {e}")
                    raise Exception(f"Failed to navigate to URL: {str(e)}")

            # Wait a bit after navigation to ensure content is loaded
            time.sleep(2)

            logger.info("Extracting page title...")
            title = page.title()

            logger.info("Extracting description...")
            description = None
            try:
                description_element = page.locator("meta[name='description']")
                if description_element.count() > 0:
                    description = description_element.get_attribute("content")
            except Exception as e:
                logger.warning(f"Error extracting description: {e}")

            logger.info("Extracting links...")
            links = []
            try:
                links = page.evaluate("""() => {
                    return Array.from(document.querySelectorAll('a')).map(a => ({
                        text: a.innerText.trim(),
                        href: a.href
                    }));
                }""")
            except Exception as e:
                logger.warning(f"Error extracting links: {e}")

            logger.info("Scraping completed successfully")

            return {
                "url": url,
                "title": title,
                "description": description,
                "links": links,
            }

    except Exception as e:
        logger.error(f"Error in sync_scrape: {str(e)}")
        logger.error(traceback.format_exc())
        error_msg = str(e) if str(e) else "Unknown error in sync_scrape"

        # Check for common Playwright issues
        if "executable doesn't exist" in error_msg.lower():
            error_msg = "Browser not installed. Please run 'playwright install'"
        elif "target closed" in error_msg.lower():
            error_msg = (
                "Browser closed unexpectedly. This might be a compatibility issue."
            )
        elif "navigation" in error_msg.lower() and "timeout" in error_msg.lower():
            error_msg = (
                "Page loading timeout. The site might be slow or blocking requests."
            )

        raise Exception(error_msg)
    finally:
        if browser:
            logger.info("Closing browser...")
            try:
                browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")


async def scrape_page(url: str) -> dict:
    """
    Async wrapper for the synchronous scraping function
    """
    try:
        # Use a thread pool executor to run synchronous code
        loop = asyncio.get_event_loop()

        # Use a separate process if thread doesn't work
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, sync_scrape, url)
            return result

    except Exception as e:
        error_msg = str(e) if str(e) else "Unknown error occurred during scraping"
        logger.error(f"Error in scrape_page: {error_msg}")
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail=error_msg)
