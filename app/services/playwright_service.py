from fastapi import HTTPException
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
import os
import time
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_playwright_browsers():
    """Install Playwright browsers if they're not already installed"""
    try:
        logger.info("Checking if Playwright browsers are installed...")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if "are already installed" not in result.stdout:
            logger.info("Installing Playwright browsers...")
            subprocess.run(
                [sys.executable, "-m", "playwright", "install"], check=True, timeout=300
            )
            logger.info("Playwright browsers installed successfully")
        else:
            logger.info("Playwright browsers are already installed")
    except Exception as e:
        logger.error(f"Failed to install Playwright browsers: {e}")
        raise Exception(f"Failed to install Playwright browsers: {e}")


def sync_scrape(url: str) -> dict:
    """
    Synchronous scraping function with enhanced error handling
    """
    # Install browsers first if needed
    try:
        install_playwright_browsers()
    except Exception as e:
        logger.error(f"Browser installation failed: {e}")
        raise Exception(f"Browser installation failed: {e}")

    # Try to import Playwright
    try:
        from playwright.sync_api import (
            sync_playwright,
            TimeoutError as PlaywrightTimeoutError,
        )
    except ImportError as e:
        logger.error(f"Playwright not installed: {e}")
        raise Exception(
            "Playwright is not installed. Please run 'pip install playwright'"
        )

    browser = None
    try:
        logger.info(f"Starting sync scrape for URL: {url}")

        with sync_playwright() as p:
            logger.info("Launching browser...")

            # Try different browsers with simpler configuration
            browsers_to_try = ["chromium", "firefox"]
            browser = None
            errors = []

            for browser_type in browsers_to_try:
                try:
                    logger.info(f"Trying to launch {browser_type}...")
                    if browser_type == "chromium":
                        browser = p.chromium.launch(
                            headless=True,
                            timeout=30000,  # 30 second timeout for launch
                        )
                    elif browser_type == "firefox":
                        browser = p.firefox.launch(headless=True, timeout=30000)
                    logger.info(f"Successfully launched {browser_type}")
                    break
                except Exception as e:
                    error_msg = f"Failed to launch {browser_type}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue

            if not browser:
                raise Exception(
                    f"All browser types failed to launch. Errors: {', '.join(errors)}"
                )

            logger.info("Creating new page...")
            page = browser.new_page()

            # Set a user agent to avoid blocking
            page.set_extra_http_headers(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )

            # Set viewport size
            page.set_viewport_size({"width": 1366, "height": 768})

            logger.info(f"Navigating to URL: {url}")

            # Try navigation with a simpler approach
            try:
                page.goto(url, timeout=45000, wait_until="domcontentloaded")
                logger.info("Navigation successful")
            except PlaywrightTimeoutError:
                logger.warning(
                    "Navigation timeout, but continuing with available content"
                )
            except Exception as e:
                logger.warning(
                    f"Navigation error: {e}, but continuing with available content"
                )

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
                # More robust link extraction
                link_elements = page.query_selector_all("a")
                for link in link_elements:
                    try:
                        href = link.get_attribute("href")
                        text = link.text_content() or ""
                        if href and href.startswith(("http://", "https://", "/")):
                            links.append({"text": text.strip(), "href": href})
                    except Exception as e:
                        logger.warning(f"Error processing link: {e}")
                        continue
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

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, sync_scrape, url)
            return result

    except Exception as e:
        error_msg = str(e) if str(e) else "Unknown error occurred during scraping"
        logger.error(f"Error in scrape_page: {error_msg}")
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail=error_msg)
