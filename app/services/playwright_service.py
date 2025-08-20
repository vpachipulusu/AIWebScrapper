from fastapi import HTTPException
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
import os
import time
import subprocess
import sys
import re
from urllib.parse import urljoin

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
    Focuses on extracting structured content (tables or articles)
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
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )

            # Set viewport size
            page.set_viewport_size({"width": 1366, "height": 768})

            logger.info(f"Navigating to URL: {url}")

            # Try navigation with a wait for network to be mostly idle
            try:
                page.goto(url, timeout=45000, wait_until="networkidle")
                logger.info("Navigation successful")
            except PlaywrightTimeoutError:
                logger.warning(
                    "Navigation timeout, but continuing with available content"
                )
                # Even if timeout, we might have some content
            except Exception as e:
                logger.warning(
                    f"Navigation error: {e}, but continuing with available content"
                )

            # Wait a bit after navigation to ensure content is loaded
            time.sleep(3)

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

            # Check if page has tables with data
            tables = extract_structured_tables(page)
            content_type = None
            content = None

            if tables and len(tables) > 0:
                logger.info(
                    f"Found {len(tables)} table(s) on page, extracting table data"
                )
                content_type = "table"
                content = {
                    "tables": tables,
                    "table_count": len(tables),
                    "total_rows": sum(len(table["rows"]) for table in tables),
                }
            else:
                logger.info("No tables found, extracting article content")
                content_type = "article"
                content = extract_article_content(page)

            # Extract key points and explanations if available
            key_points = extract_key_points(page)
            explanations = extract_explanations(page)

            # Extract links
            links = extract_structured_links(page, url)

            logger.info("Scraping completed successfully")

            return {
                "metadata": {
                    "url": url,
                    "title": title,
                    "description": description,
                    "content_type": content_type,
                },
                "content": content,
                "key_points": key_points,
                "explanations": explanations,
                "links": links[:20],  # Limit to top 20 links
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


def extract_structured_tables(page):
    """
    Extract structured data from tables on the page
    """
    tables = []

    try:
        # Find all tables on the page
        table_elements = page.query_selector_all("table")

        for i, table in enumerate(table_elements):
            try:
                # Check if table has meaningful content
                rows = table.query_selector_all("tr")
                if len(rows) < 2:  # Skip tables with less than 2 rows
                    continue

                # Extract caption if available
                caption_element = table.query_selector("caption")
                caption = caption_element.inner_text() if caption_element else None

                # Extract headers
                headers = []
                header_row = table.query_selector("thead tr") or rows[0]
                header_cells = header_row.query_selector_all("th, td")

                for cell in header_cells:
                    headers.append(
                        cell.inner_text().strip() or f"Column {len(headers) + 1}"
                    )

                # Extract rows
                table_rows = []
                body_rows = (
                    table.query_selector_all("tbody tr")
                    if table.query_selector("tbody")
                    else rows
                )

                # Skip header row if it was in the body
                start_idx = (
                    1 if not table.query_selector("thead") and header_row in rows else 0
                )

                for row in body_rows[start_idx:]:
                    row_data = []
                    cells = row.query_selector_all("td, th")

                    for cell in cells:
                        # Check for rowspan and colspan
                        rowspan = int(cell.get_attribute("rowspan") or 1)
                        colspan = int(cell.get_attribute("colspan") or 1)

                        cell_text = cell.inner_text().strip()
                        row_data.append(
                            {"value": cell_text, "rowspan": rowspan, "colspan": colspan}
                        )

                    if row_data:
                        table_rows.append(row_data)

                if table_rows:  # Only add tables with data
                    tables.append(
                        {
                            "table_index": i,
                            "caption": caption,
                            "headers": headers,
                            "rows": table_rows,
                            "summary": f"Table with {len(headers)} columns and {len(table_rows)} rows",
                        }
                    )
            except Exception as e:
                logger.warning(f"Error extracting table {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in table extraction: {e}")

    return tables


def extract_article_content(page):
    """
    Extract structured article content from the page
    """
    try:
        # Try to find article using common selectors
        article_selectors = [
            "article",
            "main",
            '[role="main"]',
            ".article",
            ".content",
            ".post",
            ".blog-post",
            ".story",
            "#content",
            "#main",
            "#article",
        ]

        article_element = None

        for selector in article_selectors:
            article_element = page.query_selector(selector)
            if article_element:
                break

        # If no article found, use body
        if not article_element:
            article_element = page.query_selector("body")

        if not article_element:
            return {"sections": [], "text": ""}

        # Extract structured content
        sections = []
        current_section = {"heading": None, "paragraphs": []}

        # Get all elements within the article
        elements = article_element.query_selector_all("*")

        for element in elements:
            tag_name = element.evaluate("el => el.tagName.toLowerCase()")

            if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # New section found
                if current_section["paragraphs"] or current_section["heading"]:
                    sections.append(current_section)

                current_section = {
                    "heading": element.inner_text(),
                    "level": int(tag_name[1]),
                    "paragraphs": [],
                }
            elif tag_name in ["p", "div"]:
                # Paragraph content
                text = element.inner_text().strip()
                if text and len(text) > 30 and not is_noisy_text(text):
                    current_section["paragraphs"].append(text)
            elif tag_name in ["ul", "ol"]:
                # List content
                list_items = [
                    li.inner_text().strip() for li in element.query_selector_all("li")
                ]
                list_items = [item for item in list_items if item and len(item) > 10]

                if list_items:
                    current_section["paragraphs"].append(
                        {
                            "list_type": "ordered" if tag_name == "ol" else "unordered",
                            "items": list_items,
                        }
                    )

        # Add the last section
        if current_section["paragraphs"] or current_section["heading"]:
            sections.append(current_section)

        # Extract full text for completeness
        full_text = article_element.inner_text()
        full_text = re.sub(r"\n\s*\n", "\n\n", full_text)

        return {
            "sections": sections,
            "text": full_text,
            "section_count": len(sections),
            "paragraph_count": sum(len(section["paragraphs"]) for section in sections),
        }

    except Exception as e:
        logger.error(f"Error extracting article content: {e}")
        return {"sections": [], "text": "", "error": str(e)}


def extract_key_points(page):
    """
    Extract key points or important information from the page
    """
    key_points = []

    try:
        # Look for key points using common selectors
        key_point_selectors = [
            ".key-point",
            ".important",
            ".highlight",
            ".note",
            ".tip",
            ".alert",
            "[data-important]",
            "strong",
            "b",
        ]

        for selector in key_point_selectors:
            elements = page.query_selector_all(selector)
            for element in elements:
                text = element.inner_text().strip()
                if text and len(text) > 10 and not is_noisy_text(text):
                    key_points.append(text)

        # Remove duplicates
        key_points = list(set(key_points))

    except Exception as e:
        logger.error(f"Error extracting key points: {e}")

    return key_points


def extract_explanations(page):
    """
    Extract explanations or descriptive text from the page
    """
    explanations = []

    try:
        # Look for explanations using common selectors
        explanation_selectors = [
            ".explanation",
            ".description",
            ".instruction",
            ".guide",
            ".tutorial",
            ".help-text",
        ]

        for selector in explanation_selectors:
            elements = page.query_selector_all(selector)
            for element in elements:
                text = element.inner_text().strip()
                if text and len(text) > 20 and not is_noisy_text(text):
                    explanations.append(text)

    except Exception as e:
        logger.error(f"Error extracting explanations: {e}")

    return explanations


def extract_structured_links(page, base_url):
    """
    Extract structured link information
    """
    links = []

    try:
        link_elements = page.query_selector_all("a[href]")

        for link in link_elements:
            try:
                href = link.get_attribute("href")
                text = link.inner_text().strip()

                # Skip navigation and noisy links
                if not text or is_noisy_text(text):
                    continue

                # Make relative URLs absolute
                if href.startswith("/"):
                    href = urljoin(base_url, href)

                # Only include HTTP/HTTPS links
                if href.startswith(("http://", "https://")):
                    links.append(
                        {
                            "text": text,
                            "url": href,
                            "is_external": not href.startswith(base_url),
                            "text_length": len(text),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error processing link: {e}")
                continue

        # Sort by text length (longer texts are usually more meaningful)
        links.sort(key=lambda x: x["text_length"], reverse=True)

    except Exception as e:
        logger.error(f"Error extracting links: {e}")

    return links


def is_noisy_text(text):
    """
    Check if text is likely noise (navigation, ads, etc.)
    """
    if not text:
        return True

    text = text.lower().strip()

    # Common navigation phrases
    navigation_phrases = [
        "home",
        "about",
        "contact",
        "login",
        "sign up",
        "register",
        "privacy policy",
        "terms of service",
        "cookie policy",
        "follow us",
        "subscribe",
        "newsletter",
        "advertisement",
        "related articles",
        "you might also like",
        "popular posts",
        "categories",
        "tags",
        "archives",
        "search",
        "menu",
        "navigation",
        "cookie",
        "accept",
        "decline",
        "skip to content",
        "skip to main",
    ]

    # Check if text contains navigation phrases
    for phrase in navigation_phrases:
        if phrase in text:
            return True

    # Check if text is very short or looks like a button/link
    if len(text) < 25 and (text.isupper() or ">" in text or "→" in text or "›" in text):
        return True

    return False


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
