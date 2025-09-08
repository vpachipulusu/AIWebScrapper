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
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
from .data_processor import convert_numpy_types
from .proxy_service import (
    get_proxy_for_request,
    report_proxy_success,
    report_proxy_failure,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define content types that can be extracted
class ContentType(str, Enum):
    TABLES = "tables"
    LISTS = "lists"
    ARTICLES = "articles"
    KEY_POINTS = "key_points"
    EXPLANATIONS = "explanations"
    LINKS = "links"
    METADATA = "metadata"
    CUSTOM = "custom"


# Define selector types
class SelectorType(str, Enum):
    CSS = "css"
    XPATH = "xpath"


# Model for custom selectors
class CustomSelector(BaseModel):
    name: str = Field(..., description="Name for this selector (e.g., 'product_title')")
    selector: str = Field(..., description="The CSS selector or XPath expression")
    selector_type: SelectorType = Field(
        SelectorType.CSS, description="Type of selector"
    )
    extract: str = Field(
        "text", description="What to extract: 'text', 'html', 'attribute', or 'all'"
    )
    attribute: Optional[str] = Field(
        None, description="If extract='attribute', which attribute to extract"
    )


# Request model for the API
class ScrapeRequest(BaseModel):
    url: str
    content_types: List[ContentType] = Field(
        default=[
            ContentType.TABLES,
            ContentType.LISTS,
            ContentType.ARTICLES,
            ContentType.METADATA,
        ],
        description="Types of content to extract",
    )
    custom_selectors: List[CustomSelector] = Field(
        default=[], description="Custom selectors for extracting specific content"
    )
    timeout: int = Field(30, description="Timeout in seconds")
    wait_after_load: int = Field(2, description="Seconds to wait after page load")
    return_html: bool = Field(
        False, description="Whether to return raw HTML for analysis"
    )


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


def sync_scrape(request: ScrapeRequest) -> dict:
    """
    Synchronous scraping function with user-selectable content extraction
    and custom selector support
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
        logger.info(f"Starting sync scrape for URL: {request.url}")
        logger.info(f"Content types to extract: {request.content_types}")
        if request.custom_selectors:
            logger.info(
                f"Custom selectors: {[s.name for s in request.custom_selectors]}"
            )

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

            logger.info(f"Navigating to URL: {request.url}")

            # Try navigation with a wait for network to be mostly idle
            try:
                page.goto(
                    request.url,
                    timeout=request.timeout * 1000,
                    wait_until="networkidle",
                )
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
            time.sleep(request.wait_after_load)

            # Initialize result dictionary
            result = {}

            # Extract metadata if requested
            if ContentType.METADATA in request.content_types:
                logger.info("Extracting metadata...")
                result["metadata"] = extract_metadata(page, request.url)

            # Extract tables if requested
            if ContentType.TABLES in request.content_types:
                logger.info("Extracting tables...")
                tables = extract_structured_tables(page)
                if tables:
                    result["tables"] = tables

            # Extract lists if requested
            if ContentType.LISTS in request.content_types:
                logger.info("Extracting lists...")
                lists = extract_structured_lists(page)
                if lists:
                    result["lists"] = lists

            # Extract articles if requested
            if ContentType.ARTICLES in request.content_types:
                logger.info("Extracting articles...")
                article = extract_article_content(page)
                if article and (article.get("sections") or article.get("text")):
                    result["article"] = article

            # Extract key points if requested
            if ContentType.KEY_POINTS in request.content_types:
                logger.info("Extracting key points...")
                key_points = extract_key_points(page)
                if key_points:
                    result["key_points"] = key_points

            # Extract explanations if requested
            if ContentType.EXPLANATIONS in request.content_types:
                logger.info("Extracting explanations...")
                explanations = extract_explanations(page)
                if explanations:
                    result["explanations"] = explanations

            # Extract links if requested
            if ContentType.LINKS in request.content_types:
                logger.info("Extracting links...")
                links = extract_structured_links(page, request.url)
                if links:
                    result["links"] = links[:20]  # Limit to top 20 links

            # Extract custom content if requested
            if ContentType.CUSTOM in request.content_types and request.custom_selectors:
                logger.info("Extracting custom content...")
                custom_content = extract_custom_content(page, request.custom_selectors)
                if custom_content:
                    result["custom"] = custom_content

            # Return HTML for analysis if requested
            if request.return_html:
                logger.info("Extracting HTML for analysis...")
                result["html"] = extract_html_for_analysis(page)

            logger.info("Scraping completed successfully")
            
            # Convert NumPy types to native Python types for JSON serialization
            result = convert_numpy_types(result)
            
            return result

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
                # Ignore "Event loop is closed" errors as they're usually harmless
                if "Event loop is closed" not in str(e):
                    logger.warning(f"Error closing browser: {e}")


def extract_custom_content(
    page, custom_selectors: List[CustomSelector]
) -> Dict[str, Any]:
    """
    Extract content using custom CSS selectors or XPaths
    """
    custom_content = {}

    for selector_def in custom_selectors:
        try:
            elements = []

            if selector_def.selector_type == SelectorType.CSS:
                elements = page.query_selector_all(selector_def.selector)
            elif selector_def.selector_type == SelectorType.XPATH:
                # For XPath, we need to use a different approach
                try:
                    elements = page.query_selector_all(f"xpath={selector_def.selector}")
                except Exception as e:
                    # Fallback to evaluate for complex XPath expressions
                    logger.warning(f"XPath selector failed, using fallback: {e}")
                    elements = page.evaluate(f"""() => {{
                        const results = [];
                        const nodes = document.evaluate('{selector_def.selector}', document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                        for (let i = 0; i < nodes.snapshotLength; i++) {{
                            results.push(nodes.snapshotItem(i));
                        }}
                        return results;
                    }}""")

            if not elements:
                custom_content[selector_def.name] = None
                continue

            extracted_data = []

            for element in elements:
                try:
                    if selector_def.extract == "text":
                        text = element.text_content()
                        extracted_data.append(text.strip() if text else "")
                    elif selector_def.extract == "html":
                        html = element.inner_html()
                        extracted_data.append(html)
                    elif selector_def.extract == "attribute" and selector_def.attribute:
                        attr_value = element.get_attribute(selector_def.attribute)
                        extracted_data.append(attr_value if attr_value else "")
                    elif selector_def.extract == "all":
                        # Extract both text and HTML
                        text = element.text_content()
                        html = element.inner_html()
                        extracted_data.append(
                            {"text": text.strip() if text else "", "html": html}
                        )
                    else:
                        # Default to text
                        text = element.text_content()
                        extracted_data.append(text.strip() if text else "")
                except Exception as e:
                    logger.error(
                        f"Error extracting content from element with selector '{selector_def.name}': {e}"
                    )
                    extracted_data.append(f"Error: {str(e)}")

            # If only one element found, return it directly instead of as a list
            if len(extracted_data) == 1:
                custom_content[selector_def.name] = extracted_data[0]
            else:
                custom_content[selector_def.name] = extracted_data

        except Exception as e:
            logger.error(
                f"Error extracting content with selector '{selector_def.name}': {e}"
            )
            custom_content[selector_def.name] = f"Error: {str(e)}"

    return custom_content


def extract_html_for_analysis(page) -> Dict[str, Any]:
    """
    Extract HTML structure for analysis to help users create selectors
    """
    try:
        # Get the main content areas
        body_html = page.query_selector("body").inner_html()

        # Find all unique CSS classes and IDs for analysis
        classes_and_ids = page.evaluate("""
            () => {
                const allElements = document.querySelectorAll('*');
                const classes = new Set();
                const ids = new Set();
                
                allElements.forEach(el => {
                    // Handle className - it might not always be a string
                    if (el.className) {
                        let className = el.className;
                        // Handle cases where className is not a string (e.g., SVG elements)
                        if (typeof className === 'string') {
                            className.split(' ').forEach(cls => {
                                if (cls) classes.add(cls);
                            });
                        } else if (className.baseVal) {
                            // Handle SVG elements which have className as SVGAnimatedString
                            className.baseVal.split(' ').forEach(cls => {
                                if (cls) classes.add(cls);
                            });
                        }
                    }
                    if (el.id) ids.add(el.id);
                });
                
                return {
                    classes: Array.from(classes),
                    ids: Array.from(ids)
                };
            }
        """)

        # Get common content containers
        content_containers = {}
        container_selectors = {
            "articles": "article",
            "main_content": "main, [role='main'], .main, #main, .content, #content",
            "headers": "h1, h2, h3, h4, h5, h6",
            "tables": "table",
            "lists": "ul, ol",
        }

        for name, selector in container_selectors.items():
            elements = page.query_selector_all(selector)
            content_containers[name] = len(elements)

        return {
            "structure": content_containers,
            "classes": classes_and_ids["classes"],
            "ids": classes_and_ids["ids"],
            "body_html": body_html[:5000] + "..."
            if len(body_html) > 5000
            else body_html,  # Limit size
        }
    except Exception as e:
        logger.error(f"Error extracting HTML for analysis: {e}")
        return {"error": str(e)}


def extract_metadata(page, url):
    """
    Extract metadata from the page
    """
    try:
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

        return {
            "url": url,
            "title": title,
            "description": description,
        }
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {"error": str(e)}


def extract_structured_tables(page):
    """
    Extract structured data from tables on the page with pandas processing
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

                # Extract rows data for pandas DataFrame
                table_data = []
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
                        cell_text = cell.inner_text().strip()
                        row_data.append(cell_text)

                    # Pad row to match header length
                    while len(row_data) < len(headers):
                        row_data.append("")

                    # Truncate if too long
                    row_data = row_data[: len(headers)]

                    if any(row_data):  # Only add non-empty rows
                        table_data.append(row_data)

                if table_data:  # Only process tables with data
                    # Create pandas DataFrame
                    try:
                        df = pd.DataFrame(table_data, columns=headers)

                        # Clean the DataFrame
                        df = clean_playwright_dataframe(df)

                        # Generate statistics
                        df_stats = generate_dataframe_stats(df)

                        tables.append(
                            {
                                "table_index": i,
                                "caption": caption,
                                "headers": headers,
                                "dataframe": df.to_dict("records"),
                                "shape": df.shape,
                                "statistics": df_stats,
                                "pandas_summary": {
                                    "dtypes": df.dtypes.to_dict(),
                                    "null_counts": df.isnull().sum().to_dict(),
                                    "description": df.describe(include="all").to_dict()
                                    if not df.empty
                                    else {},
                                },
                                "summary": f"Table with {len(headers)} columns and {len(table_data)} rows",
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error processing table {i} with pandas: {e}")
                        # Fallback to original format
                        table_rows = []
                        for row_data in table_data:
                            row_cells = []
                            for cell_text in row_data:
                                row_cells.append(
                                    {"value": cell_text, "rowspan": 1, "colspan": 1}
                                )
                            table_rows.append(row_cells)

                        tables.append(
                            {
                                "table_index": i,
                                "caption": caption,
                                "headers": headers,
                                "rows": table_rows,
                                "summary": f"Table with {len(headers)} columns and {len(table_data)} rows",
                            }
                        )

            except Exception as e:
                logger.warning(f"Error extracting table {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in table extraction: {e}")

    return tables


def clean_playwright_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and format a pandas DataFrame from Playwright extraction."""
    # Remove completely empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["", "None", "nan", "NaN"], np.nan)

    # Try to convert numeric columns
    for col in df.columns:
        if df[col].dtype == "object":
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if numeric_series.notna().sum() > len(df) * 0.6:  # If 60% can be converted
                df[col] = numeric_series
            else:
                # Try to convert to datetime
                try:
                    datetime_series = pd.to_datetime(
                        df[col], errors="coerce", infer_datetime_format=True
                    )
                    if datetime_series.notna().sum() > len(df) * 0.6:
                        df[col] = datetime_series
                except Exception:
                    pass

    # Reset index
    df = df.reset_index(drop=True)

    return df


def generate_dataframe_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive statistics for a DataFrame."""
    if df.empty:
        return {}

    stats = {
        "shape": df.shape,
        "memory_usage": df.memory_usage(deep=True).sum(),
        "column_types": df.dtypes.to_dict(),
    }

    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
        stats["correlations"] = (
            df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {}
        )

    # Text columns analysis
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if text_cols:
        stats["text_summary"] = {}
        for col in text_cols:
            valid_values = df[col].dropna()
            if not valid_values.empty:
                stats["text_summary"][col] = {
                    "unique_count": valid_values.nunique(),
                    "top_values": valid_values.value_counts().head(5).to_dict(),
                    "avg_length": valid_values.astype(str).str.len().mean(),
                    "max_length": valid_values.astype(str).str.len().max(),
                }

    # Missing data analysis
    stats["missing_data"] = {
        "total_missing": df.isnull().sum().sum(),
        "missing_by_column": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }

    # Convert NumPy types to native Python types for JSON serialization
    stats = convert_numpy_types(stats)

    return stats


def extract_structured_lists(page):
    """
    Extract structured data from lists on the page with pandas processing
    """
    lists = []

    try:
        # Find all lists on the page
        list_elements = page.query_selector_all("ul, ol")

        for i, list_element in enumerate(list_elements):
            try:
                # Skip lists with few items (likely navigation)
                items = list_element.query_selector_all("li")
                if len(items) < 3:  # Skip lists with less than 3 items
                    continue

                # Determine list type
                list_type = "unordered"
                if list_element.evaluate("el => el.tagName.toLowerCase()") == "ol":
                    list_type = "ordered"

                # Extract list items
                list_items = []
                for item in items:
                    item_text = item.text_content().strip()
                    if (
                        item_text and len(item_text) > 2
                    ):  # Skip empty or very short items
                        list_items.append(item_text)

                if list_items:  # Only add lists with items
                    # Try to find a heading or context for the list
                    context = find_list_context(list_element)

                    # Create pandas DataFrame for list analysis
                    try:
                        list_df = pd.DataFrame(
                            {
                                "item": list_items,
                                "item_number": range(1, len(list_items) + 1),
                                "length": [len(item) for item in list_items],
                                "word_count": [
                                    len(item.split()) for item in list_items
                                ],
                                "has_numbers": [
                                    bool(re.search(r"\d", item)) for item in list_items
                                ],
                                "has_special_chars": [
                                    bool(re.search(r"[^\w\s]", item))
                                    for item in list_items
                                ],
                            }
                        )

                        # Generate list statistics
                        list_stats = {
                            "total_items": len(list_items),
                            "avg_length": list_df["length"].mean(),
                            "avg_word_count": list_df["word_count"].mean(),
                            "longest_item": list_df.loc[
                                list_df["length"].idxmax(), "item"
                            ]
                            if not list_df.empty
                            else "",
                            "shortest_item": list_df.loc[
                                list_df["length"].idxmin(), "item"
                            ]
                            if not list_df.empty
                            else "",
                            "items_with_numbers": list_df["has_numbers"].sum(),
                            "items_with_special_chars": list_df[
                                "has_special_chars"
                            ].sum(),
                            "length_distribution": list_df["length"]
                            .describe()
                            .to_dict(),
                        }

                        # Analyze patterns in list items
                        patterns = analyze_list_patterns(list_items)

                        lists.append(
                            {
                                "list_index": i,
                                "type": list_type,
                                "context": context,
                                "items": list_items,
                                "item_count": len(list_items),
                                "dataframe": list_df.to_dict("records"),
                                "statistics": list_stats,
                                "patterns": patterns,
                                "pandas_analysis": {
                                    "dtypes": list_df.dtypes.to_dict(),
                                    "correlations": list_df.select_dtypes(
                                        include=[np.number]
                                    )
                                    .corr()
                                    .to_dict()
                                    if len(
                                        list_df.select_dtypes(
                                            include=[np.number]
                                        ).columns
                                    )
                                    > 1
                                    else {},
                                },
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error processing list {i} with pandas: {e}")
                        # Fallback to original format
                        lists.append(
                            {
                                "list_index": i,
                                "type": list_type,
                                "context": context,
                                "items": list_items,
                                "item_count": len(list_items),
                            }
                        )

            except Exception as e:
                logger.warning(f"Error extracting list {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in list extraction: {e}")

    return lists


def analyze_list_patterns(items: List[str]) -> Dict[str, Any]:
    """Analyze patterns in list items using pandas."""
    if not items:
        return {}

    # Create DataFrame for pattern analysis
    df = pd.DataFrame({"item": items})

    # Common patterns
    patterns = {
        "starts_with_number": df["item"].str.match(r"^\d+").sum(),
        "starts_with_bullet": df["item"].str.match(r"^[•·▪▫◦‣⁃]").sum(),
        "contains_colon": df["item"].str.contains(":").sum(),
        "contains_dash": df["item"].str.contains("-").sum(),
        "contains_parentheses": df["item"].str.contains(r"\(.*\)").sum(),
        "all_caps_words": df["item"].str.findall(r"\b[A-Z]{2,}\b").apply(len).sum(),
        "contains_urls": df["item"].str.contains(r"http[s]?://").sum(),
        "contains_emails": df["item"]
        .str.contains(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        .sum(),
    }

    # Length patterns
    lengths = df["item"].str.len()
    patterns.update(
        {
            "length_variance": lengths.var(),
            "consistent_length": (lengths.max() - lengths.min())
            < 10,  # Low variance in length
            "similar_structure": check_similar_structure(items),
        }
    )

    # Convert NumPy types to native Python types for JSON serialization
    patterns = convert_numpy_types(patterns)

    return patterns


def check_similar_structure(items: List[str]) -> bool:
    """Check if list items have similar structure."""
    if len(items) < 3:
        return False

    # Simple structure check: similar number of words
    word_counts = [len(item.split()) for item in items]
    avg_words = sum(word_counts) / len(word_counts)
    variance = sum((x - avg_words) ** 2 for x in word_counts) / len(word_counts)

    return variance < 2.0  # Low variance suggests similar structure


def find_list_context(list_element):
    """
    Try to find context or heading for a list using Playwright's JavaScript evaluation
    """
    try:
        # Use JavaScript evaluation to find context
        context = list_element.evaluate("""
            (el) => {
                // Look for previous sibling that might be a heading
                let previous = el.previousElementSibling;
                if (previous) {
                    let tagName = previous.tagName ? previous.tagName.toLowerCase() : '';
                    if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
                        return previous.innerText || previous.textContent;
                    }
                    
                    // Check if previous element has a class that suggests it's a heading
                    let className = previous.className || '';
                    if (typeof className === 'string') {
                        if (className.includes('title') || className.includes('heading') || 
                            className.includes('header') || className.includes('caption')) {
                            return previous.innerText || previous.textContent;
                        }
                    } else if (className.baseVal) {
                        // Handle SVG elements
                        if (className.baseVal.includes('title') || className.baseVal.includes('heading') || 
                            className.baseVal.includes('header') || className.baseVal.includes('caption')) {
                            return previous.innerText || previous.textContent;
                        }
                    }
                }
                
                // Look for parent element that might provide context
                let parent = el.parentElement;
                if (parent) {
                    // Check if parent has a class that suggests it's a content container
                    let parentClassName = parent.className || '';
                    if (typeof parentClassName === 'string') {
                        if (parentClassName.includes('content') || parentClassName.includes('section') || 
                            parentClassName.includes('article') || parentClassName.includes('block') || 
                            parentClassName.includes('box')) {
                            // Look for a heading within the parent
                            let heading = parent.querySelector('h1, h2, h3, h4, h5, h6');
                            if (heading) {
                                return heading.innerText || heading.textContent;
                            }
                        }
                    } else if (parentClassName.baseVal) {
                        // Handle SVG elements
                        if (parentClassName.baseVal.includes('content') || parentClassName.baseVal.includes('section') || 
                            parentClassName.baseVal.includes('article') || parentClassName.baseVal.includes('block') || 
                            parentClassName.baseVal.includes('box')) {
                            let heading = parent.querySelector('h1, h2, h3, h4, h5, h6');
                            if (heading) {
                                return heading.innerText || heading.textContent;
                            }
                        }
                    }
                }
                
                return null;
            }
        """)

        return context
    except Exception as e:
        logger.warning(f"Error finding list context: {e}")
        return None


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
        current_section = {"heading": None, "paragraphs": [], "lists": []}

        # Get all elements within the article
        elements = article_element.query_selector_all("*")

        for element in elements:
            tag_name = element.evaluate("el => el.tagName.toLowerCase()")

            if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # New section found
                if (
                    current_section["paragraphs"]
                    or current_section["heading"]
                    or current_section["lists"]
                ):
                    sections.append(current_section)

                current_section = {
                    "heading": element.inner_text(),
                    "level": int(tag_name[1]),
                    "paragraphs": [],
                    "lists": [],
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
                    list_type = "ordered" if tag_name == "ol" else "unordered"
                    current_section["lists"].append(
                        {"type": list_type, "items": list_items}
                    )

        # Add the last section
        if (
            current_section["paragraphs"]
            or current_section["heading"]
            or current_section["lists"]
        ):
            sections.append(current_section)

        # Extract full text for completeness
        full_text = article_element.inner_text()
        full_text = re.sub(r"\n\s*\n", "\n\n", full_text)

        return {
            "sections": sections,
            "text": full_text,
            "section_count": len(sections),
            "paragraph_count": sum(len(section["paragraphs"]) for section in sections),
            "list_count": sum(len(section["lists"]) for section in sections),
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


async def scrape_all_page(url: str) -> dict:
    """
    Async wrapper for the synchronous scraping function
    """
    try:
        # Use a thread pool executor to run synchronous code
        loop = asyncio.get_event_loop()
        # Construct ScrapeRequest from url with required arguments
        request = ScrapeRequest(
            url=url, timeout=30, wait_after_load=2, return_html=False
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, sync_scrape, request)
            return result

    except Exception as e:
        error_msg = str(e) if str(e) else "Unknown error occurred during scraping"
        logger.error(f"Error in scrape_page: {error_msg}")
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail=error_msg)


async def scrape_specific_page_content(request: ScrapeRequest) -> dict:
    """
    Async wrapper for the synchronous scraping function
    """
    try:
        # Use a thread pool executor to run synchronous code
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, sync_scrape, request)
            return result

    except Exception as e:
        error_msg = str(e) if str(e) else "Unknown error occurred during scraping"
        logger.error(f"Error in scrape_page: {error_msg}")
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail=error_msg)
