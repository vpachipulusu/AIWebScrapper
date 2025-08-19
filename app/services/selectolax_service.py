from fastapi import HTTPException
import httpx
from selectolax.parser import HTMLParser
import asyncio
import logging
from urllib.parse import urljoin
import re
from typing import Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def structured_scrape(url: str) -> dict:
    """
    Structured scraping that extracts either table data or article content
    in a well-organized JSON format
    """
    try:
        logger.info(f"Starting structured scrape for URL: {url}")

        # Set headers to mimic a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        async with httpx.AsyncClient(
            headers=headers, timeout=30.0, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Parse HTML with selectolax
            parser = HTMLParser(response.text)

            # Extract metadata
            title = get_text(parser.css_first("title"))
            description = get_meta_content(parser, "description")
            author = get_meta_content(parser, "author") or get_meta_content(
                parser, "article:author"
            )
            published_time = get_meta_content(parser, "article:published_time")

            # Check if page has tables with meaningful data
            tables = find_meaningful_tables(parser)
            content_type = None
            content = None

            if tables:
                logger.info("Table(s) found on page, extracting table data")
                content_type = "table"
                content = extract_structured_tables(tables)
            else:
                logger.info("No tables found, extracting article content")
                content_type = "article"
                content = extract_structured_article(parser)

            # Extract links
            links = extract_structured_links(parser, url)

            logger.info(
                f"Structured scraping completed successfully, found: {content_type}"
            )

            # Build structured response
            return {
                "metadata": {
                    "url": url,
                    "title": title,
                    "description": description,
                    "author": author,
                    "published_time": published_time,
                    "content_type": content_type,
                },
                "content": content,
                "links": links,
            }

    except Exception as e:
        logger.error(f"Error in structured_scrape: {str(e)}")
        error_msg = str(e) if str(e) else "Unknown error occurred during scraping"
        raise HTTPException(status_code=500, detail=error_msg)


def find_meaningful_tables(parser):
    """
    Find tables that likely contain meaningful data (not layout tables)
    """
    all_tables = parser.css("table")
    meaningful_tables = []

    for table in all_tables:
        # Skip tables that are likely for layout
        if is_layout_table(table):
            continue

        # Check if table has meaningful content
        rows = table.css("tr")
        if len(rows) >= 2:  # At least header row and one data row
            meaningful_tables.append(table)

    return meaningful_tables


def is_layout_table(table):
    """
    Determine if a table is likely used for layout rather than data
    """
    # Check for common layout table attributes
    if table.attributes.get("role") == "presentation":
        return True

    if table.attributes.get("class") and any(
        "layout" in cls.lower() for cls in table.attributes["class"].split()
    ):
        return True

    # Check if table has very few rows or columns
    rows = table.css("tr")
    if len(rows) < 2:
        return True

    # Check if table has mostly empty cells
    empty_cell_count = 0
    total_cell_count = 0

    for row in rows:
        cells = row.css("td, th")
        total_cell_count += len(cells)
        for cell in cells:
            text = get_text(cell)
            if not text or len(text.strip()) < 3:
                empty_cell_count += 1

    if total_cell_count > 0 and empty_cell_count / total_cell_count > 0.7:
        return True

    return False


def get_text(node) -> Optional[str]:
    """Safely extract text from a node"""
    return node.text(separator=" ", strip=True) if node else None


def get_meta_content(parser, name: str) -> Optional[str]:
    """Extract content from meta tag"""
    meta = parser.css_first(f'meta[name="{name}"], meta[property="{name}"]')
    return meta.attributes.get("content") if meta else None


def extract_structured_tables(tables) -> List[Dict]:
    """
    Extract structured data from tables
    """
    structured_tables = []

    for i, table in enumerate(tables):
        try:
            # Extract caption if available
            caption = get_text(table.css_first("caption"))

            # Extract headers
            headers = []
            header_rows = (
                table.css("thead tr") if table.css_first("thead") else table.css("tr")
            )

            if header_rows:
                for cell in header_rows[0].css("th, td"):
                    headers.append(get_text(cell) or f"Column {len(headers) + 1}")

            # Extract rows
            rows = []
            body_rows = (
                table.css("tbody tr") if table.css_first("tbody") else table.css("tr")
            )

            # Skip header row if it was in the body
            start_idx = 1 if not table.css_first("thead") and header_rows else 0

            for row in body_rows[start_idx:]:
                row_data = []
                for cell in row.css("td, th"):
                    # Check for rowspan and colspan
                    rowspan = int(cell.attributes.get("rowspan", 1))
                    colspan = int(cell.attributes.get("colspan", 1))

                    cell_text = get_text(cell) or ""
                    row_data.append(
                        {"value": cell_text, "rowspan": rowspan, "colspan": colspan}
                    )

                if row_data:
                    rows.append(row_data)

            structured_tables.append(
                {
                    "table_index": i,
                    "caption": caption,
                    "headers": headers,
                    "rows": rows,
                    "summary": f"Table with {len(headers)} columns and {len(rows)} rows",
                }
            )
        except Exception as e:
            logger.error(f"Error processing table {i}: {e}")
            continue

    return structured_tables


def extract_structured_article(parser) -> Dict:
    """
    Extract structured article content, focusing on main content only
    """
    # Remove unwanted elements first
    remove_unwanted_elements(parser)

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
        ".main-content",
        ".entry-content",
    ]

    article_elem = None
    for selector in article_selectors:
        article_elem = parser.css_first(selector)
        if article_elem:
            break

    # If no article found, use body but remove common noise elements
    if not article_elem:
        article_elem = parser.css_first("body")

    if not article_elem:
        return {"sections": [], "text": ""}

    # Extract structured content
    sections = []
    current_section = {"heading": None, "paragraphs": []}

    # Process all elements in the article
    for elem in article_elem.iter():
        if elem.tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # New section found
            if current_section["paragraphs"] or current_section["heading"]:
                sections.append(current_section)

            current_section = {
                "heading": get_text(elem),
                "level": int(elem.tag[1]),
                "paragraphs": [],
            }
        elif elem.tag in ["p", "div"] and get_text(elem):
            # Paragraph content - filter out short/noisy text
            text = get_text(elem)
            if (
                text and len(text.strip()) > 20 and not is_noisy_text(text)
            ):  # Increased minimum length
                current_section["paragraphs"].append(text)
        elif elem.tag in ["ul", "ol"]:
            # List content
            list_items = [
                get_text(li)
                for li in elem.css("li")
                if get_text(li) and len(get_text(li)) > 10
            ]
            if list_items:
                current_section["paragraphs"].append(
                    {
                        "list_type": "ordered" if elem.tag == "ol" else "unordered",
                        "items": list_items,
                    }
                )

    # Add the last section
    if current_section["paragraphs"] or current_section["heading"]:
        sections.append(current_section)

    # Filter out sections with little content
    sections = [s for s in sections if len(s["paragraphs"]) > 0 or s["heading"]]

    # Extract full text for completeness
    full_text = article_elem.text(separator="\n", strip=True)
    full_text = re.sub(r"\n\s*\n", "\n\n", full_text)

    return {
        "sections": sections,
        "text": full_text,
        "section_count": len(sections),
        "paragraph_count": sum(len(section["paragraphs"]) for section in sections),
    }


def remove_unwanted_elements(parser):
    """Remove unwanted elements from the HTML to reduce noise"""
    unwanted_selectors = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        ".ad",
        ".ads",
        ".advertisement",
        ".social",
        ".share",
        ".comments",
        ".comment",
        ".navbar",
        ".menu",
        ".sidebar",
        ".popup",
        ".modal",
        ".overlay",
        ".cookie-consent",
        ".newsletter",
        ".subscription",
        ".promo",
        ".banner",
        ".navigation",
        ".nav",
        ".header",
        ".footer",
        ".site-header",
        ".site-footer",
        ".main-nav",
    ]

    for selector in unwanted_selectors:
        for element in parser.css(selector):
            element.decompose()


def is_noisy_text(text):
    """Check if text is likely noise (navigation, ads, etc.)"""
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
    ]

    # Check if text contains navigation phrases
    for phrase in navigation_phrases:
        if phrase in text:
            return True

    # Check if text is very short or looks like a button/link
    if len(text) < 25 and (text.isupper() or ">" in text or "→" in text or "›" in text):
        return True

    return False


def extract_structured_links(parser, base_url: str) -> List[Dict]:
    """
    Extract structured link information
    """
    links = []

    for link in parser.css("a[href]"):
        href = link.attributes.get("href", "")
        text = get_text(link)

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
                    "text_length": len(text) if text else 0,
                }
            )

    # Sort by text length (longer texts are usually more meaningful)
    links.sort(key=lambda x: x["text_length"], reverse=True)

    return links


# For backward compatibility
async def scrape_page(url: str) -> dict:
    return await structured_scrape(url)
