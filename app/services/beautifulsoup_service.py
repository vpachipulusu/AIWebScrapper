from fastapi import HTTPException
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
import requests
from bs4 import BeautifulSoup, Comment, Tag
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Any
import re

# Import models from playwright_service for consistency
from .playwright_service import ContentType, SelectorType, CustomSelector, ScrapeRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced user agents for better bot detection avoidance
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]


def get_headers() -> Dict[str, str]:
    """Generate browser-like headers"""
    import random

    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "DNT": "1",
        "Cache-Control": "max-age=0",
    }


def sync_scrape_with_beautifulsoup(request: ScrapeRequest) -> Dict[str, Any]:
    """
    Synchronous scraping function using BeautifulSoup with user-selectable content extraction
    """
    try:
        logger.info(f"Starting BeautifulSoup scrape for URL: {request.url}")
        logger.info(f"Content types to extract: {request.content_types}")

        # Make HTTP request with browser-like headers
        session = requests.Session()
        headers = get_headers()

        response = session.get(
            request.url, headers=headers, timeout=request.timeout, allow_redirects=True
        )
        response.raise_for_status()

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements for cleaner extraction
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Initialize result dictionary
        result: Dict[str, Any] = {}

        # Extract metadata if requested
        if ContentType.METADATA in request.content_types:
            logger.info("Extracting metadata...")
            result["metadata"] = extract_metadata_bs(soup, request.url)

        # Extract tables if requested
        if ContentType.TABLES in request.content_types:
            logger.info("Extracting tables...")
            tables: Optional[List[Dict[str, Any]]] = extract_structured_tables_bs(soup)
            if tables:
                result["tables"] = tables

        # Extract lists if requested
        if ContentType.LISTS in request.content_types:
            logger.info("Extracting lists...")
            lists: Optional[List[Dict[str, Any]]] = extract_structured_lists_bs(soup)
            if lists:
                result["lists"] = lists

        # Extract articles if requested
        if ContentType.ARTICLES in request.content_types:
            logger.info("Extracting articles...")
            article: Optional[Dict[str, Any]] = extract_article_content_bs(soup)
            if article and (article.get("sections") or article.get("text")):
                result["article"] = article

        # Extract key points if requested
        if ContentType.KEY_POINTS in request.content_types:
            logger.info("Extracting key points...")
            key_points: Optional[List[str]] = extract_key_points_bs(soup)
            if key_points:
                result["key_points"] = key_points

        # Extract explanations if requested
        if ContentType.EXPLANATIONS in request.content_types:
            logger.info("Extracting explanations...")
            explanations = extract_explanations_bs(soup)
            if explanations:
                result["explanations"] = explanations

        # Extract links if requested
        if ContentType.LINKS in request.content_types:
            logger.info("Extracting links...")
            links = extract_structured_links_bs(soup, request.url)
            if links:
                result["links"] = links[:20]  # Limit to top 20 links

        # Extract custom content if requested
        if ContentType.CUSTOM in request.content_types and request.custom_selectors:
            logger.info("Extracting custom content...")
            custom_content = extract_custom_content_bs(soup, request.custom_selectors)
            if custom_content:
                result["custom"] = custom_content

        # Return HTML for analysis if requested
        if request.return_html:
            logger.info("Extracting HTML for analysis...")
            result["html"] = extract_html_for_analysis_bs(soup)

        logger.info("BeautifulSoup scraping completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error in sync_scrape_with_beautifulsoup: {str(e)}")
        logger.error(traceback.format_exc())
        error_msg = str(e) if str(e) else "Unknown error in BeautifulSoup scraping"
        raise Exception(error_msg)


def extract_metadata_bs(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Extract metadata from the page using BeautifulSoup"""
    try:
        # Extract title
        title_tag = soup.find("title")
        title: str = title_tag.get_text().strip() if title_tag else "No title found"

        # Extract description
        description: Optional[str] = None
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and isinstance(desc_tag, Tag) and desc_tag.get("content"):
            content_value = desc_tag.get("content")
            if isinstance(content_value, str):
                description = content_value.strip()

        return {
            "url": url,
            "title": title,
            "description": description,
        }
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return {"error": str(e)}


def extract_structured_tables_bs(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract structured data from tables using BeautifulSoup"""
    tables: List[Dict[str, Any]] = []

    try:
        table_elements = soup.find_all("table")

        for i, table in enumerate(table_elements):
            try:
                if not isinstance(table, Tag):
                    continue

                rows = table.find_all("tr")
                if len(rows) < 2:  # Skip tables with less than 2 rows
                    continue

                # Extract caption if available
                caption_tag = table.find("caption")
                caption: Optional[str] = (
                    caption_tag.get_text().strip() if caption_tag else None
                )

                # Extract headers
                headers: List[str] = []
                header_row_element = table.find("thead")
                header_row = None
                if header_row_element and isinstance(header_row_element, Tag):
                    header_row = header_row_element.find("tr")
                else:
                    header_row = rows[0] if rows else None

                if header_row and isinstance(header_row, Tag):
                    header_cells = header_row.find_all(["th", "td"])
                    for cell in header_cells:
                        headers.append(
                            cell.get_text().strip() or f"Column {len(headers) + 1}"
                        )

                # Extract rows
                table_rows: List[List[Dict[str, Any]]] = []
                tbody = table.find("tbody")
                if tbody and isinstance(tbody, Tag):
                    body_rows = tbody.find_all("tr")
                else:
                    body_rows = rows

                # Skip header row if it was in the body
                start_idx = 1 if not table.find("thead") and header_row in rows else 0

                for row in body_rows[start_idx:]:
                    if not isinstance(row, Tag):
                        continue

                    row_data: List[Dict[str, Any]] = []
                    cells = row.find_all(["td", "th"])

                    for cell in cells:
                        if not isinstance(cell, Tag):
                            continue

                        # Check for rowspan and colspan
                        rowspan_attr = cell.get("rowspan", "1")
                        colspan_attr = cell.get("colspan", "1")

                        rowspan: int = (
                            int(rowspan_attr) if isinstance(rowspan_attr, str) else 1
                        )
                        colspan: int = (
                            int(colspan_attr) if isinstance(colspan_attr, str) else 1
                        )

                        cell_text: str = cell.get_text().strip()
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


def extract_structured_lists_bs(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract structured data from lists using BeautifulSoup"""
    lists: List[Dict[str, Any]] = []

    try:
        list_elements = soup.find_all(["ul", "ol"])

        for i, list_element in enumerate(list_elements):
            try:
                if not isinstance(list_element, Tag):
                    continue

                items = list_element.find_all(
                    "li", recursive=False
                )  # Direct children only
                if len(items) < 3:  # Skip lists with less than 3 items
                    continue

                # Determine list type
                list_type: str = "ordered" if list_element.name == "ol" else "unordered"

                # Extract list items
                list_items: List[str] = []
                for item in items:
                    item_text: str = item.get_text().strip()
                    if (
                        item_text and len(item_text) > 2
                    ):  # Skip empty or very short items
                        list_items.append(item_text)

                if list_items:  # Only add lists with items
                    # Try to find context for the list
                    context: Optional[str] = find_list_context_bs(list_element)

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


def find_list_context_bs(list_element: Tag) -> Optional[str]:
    """Try to find context or heading for a list using BeautifulSoup"""
    try:
        # Look for previous sibling that might be a heading
        previous = list_element.find_previous_sibling()
        if (
            previous
            and hasattr(previous, "name")
            and previous.name in ["h1", "h2", "h3", "h4", "h5", "h6"]
        ):
            return previous.get_text().strip()

        # Look for parent element that might provide context
        parent = list_element.parent
        if parent and isinstance(parent, Tag):
            # Check if parent has a class that suggests it's a content container
            parent_classes_attr = parent.get("class")
            parent_classes: List[str] = (
                parent_classes_attr if isinstance(parent_classes_attr, list) else []
            )
            if any(
                cls in ["content", "section", "article", "block", "box"]
                for cls in parent_classes
            ):
                # Look for a heading within the parent
                heading = parent.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                if heading:
                    return heading.get_text().strip()

        return None
    except Exception as e:
        logger.warning(f"Error finding list context: {e}")
        return None


def extract_article_content_bs(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract structured article content using BeautifulSoup"""
    try:
        # Try to find article using common selectors
        article_selectors: List[str] = [
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

        article_element: Optional[Tag] = None
        for selector in article_selectors:
            found_element = soup.select_one(selector)
            if found_element and isinstance(found_element, Tag):
                article_element = found_element
                break

        # If no article found, use body
        if not article_element:
            body_element = soup.find("body")
            if body_element and isinstance(body_element, Tag):
                article_element = body_element

        if not article_element:
            return {"sections": [], "text": ""}

        # Extract structured content
        sections: List[Dict[str, Any]] = []
        current_section: Dict[str, Any] = {
            "heading": None,
            "paragraphs": [],
            "lists": [],
        }

        # Get all elements within the article
        elements = article_element.find_all(True)

        for element in elements:
            if not isinstance(element, Tag) or not hasattr(element, "name"):
                continue

            tag_name: str = element.name.lower()

            if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # New section found
                if (
                    current_section["paragraphs"]
                    or current_section["heading"]
                    or current_section["lists"]
                ):
                    sections.append(current_section)

                current_section = {
                    "heading": element.get_text().strip(),
                    "level": int(tag_name[1]),
                    "paragraphs": [],
                    "lists": [],
                }
            elif tag_name in ["p", "div"]:
                # Paragraph content
                text: str = element.get_text().strip()
                if text and len(text) > 30 and not is_noisy_text_bs(text):
                    current_section["paragraphs"].append(text)
            elif tag_name in ["ul", "ol"]:
                # List content
                list_items: List[str] = [
                    li.get_text().strip() for li in element.find_all("li")
                ]
                list_items = [item for item in list_items if item and len(item) > 10]

                if list_items:
                    list_type: str = "ordered" if tag_name == "ol" else "unordered"
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
        full_text = article_element.get_text()
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


def extract_key_points_bs(soup: BeautifulSoup) -> List[str]:
    """Extract key points or important information using BeautifulSoup"""
    key_points: List[str] = []

    try:
        # Look for key points using common selectors
        key_point_selectors: List[str] = [
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
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) > 10 and not is_noisy_text_bs(text):
                    key_points.append(text)

        # Remove duplicates
        key_points = list(set(key_points))

    except Exception as e:
        logger.error(f"Error extracting key points: {e}")

    return key_points


def extract_explanations_bs(soup: BeautifulSoup) -> List[str]:
    """Extract explanations or descriptive text using BeautifulSoup"""
    explanations: List[str] = []

    try:
        # Look for explanations using common selectors
        explanation_selectors: List[str] = [
            ".explanation",
            ".description",
            ".instruction",
            ".guide",
            ".tutorial",
            ".help-text",
        ]

        for selector in explanation_selectors:
            elements = soup.select(selector)
            for element in elements:
                text: str = element.get_text().strip()
                if text and len(text) > 20 and not is_noisy_text_bs(text):
                    explanations.append(text)

    except Exception as e:
        logger.error(f"Error extracting explanations: {e}")

    return explanations


def extract_structured_links_bs(
    soup: BeautifulSoup, base_url: str
) -> List[Dict[str, Any]]:
    """Extract structured link information using BeautifulSoup"""
    links: List[Dict[str, Any]] = []

    try:
        link_elements = soup.find_all("a", href=True)

        for link in link_elements:
            try:
                if not isinstance(link, Tag):
                    continue

                href_attr = link.get("href")
                if not href_attr or not isinstance(href_attr, str):
                    continue

                href: str = href_attr
                text: str = link.get_text().strip()

                # Skip navigation and noisy links
                if not text or is_noisy_text_bs(text):
                    continue

                # Make relative URLs absolute
                if href.startswith("/"):
                    href = urljoin(base_url, href)

                # Only include HTTP/HTTPS links
                if href.startswith(("http://", "https://")):
                    parsed_base = urlparse(base_url)
                    parsed_href = urlparse(href)
                    is_external = parsed_base.netloc != parsed_href.netloc

                    links.append(
                        {
                            "text": text,
                            "url": href,
                            "is_external": is_external,
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


def extract_custom_content_bs(
    soup: BeautifulSoup, custom_selectors: List[CustomSelector]
) -> Dict[str, Any]:
    """Extract content using custom CSS selectors using BeautifulSoup"""
    custom_content: Dict[str, Any] = {}

    for selector_def in custom_selectors:
        try:
            elements: List[Tag] = []

            if selector_def.selector_type == SelectorType.CSS:
                found_elements = soup.select(selector_def.selector)
                elements = [elem for elem in found_elements if isinstance(elem, Tag)]
            elif selector_def.selector_type == SelectorType.XPATH:
                # BeautifulSoup doesn't support XPath directly, skip or convert to CSS
                logger.warning(
                    f"XPath not supported in BeautifulSoup for selector '{selector_def.name}', skipping"
                )
                custom_content[selector_def.name] = (
                    "XPath not supported in BeautifulSoup"
                )
                continue

            if not elements:
                custom_content[selector_def.name] = []
                continue

            extracted_data: List[Any] = []

            for element in elements:
                try:
                    if selector_def.extract == "text":
                        text: str = element.get_text().strip()
                        extracted_data.append(text)
                    elif selector_def.extract == "html":
                        html_content: str = str(element)
                        extracted_data.append(html_content)
                    elif selector_def.extract == "attribute" and selector_def.attribute:
                        attr_value = element.get(selector_def.attribute, "")
                        if isinstance(attr_value, str):
                            extracted_data.append(attr_value)
                    elif selector_def.extract == "all":
                        # Extract both text and HTML
                        element_text: str = element.get_text().strip()
                        element_html: str = str(element)
                        extracted_data.append(
                            {"text": element_text, "html": element_html}
                        )
                    else:
                        # Default to text
                        text = element.get_text().strip()
                        extracted_data.append(text)
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


def extract_html_for_analysis_bs(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract HTML structure for analysis using BeautifulSoup"""
    try:
        # Get the main content areas
        body = soup.find("body")
        body_html: str = str(body) if body else str(soup)

        # Find all unique CSS classes and IDs for analysis
        all_elements = soup.find_all(True)
        classes: set = set()
        ids: set = set()

        for element in all_elements:
            if isinstance(element, Tag):
                element_classes_attr = element.get("class")
                element_classes: List[str] = (
                    element_classes_attr
                    if isinstance(element_classes_attr, list)
                    else []
                )
                for cls in element_classes:
                    if cls:
                        classes.add(cls)

                element_id = element.get("id")
                if element_id and isinstance(element_id, str):
                    ids.add(element_id)

        # Get common content containers
        content_containers = {
            "articles": len(soup.find_all("article")),
            "main_content": len(
                soup.select("main, [role='main'], .main, #main, .content, #content")
            ),
            "headers": len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])),
            "tables": len(soup.find_all("table")),
            "lists": len(soup.find_all(["ul", "ol"])),
        }

        return {
            "structure": content_containers,
            "classes": list(classes),
            "ids": list(ids),
            "body_html": body_html[:5000] + "..."
            if len(body_html) > 5000
            else body_html,  # Limit size
        }
    except Exception as e:
        logger.error(f"Error extracting HTML for analysis: {e}")
        return {"error": str(e)}


def is_noisy_text_bs(text: str) -> bool:
    """Check if text is likely noise (navigation, ads, etc.)"""
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


async def scrape_with_beautifulsoup(request: ScrapeRequest) -> dict:
    """
    Async wrapper for the synchronous BeautifulSoup scraping function
    """
    try:
        # Use a thread pool executor to run synchronous code
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor, sync_scrape_with_beautifulsoup, request
            )
            return result

    except Exception as e:
        error_msg = (
            str(e) if str(e) else "Unknown error occurred during BeautifulSoup scraping"
        )
        logger.error(f"Error in scrape_with_beautifulsoup: {error_msg}")
        logger.error(traceback.format_exc())

        raise HTTPException(status_code=500, detail=error_msg)
