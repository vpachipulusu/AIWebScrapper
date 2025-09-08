"""
Base service for web scraping with common functionality shared between
BeautifulSoup and Playwright services.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .data_processor import convert_numpy_types, process_web_data

logger = logging.getLogger(__name__)


class BaseScrapingService(ABC):
    """
    Abstract base class for web scraping services.
    Contains common functionality for data extraction and processing.
    """

    def __init__(self):
        self.logger = logger

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def fetch_page_content(self, url: str, **kwargs) -> str:
        """Fetch the raw HTML content of a page."""
        pass

    @abstractmethod
    def extract_raw_data(self, content: str, url: str, **kwargs) -> Dict[str, Any]:
        """Extract raw data from HTML content."""
        pass

    # Common data extraction methods
    def extract_tables_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract and process tables using BeautifulSoup."""
        tables = []

        for i, table in enumerate(soup.find_all("table")):
            try:
                # Skip tables with minimal content
                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue

                # Extract caption
                caption = None
                caption_elem = table.find("caption")
                if caption_elem:
                    caption = caption_elem.get_text(strip=True)

                # Extract headers
                headers = []
                header_row = table.find("thead")
                if header_row:
                    header_row = header_row.find("tr")
                else:
                    header_row = rows[0]

                for cell in header_row.find_all(["th", "td"]):
                    headers.append(
                        cell.get_text(strip=True) or f"Column {len(headers) + 1}"
                    )

                # Extract data rows
                table_data = []
                data_rows = table.find_all("tbody")
                if data_rows:
                    data_rows = data_rows[0].find_all("tr")
                else:
                    # Skip header row if no tbody
                    data_rows = rows[1:] if not table.find("thead") else rows

                for row in data_rows:
                    row_data = []
                    for cell in row.find_all(["td", "th"]):
                        row_data.append(cell.get_text(strip=True))

                    # Ensure row matches header length
                    while len(row_data) < len(headers):
                        row_data.append("")
                    row_data = row_data[: len(headers)]

                    if any(row_data):  # Only add non-empty rows
                        table_data.append(row_data)

                if table_data:
                    # Create pandas DataFrame for analysis
                    df = pd.DataFrame(table_data, columns=headers)
                    df = self.clean_dataframe(df)

                    table_info = {
                        "table_index": i,
                        "caption": caption,
                        "headers": headers,
                        "dataframe": df.to_dict("records"),
                        "shape": df.shape,
                        "statistics": self.generate_table_statistics(df),
                        "summary": f"Table with {len(headers)} columns and {len(table_data)} rows",
                    }
                    tables.append(table_info)

            except Exception as e:
                logger.warning(f"Error processing table {i}: {e}")
                continue

        return tables

    def extract_lists_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract and process lists using BeautifulSoup."""
        lists = []

        for i, list_elem in enumerate(soup.find_all(["ul", "ol"])):
            try:
                # Skip lists with minimal content
                items = list_elem.find_all("li", recursive=False)
                if len(items) < 3:
                    continue

                # Extract list items
                list_items = []
                for item in items:
                    text = item.get_text(strip=True)
                    if text and len(text) > 2:
                        list_items.append(text)

                if list_items:
                    # Determine list type
                    list_type = "ordered" if list_elem.name == "ol" else "unordered"

                    # Find context/heading
                    context = self.find_list_context(list_elem)

                    # Create pandas DataFrame for analysis
                    list_df = pd.DataFrame(
                        {
                            "item": list_items,
                            "item_number": range(1, len(list_items) + 1),
                            "length": [len(item) for item in list_items],
                            "word_count": [len(item.split()) for item in list_items],
                        }
                    )

                    list_info = {
                        "list_index": i,
                        "type": list_type,
                        "context": context,
                        "items": list_items,
                        "item_count": len(list_items),
                        "dataframe": list_df.to_dict("records"),
                        "statistics": self.generate_list_statistics(list_df),
                        "patterns": self.analyze_list_patterns(list_items),
                    }
                    lists.append(list_info)

            except Exception as e:
                logger.warning(f"Error processing list {i}: {e}")
                continue

        return lists

    def extract_text_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract and analyze text content."""
        # Extract different text elements
        headings = []
        for tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            for heading in soup.find_all(tag):
                text = heading.get_text(strip=True)
                if text:
                    headings.append({"level": int(tag[1]), "text": text, "tag": tag})

        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # Filter out short paragraphs
                paragraphs.append(text)

        # Get main text content
        body_text = soup.get_text(separator=" ", strip=True)

        return {
            "headings": headings,
            "paragraphs": paragraphs,
            "body_text": body_text,
            "heading_count": len(headings),
            "paragraph_count": len(paragraphs),
            "word_count": len(body_text.split()) if body_text else 0,
        }

    def extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract page metadata."""
        metadata = {
            "url": url,
            "domain": urlparse(url).netloc,
            "title": "",
            "description": "",
            "keywords": "",
            "author": "",
            "language": "",
            "charset": "",
        }

        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Extract meta tags
        meta_tags = soup.find_all("meta")
        for meta in meta_tags:
            name = meta.get("name", "").lower()
            property_name = meta.get("property", "").lower()
            content = meta.get("content", "")

            if name == "description" or property_name == "og:description":
                metadata["description"] = content
            elif name == "keywords":
                metadata["keywords"] = content
            elif name == "author":
                metadata["author"] = content
            elif name == "language" or property_name == "og:locale":
                metadata["language"] = content
            elif meta.get("charset"):
                metadata["charset"] = meta.get("charset")

        return metadata

    # Common utility methods
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize DataFrame data."""
        # Remove completely empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Clean text data
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("", np.nan)

        # Attempt to convert numeric columns
        for col in df.columns:
            # Try to convert to numeric if possible
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if not numeric_series.isnull().all():
                df[col] = numeric_series

        return df

    def generate_table_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics for a table DataFrame."""
        if df.empty:
            return {}

        stats = {
            "shape": df.shape,
            "column_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        # Numeric analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
            if len(numeric_cols) > 1:
                stats["correlations"] = df[numeric_cols].corr().to_dict()

        # Text analysis
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if text_cols:
            stats["text_summary"] = {}
            for col in text_cols:
                valid_values = df[col].dropna()
                if not valid_values.empty:
                    stats["text_summary"][col] = {
                        "unique_count": valid_values.nunique(),
                        "top_values": valid_values.value_counts().head(3).to_dict(),
                        "avg_length": valid_values.astype(str).str.len().mean(),
                    }

        # Missing data
        stats["missing_data"] = {
            "total_missing": df.isnull().sum().sum(),
            "missing_by_column": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        }

        return convert_numpy_types(stats)

    def generate_list_statistics(self, list_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics for a list DataFrame."""
        if list_df.empty:
            return {}

        stats = {
            "total_items": len(list_df),
            "avg_length": list_df["length"].mean(),
            "avg_word_count": list_df["word_count"].mean(),
            "length_distribution": list_df["length"].describe().to_dict(),
            "word_count_distribution": list_df["word_count"].describe().to_dict(),
        }

        if not list_df.empty:
            stats["longest_item"] = list_df.loc[list_df["length"].idxmax(), "item"]
            stats["shortest_item"] = list_df.loc[list_df["length"].idxmin(), "item"]

        return convert_numpy_types(stats)

    def analyze_list_patterns(self, items: List[str]) -> Dict[str, Any]:
        """Analyze patterns in list items."""
        if not items:
            return {}

        df = pd.DataFrame({"item": items})

        patterns = {
            "starts_with_number": df["item"].str.match(r"^\d+").sum(),
            "starts_with_bullet": df["item"].str.match(r"^[•·▪▫◦‣⁃]").sum(),
            "contains_colon": df["item"].str.contains(":").sum(),
            "contains_dash": df["item"].str.contains("-").sum(),
            "contains_parentheses": df["item"].str.contains(r"\(.*\)").sum(),
            "contains_urls": df["item"].str.contains(r"http[s]?://").sum(),
            "all_caps_words": df["item"].str.findall(r"\b[A-Z]{2,}\b").apply(len).sum(),
        }

        # Length analysis
        lengths = df["item"].str.len()
        patterns.update(
            {
                "length_variance": lengths.var(),
                "consistent_length": (lengths.max() - lengths.min()) < 10,
                "similar_structure": self.check_similar_structure(items),
            }
        )

        return convert_numpy_types(patterns)

    def check_similar_structure(self, items: List[str]) -> bool:
        """Check if list items have similar structure."""
        if len(items) < 3:
            return False

        word_counts = [len(item.split()) for item in items]
        avg_words = sum(word_counts) / len(word_counts)
        variance = sum((x - avg_words) ** 2 for x in word_counts) / len(word_counts)

        return variance < 2.0

    def find_list_context(self, list_element) -> Optional[str]:
        """Find context or heading for a list element."""
        try:
            # Look for previous sibling headings
            current = list_element.previous_sibling
            while current:
                if hasattr(current, "name") and current.name in [
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                ]:
                    return current.get_text(strip=True)
                elif hasattr(current, "name") and current.name in ["p", "div"]:
                    text = current.get_text(strip=True)
                    if text and len(text) < 100:  # Short descriptive text
                        return text
                current = current.previous_sibling
        except Exception:
            pass
        return None

    # Main processing methods
    def scrape_and_process(self, url: str, **kwargs) -> Dict[str, Any]:
        """Main method to scrape and process a URL."""
        try:
            # Fetch content
            content = self.fetch_page_content(url, **kwargs)

            # Extract raw data
            raw_data = self.extract_raw_data(content, url, **kwargs)

            # Process with pandas for enhanced analysis
            enhanced_data = process_web_data(raw_data, url)

            return enhanced_data

        except Exception as e:
            logger.error(f"Error in scrape_and_process for {url}: {e}")
            return {"url": url, "error": str(e), "status": "failed"}

    def extract_structured_data(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract all structured data from a BeautifulSoup object."""
        structured_data = {}

        # Extract tables
        tables = self.extract_tables_from_soup(soup)
        if tables:
            structured_data["tables"] = tables

        # Extract lists
        lists = self.extract_lists_from_soup(soup)
        if lists:
            structured_data["lists"] = lists

        # Extract text content
        text_content = self.extract_text_content(soup)
        structured_data.update(text_content)

        # Extract metadata
        metadata = self.extract_metadata(soup, url)
        structured_data["metadata"] = metadata

        return structured_data
