import random
import time
import re
from typing import Any, Dict, List, Optional
import requests
from lxml import html
from requests.exceptions import RequestException
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import json
import logging
from .data_processor import process_web_data

logger = logging.getLogger(__name__)

# Rotating user agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Linux; Android 13; SM-S901B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]

# Browser-like headers template
HEADERS_TEMPLATE = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def generate_headers() -> dict:
    headers = HEADERS_TEMPLATE.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)
    return headers


def detect_anti_bot(response: requests.Response) -> bool:
    if response.status_code in (403, 429, 503):
        return True
    patterns = [
        r"captcha",
        r"robot",
        r"access denied",
        r"cloudflare",
        r"distil",
        r"incapsula",
        r"perimeterx",
        r"blocked",
        r"security check",
    ]
    text = response.text[:5000].lower()
    return any(re.search(pattern, text) for pattern in patterns)


def extract_tables_to_dataframe(tree) -> List[Dict[str, Any]]:
    """Extract HTML tables and convert to pandas DataFrames."""
    tables_data = []
    tables = tree.xpath("//table")

    for i, table in enumerate(tables):
        try:
            # Convert table to list of lists
            rows = []
            header_row = table.xpath(".//thead//tr | .//tr[1]")

            if header_row:
                headers = [
                    th.text_content().strip()
                    for th in header_row[0].xpath(".//th | .//td")
                ]
                if headers:
                    rows.append(headers)

            # Get data rows
            data_rows = table.xpath(".//tbody//tr | .//tr[position()>1]")
            for row in data_rows:
                cells = [td.text_content().strip() for td in row.xpath(".//td")]
                if cells:
                    rows.append(cells)

            if len(rows) > 1:  # Has header and at least one data row
                # Create DataFrame
                df = pd.DataFrame(
                    rows[1:], columns=rows[0] if len(rows[0]) == len(rows[1]) else None
                )

                # Clean the DataFrame
                df = clean_dataframe(df)

                tables_data.append(
                    {
                        "table_index": i,
                        "dataframe": df.to_dict("records"),
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "summary": df.describe(include="all").to_dict()
                        if not df.empty
                        else {},
                    }
                )
        except Exception as e:
            logger.info(f"Could not process table {i}: {e}")
            continue

    return tables_data


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and format a pandas DataFrame."""
    # Remove completely empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["", "None", "nan"], np.nan)

    # Try to convert numeric columns
    for col in df.columns:
        if df[col].dtype == "object":
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if numeric_series.notna().sum() > len(df) * 0.7:  # If 70% can be converted
                df[col] = numeric_series

    # Reset index
    df = df.reset_index(drop=True)

    return df


def extract_structured_lists(tree) -> List[Dict[str, Any]]:
    """Extract structured lists and convert to DataFrames."""
    lists_data = []

    # Extract ordered and unordered lists
    lists = tree.xpath("//ol | //ul")

    for i, list_elem in enumerate(lists):
        try:
            items = [
                li.text_content().strip()
                for li in list_elem.xpath(".//li")
                if li.text_content().strip()
            ]

            if len(items) >= 3:  # Only process lists with 3+ items
                # Create DataFrame from list items
                df = pd.DataFrame(
                    {
                        "item": items,
                        "item_number": range(1, len(items) + 1),
                        "length": [len(item) for item in items],
                    }
                )

                lists_data.append(
                    {
                        "list_index": i,
                        "list_type": list_elem.tag,
                        "items_count": len(items),
                        "dataframe": df.to_dict("records"),
                        "summary": {
                            "avg_length": df["length"].mean(),
                            "total_items": len(items),
                            "longest_item": df.loc[df["length"].idxmax(), "item"]
                            if not df.empty
                            else "",
                        },
                    }
                )
        except Exception:
            continue

    return lists_data


def analyze_text_content(content: str) -> Dict[str, Any]:
    """Analyze text content using pandas for statistical analysis."""
    if not content:
        return {}

    # Split into words and sentences
    words = re.findall(r"\b\w+\b", content.lower())
    sentences = re.split(r"[.!?]+", content)

    # Create DataFrames for analysis
    words_df = pd.DataFrame({"word": words})
    sentences_df = pd.DataFrame(
        {
            "sentence": [s.strip() for s in sentences if s.strip()],
            "length": [len(s.strip()) for s in sentences if s.strip()],
        }
    )

    # Word frequency analysis
    word_counts = words_df["word"].value_counts().head(20)

    # Sentence length analysis
    sentence_stats = (
        sentences_df["length"].describe() if not sentences_df.empty else pd.Series()
    )

    return {
        "total_words": len(words),
        "unique_words": len(words_df["word"].unique()),
        "total_sentences": len(sentences_df),
        "top_words": word_counts.to_dict(),
        "sentence_stats": sentence_stats.to_dict(),
        "avg_words_per_sentence": len(words) / len(sentences_df)
        if len(sentences_df) > 0
        else 0,
        "readability_score": calculate_readability_score(words, sentences_df),
    }


def calculate_readability_score(words: List[str], sentences_df: pd.DataFrame) -> float:
    """Calculate a simple readability score."""
    if len(sentences_df) == 0 or len(words) == 0:
        return 0.0

    avg_sentence_length = len(words) / len(sentences_df)

    # Count complex words (3+ syllables, simplified)
    complex_words = [w for w in words if len(w) > 6]
    complex_word_ratio = len(complex_words) / len(words) if words else 0

    # Simplified Flesch-Kincaid grade level approximation
    score = 0.39 * avg_sentence_length + 11.8 * complex_word_ratio - 15.59
    return max(0, min(20, score))  # Clamp between 0 and 20


def parse_page(content: bytes, url: str) -> Dict[str, Any]:
    tree = html.fromstring(content)
    title = tree.findtext(".//title", "").strip()

    headers = [
        elem.text_content().strip()
        for tag in ["h1", "h2", "h3"]
        for elem in tree.xpath(f"//{tag}")[:10]
        if elem.text_content().strip()
    ]

    paragraphs = [
        p.text_content().strip()
        for p in tree.xpath("//p")[:30]
        if len(p.text_content().strip()) > 10
    ]

    meta_desc = tree.xpath("//meta[@name='description']/@content")
    meta_description = meta_desc[0] if meta_desc else ""

    body_text = ""
    candidates = []
    article = tree.xpath("//article")
    main = tree.xpath("//main")
    body = tree.xpath("//body")
    if article:
        candidates.append(article[0].text_content())
    if main:
        candidates.append(main[0].text_content())
    if body:
        candidates.append(body[0].text_content())

    if candidates:
        body_text = max(candidates, key=lambda t: len(t) if len(t) < 20000 else 0)
        body_text = re.sub(r"\s+", " ", body_text).strip()[:5000]

    # Enhanced data processing with pandas
    tables_data = extract_tables_to_dataframe(tree)
    lists_data = extract_structured_lists(tree)
    text_analysis = analyze_text_content(body_text)

    # Create summary DataFrame for headers
    headers_df = (
        pd.DataFrame(
            {
                "header": headers,
                "level": [
                    f"h{tag}"
                    for tag in ["h1", "h2", "h3"]
                    for _ in range(
                        len(
                            [
                                h
                                for h in headers
                                if h
                                in [
                                    elem.text_content().strip()
                                    for elem in tree.xpath(f"//{tag}")
                                ]
                            ]
                        )
                    )
                ][: len(headers)],
                "length": [len(h) for h in headers],
            }
        )
        if headers
        else pd.DataFrame()
    )

    # Create summary for paragraphs
    paragraphs_df = (
        pd.DataFrame(
            {
                "paragraph": paragraphs,
                "length": [len(p) for p in paragraphs],
                "word_count": [len(p.split()) for p in paragraphs],
            }
        )
        if paragraphs
        else pd.DataFrame()
    )

    return {
        "url": url,
        "title": title,
        "headers": headers,
        "paragraphs": paragraphs,
        "meta_description": meta_description,
        "body_text": body_text,
        # Enhanced pandas-powered data
        "structured_data": {
            "tables": tables_data,
            "lists": lists_data,
            "headers_analysis": {
                "dataframe": headers_df.to_dict("records")
                if not headers_df.empty
                else [],
                "summary": headers_df.describe().to_dict()
                if not headers_df.empty
                else {},
            },
            "paragraphs_analysis": {
                "dataframe": paragraphs_df.to_dict("records")
                if not paragraphs_df.empty
                else [],
                "summary": paragraphs_df.describe().to_dict()
                if not paragraphs_df.empty
                else {},
            },
            "text_analysis": text_analysis,
        },
    }


def run_scraper_with_pandas(url: str) -> Dict[str, Any]:
    """Enhanced scraper with comprehensive pandas data processing."""
    # Get basic scraped data
    basic_data = run_scraper(url)

    # Process with pandas for enhanced analysis
    enhanced_data = process_web_data(basic_data, url)

    return enhanced_data


def run_scraper(url: str) -> Dict[str, Any]:
    session = requests.Session()
    headers = generate_headers()
    response = session.get(url, headers=headers, timeout=(5, 15))

    print("Status:", response.status_code)
    print("First 500 chars of response:\n", response.text[:500])  # 👈 DEBUG

    if detect_anti_bot(response):
        print("⚠️ Possible bot detection")

    return parse_page(response.content, url)
