import random
import time
import re
from typing import Any, Dict
import requests
from lxml import html
from requests.exceptions import RequestException

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

    return {
        "url": url,
        "title": title,
        "headers": headers,
        "paragraphs": paragraphs,
        "meta_description": meta_description,
        "body_text": body_text,
    }


def run_scraper(url: str) -> Dict[str, Any]:
    session = requests.Session()
    headers = generate_headers()
    response = session.get(url, headers=headers, timeout=(5, 15))

    print("Status:", response.status_code)
    print("First 500 chars of response:\n", response.text[:500])  # 👈 DEBUG

    if detect_anti_bot(response):
        print("⚠️ Possible bot detection")

    return parse_page(response.content, url)
