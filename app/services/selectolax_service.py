from fastapi import HTTPException
import httpx
from selectolax.parser import HTMLParser
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def selectolax_scrape(url: str) -> dict:
    """
    Alternative scraping approach using httpx and selectolax
    """
    try:
        logger.info(f"Starting selectolax scrape for URL: {url}")

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

            # Extract title
            title = parser.css_first("title")
            title = title.text() if title else None

            # Extract description
            description_meta = parser.css_first('meta[name="description"]')
            description = (
                description_meta.attributes.get("content") if description_meta else None
            )

            # Extract links
            links = []
            for link in parser.css("a[href]"):
                href = link.attributes.get("href", "")
                text = link.text().strip()
                links.append({"text": text, "href": href})

            logger.info("selectolax scraping completed successfully")

            return {
                "url": url,
                "title": title,
                "description": description,
                "links": links,
            }

    except Exception as e:
        logger.error(f"Error in selectolax_scrape: {str(e)}")
        error_msg = str(e) if str(e) else "Unknown error occurred during scraping"
        raise HTTPException(status_code=500, detail=error_msg)
