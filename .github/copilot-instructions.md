# Copilot Coding Agent Instructions for AIWebScrapper

## Project Architecture

- **FastAPI-based web service** (`app/main.py`) exposes scraping endpoints.
- **Scraping strategies:**
  - `app/services/scraper_service.py`: Fast, bot-resistant scraping using `requests` + `lxml` for static sites.
  - `app/services/playwright_service.py`: Uses Playwright (async API) for dynamic/JS-heavy sites.
- **Routing:**
  - `app/routes/scraper.py` defines endpoints for both scraping methods.
  - `/scrape/beautiful-soup` uses `run_scraper` (static HTML).
  - `/scrape/playwright-scrape` uses `scrape_page` (dynamic content).

## Key Patterns & Conventions

- **User-Agent rotation** and browser-like headers in `scraper_service.py` to bypass basic bot detection.
- **Bot detection logic**: If anti-bot is detected, returns a clear error and suggests Playwright/Selenium.
- **Domain checks**: For known bot-protected domains (Amazon, Tesco, etc.), immediately suggest Playwright.
- **Playwright integration**:
  - `playwright_service.py` uses async API.
  - For Windows, prefer sync API (`playwright.sync_api`) if you encounter `NotImplementedError` with async.
- **Error handling**:
  - All endpoints return structured error messages and actionable suggestions.
  - Debug prints in `scraper_service.py` for status and response preview.

## Developer Workflows

- **Install dependencies:**
  - `uv pip install -r requirements.txt`
  - `playwright install` (to install browser binaries)
- **Run server:**
  - `uvicorn app.main:app --reload`
- **Test scraping endpoints:**
  - Use `/scrape/beautiful-soup` for static sites.
  - Use `/scrape/playwright-scrape` for dynamic/JS-heavy sites.

## Integration Points

- **External dependencies:**
  - `requests`, `lxml` for static scraping.
  - `playwright` for browser automation.
  - `scrapegraphai` (optional, for AI analysis).
- **Cross-component communication:**
  - Routers call service functions directly.
  - Data flows: request → service → response (no database).

## Project-Specific Advice

- **For Amazon, Tesco, and similar:**
  - Always use Playwright for real content.
  - Static scraping will only return page title or minimal info.
- **For Windows:**
  - If Playwright async API fails, switch to sync API.
- **Debugging:**
  - Use print statements in `scraper_service.py` to inspect HTTP status and response content.
- **Extendability:**
  - Add new scraping strategies as separate service modules.
  - Keep endpoint logic thin; put scraping logic in services.

## Example: Adding a New Scraper

- Create `app/services/new_scraper.py` with a function `run_new_scraper(url: str) -> dict`.
- Add an endpoint in `app/routes/scraper.py` that calls this function.
- Follow the error handling and response structure used in existing endpoints.

---

**Feedback needed:**

- Are there any custom build/test/deploy steps not covered here?
- Are there project-specific patterns or integrations that need more detail?

Let me know if you want to clarify or expand any section!
