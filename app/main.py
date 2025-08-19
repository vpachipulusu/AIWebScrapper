from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference
from app.routes.scraper import router as scraper_router


app = FastAPI(title="AI Web Scraper", description="A web scraper for AI-related content.")

@app.get("/scalar", include_in_schema=False)
def get_scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


app.include_router(scraper_router, prefix="/scrape", tags=["scraper"])