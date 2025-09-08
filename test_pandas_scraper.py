from app.services.scraper_service import run_scraper_with_pandas
import json

print("Testing pandas-enhanced BeautifulSoup scraper...")

# Test with a simple webpage
url = "https://httpbin.org/html"
try:
    result = run_scraper_with_pandas(url)

    print("✓ Pandas-enhanced scraping successful!")
    print(f"URL: {result['url']}")
    print(f"Processing timestamp: {result['processed_at']}")

    # Check enhanced data
    enhanced = result.get("enhanced_data", {})
    print(f"\nEnhanced features found:")
    for feature, data in enhanced.items():
        if isinstance(data, list):
            print(f"  {feature}: {len(data)} items")
        elif isinstance(data, dict):
            print(f"  {feature}: {len(data)} keys")
        else:
            print(f"  {feature}: {type(data).__name__}")

    # Show statistics if available
    stats = result.get("statistics", {})
    if stats:
        processing_summary = stats.get("processing_summary", {})
        print(f"\nProcessing Summary:")
        print(
            f"  Data richness score: {processing_summary.get('data_richness_score', 0)}"
        )
        print(f"  Total tables: {processing_summary.get('total_tables', 0)}")
        print(f"  Total lists: {processing_summary.get('total_lists', 0)}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
