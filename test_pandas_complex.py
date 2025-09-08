from app.services.scraper_service import run_scraper_with_pandas
import json

print("Testing pandas-enhanced scraper with Wikipedia page (has tables and lists)...")

# Test with Wikipedia page that has structured data
url = "https://en.wikipedia.org/wiki/Pandas_(software)"
try:
    result = run_scraper_with_pandas(url)

    print("✓ Pandas-enhanced scraping successful!")
    print(f"URL: {result['url']}")

    # Check enhanced data
    enhanced = result.get("enhanced_data", {})
    print("\nEnhanced features found:")

    # Tables analysis
    tables = enhanced.get("tables", [])
    print(f"  Tables: {len(tables)} found")
    for i, table in enumerate(tables):
        if "statistics" in table:
            stats = table["statistics"]
            print(
                f"    Table {i + 1}: {stats['shape']} shape, {stats.get('missing_data', {}).get('missing_percentage', 0):.1f}% missing data"
            )

    # Lists analysis
    lists = enhanced.get("lists", [])
    print(f"  Lists: {len(lists)} found")
    for i, list_item in enumerate(lists):
        if "statistics" in list_item:
            stats = list_item["statistics"]
            print(
                f"    List {i + 1}: {stats['total_items']} items, avg length: {stats['avg_length']:.1f}"
            )

    # Text analysis
    text_analysis = enhanced.get("text_analysis", {})
    if text_analysis:
        print(f"  Text Analysis:")
        print(f"    Total words: {text_analysis.get('total_words', 0)}")
        print(
            f"    Vocabulary richness: {text_analysis.get('vocabulary_richness', 0):.2f}"
        )
        if "readability_metrics" in text_analysis:
            readability = text_analysis["readability_metrics"]
            print(f"    Reading ease: {readability.get('reading_ease', 0):.1f}")

    # Overall statistics
    stats = result.get("statistics", {})
    if stats:
        processing_summary = stats.get("processing_summary", {})
        print(f"\nProcessing Summary:")
        print(
            f"  Data richness score: {processing_summary.get('data_richness_score', 0)}"
        )

        # Show actionable insights
        insights = stats.get("actionable_insights", [])
        if insights:
            print(f"  Insights: {len(insights)} recommendations")
            for insight in insights[:3]:  # Show first 3
                print(f"    - {insight}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
