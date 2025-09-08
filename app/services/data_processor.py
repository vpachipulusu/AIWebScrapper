"""
Advanced Data Processing Service using Pandas
Provides comprehensive data cleaning, formatting, and analysis capabilities
for web scraping results from both BeautifulSoup and Playwright services.
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


class WebDataProcessor:
    """Advanced data processor for web scraping results using pandas."""

    def __init__(self):
        self.processed_data = {}
        self.statistics = {}

    def process_scraped_data(
        self, raw_data: Dict[str, Any], url: str
    ) -> Dict[str, Any]:
        """
        Main method to process all scraped data with pandas.

        Args:
            raw_data: Raw scraped data from BeautifulSoup or Playwright
            url: Source URL

        Returns:
            Enhanced data with pandas analysis
        """
        try:
            processed_result = {
                "url": url,
                "processed_at": datetime.now().isoformat(),
                "source_domain": urlparse(url).netloc,
                "original_data": raw_data,
                "enhanced_data": {},
            }

            # Process different types of structured data
            if "structured_data" in raw_data:
                processed_result["enhanced_data"] = self._process_structured_data(
                    raw_data["structured_data"]
                )

            # Process tables if present
            if "tables" in raw_data:
                processed_result["enhanced_data"]["tables"] = self._process_tables_data(
                    raw_data["tables"]
                )

            # Process text content
            if "body_text" in raw_data:
                processed_result["enhanced_data"]["text_analysis"] = (
                    self._analyze_text_content(raw_data["body_text"])
                )

            # Generate comprehensive statistics
            processed_result["statistics"] = self._generate_comprehensive_stats(
                processed_result["enhanced_data"]
            )

            # Store processing metadata
            processed_result["processing_metadata"] = (
                self._generate_processing_metadata(
                    raw_data, processed_result["enhanced_data"]
                )
            )

            # Convert all NumPy types to native Python types for JSON serialization
            processed_result = convert_numpy_types(processed_result)

            return processed_result

        except Exception as e:
            logger.error(f"Error processing scraped data: {e}")
            return {
                "url": url,
                "error": str(e),
                "original_data": raw_data,
                "processed_at": datetime.now().isoformat(),
            }

    def _process_structured_data(
        self, structured_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process structured data with pandas analysis."""
        enhanced = {}

        # Process tables
        if "tables" in structured_data:
            enhanced["tables"] = self._enhance_tables_with_pandas(
                structured_data["tables"]
            )

        # Process lists
        if "lists" in structured_data:
            enhanced["lists"] = self._enhance_lists_with_pandas(
                structured_data["lists"]
            )

        # Process headers
        if "headers_analysis" in structured_data:
            enhanced["headers"] = self._process_headers_data(
                structured_data["headers_analysis"]
            )

        # Process paragraphs
        if "paragraphs_analysis" in structured_data:
            enhanced["paragraphs"] = self._process_paragraphs_data(
                structured_data["paragraphs_analysis"]
            )

        return enhanced

    def _enhance_tables_with_pandas(self, tables_data: List[Dict]) -> List[Dict]:
        """Enhance table data with advanced pandas analysis."""
        enhanced_tables = []

        for table in tables_data:
            try:
                if "dataframe" in table and table["dataframe"]:
                    # Recreate DataFrame from stored data
                    df = pd.DataFrame(table["dataframe"])

                    if not df.empty:
                        # Advanced cleaning and analysis
                        df_cleaned = self._advanced_dataframe_cleaning(df)

                        # Generate insights
                        insights = self._generate_table_insights(df_cleaned)

                        # Detect data patterns
                        patterns = self._detect_data_patterns(df_cleaned)

                        # Create enhanced table data
                        enhanced_table = {
                            **table,
                            "cleaned_dataframe": df_cleaned.to_dict("records"),
                            "advanced_statistics": self._comprehensive_table_stats(
                                df_cleaned
                            ),
                            "insights": insights,
                            "patterns": patterns,
                            "data_quality": self._assess_data_quality(df_cleaned),
                            "export_ready": self._prepare_export_formats(df_cleaned),
                        }

                        enhanced_tables.append(enhanced_table)
                    else:
                        enhanced_tables.append(table)
                else:
                    enhanced_tables.append(table)

            except Exception as e:
                logger.warning(f"Error enhancing table data: {e}")
                enhanced_tables.append(table)

        return enhanced_tables

    def _enhance_lists_with_pandas(self, lists_data: List[Dict]) -> List[Dict]:
        """Enhance list data with pandas analysis."""
        enhanced_lists = []

        for list_item in lists_data:
            try:
                if "items" in list_item:
                    items = list_item["items"]

                    # Create comprehensive DataFrame for list analysis
                    list_df = self._create_comprehensive_list_dataframe(items)

                    # Generate advanced insights
                    insights = self._generate_list_insights(list_df, items)

                    # Classify list content
                    classification = self._classify_list_content(items)

                    enhanced_list = {
                        **list_item,
                        "comprehensive_analysis": list_df.to_dict("records"),
                        "insights": insights,
                        "classification": classification,
                        "advanced_patterns": self._detect_list_patterns(items),
                        "export_ready": {
                            "csv": list_df.to_csv(index=False),
                            "json": list_df.to_json(orient="records"),
                            "summary": list_df.describe().to_dict(),
                        },
                    }

                    enhanced_lists.append(enhanced_list)
                else:
                    enhanced_lists.append(list_item)

            except Exception as e:
                logger.warning(f"Error enhancing list data: {e}")
                enhanced_lists.append(list_item)

        return enhanced_lists

    def _advanced_dataframe_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform advanced cleaning on DataFrame."""
        df_cleaned = df.copy()

        # Remove completely empty rows and columns
        df_cleaned = df_cleaned.dropna(how="all").dropna(axis=1, how="all")

        # Advanced string cleaning
        for col in df_cleaned.select_dtypes(include=["object"]).columns:
            # Clean whitespace and normalize
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            df_cleaned[col] = df_cleaned[col].str.replace(r"\s+", " ", regex=True)

            # Handle missing value indicators
            missing_indicators = [
                "",
                "None",
                "nan",
                "NaN",
                "null",
                "NULL",
                "-",
                "N/A",
                "n/a",
            ]
            df_cleaned[col] = df_cleaned[col].replace(missing_indicators, np.nan)

        # Advanced type conversion
        for col in df_cleaned.columns:
            df_cleaned[col] = self._smart_type_conversion(df_cleaned[col])

        return df_cleaned

    def _smart_type_conversion(self, series: pd.Series) -> pd.Series:
        """Intelligently convert series to appropriate data type."""
        if series.dtype == "object":
            # Try numeric conversion first
            numeric_series = pd.to_numeric(series, errors="coerce")
            non_null_count = series.notna().sum()
            numeric_success_rate = (
                numeric_series.notna().sum() / non_null_count
                if non_null_count > 0
                else 0
            )

            if numeric_success_rate > 0.7:  # 70% successfully converted
                return numeric_series

            # Try datetime conversion
            try:
                datetime_series = pd.to_datetime(
                    series, errors="coerce", infer_datetime_format=True
                )
                datetime_success_rate = (
                    datetime_series.notna().sum() / non_null_count
                    if non_null_count > 0
                    else 0
                )

                if datetime_success_rate > 0.7:
                    return datetime_series
            except Exception:
                pass

            # Try boolean conversion
            boolean_values = ["true", "false", "yes", "no", "1", "0"]
            if series.str.lower().isin(boolean_values).sum() / non_null_count > 0.8:
                boolean_map = {
                    "true": True,
                    "yes": True,
                    "1": True,
                    "false": False,
                    "no": False,
                    "0": False,
                }
                return series.str.lower().map(boolean_map)

        return series

    def _generate_table_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights from table data."""
        if df.empty:
            return {}

        insights = {
            "data_completeness": (
                1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            )
            * 100,
            "column_insights": {},
            "relationships": {},
            "anomalies": [],
        }

        # Column-level insights
        for col in df.columns:
            col_insights = {
                "data_type": str(df[col].dtype),
                "completeness": (1 - df[col].isnull().sum() / len(df)) * 100,
                "uniqueness": df[col].nunique() / len(df) * 100,
            }

            if df[col].dtype in ["int64", "float64"]:
                col_insights.update(
                    {
                        "outliers": self._detect_outliers(df[col]),
                        "distribution": "normal"
                        if self._is_normal_distribution(df[col])
                        else "skewed",
                    }
                )

            insights["column_insights"][col] = col_insights

        # Detect relationships between columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_correlations = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_correlations.append(
                            {
                                "column1": corr_matrix.columns[i],
                                "column2": corr_matrix.columns[j],
                                "correlation": corr_value,
                            }
                        )

            insights["relationships"]["high_correlations"] = high_correlations

        return insights

    def _create_comprehensive_list_dataframe(self, items: List[str]) -> pd.DataFrame:
        """Create a comprehensive DataFrame for list analysis."""
        list_df = pd.DataFrame(
            {
                "item": items,
                "item_number": range(1, len(items) + 1),
                "character_count": [len(item) for item in items],
                "word_count": [len(item.split()) for item in items],
                "sentence_count": [len(re.split(r"[.!?]+", item)) for item in items],
                "has_numbers": [bool(re.search(r"\d", item)) for item in items],
                "has_special_chars": [
                    bool(re.search(r"[^\w\s]", item)) for item in items
                ],
                "starts_with_capital": [
                    item[0].isupper() if item else False for item in items
                ],
                "ends_with_punctuation": [
                    item[-1] in ".!?:;" if item else False for item in items
                ],
                "contains_url": [
                    bool(re.search(r"http[s]?://", item)) for item in items
                ],
                "contains_email": [
                    bool(
                        re.search(
                            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", item
                        )
                    )
                    for item in items
                ],
                "all_caps_ratio": [
                    len(re.findall(r"[A-Z]", item)) / len(item) if item else 0
                    for item in items
                ],
            }
        )

        return list_df

    def _classify_list_content(self, items: List[str]) -> Dict[str, Any]:
        """Classify the type of content in a list."""
        classification = {
            "content_type": "mixed",
            "confidence": 0.0,
            "characteristics": {},
        }

        # Patterns to detect content types
        patterns = {
            "navigation": [r"home", r"about", r"contact", r"menu", r"link"],
            "product": [r"price", r"\$", r"buy", r"cart", r"product"],
            "feature": [r"feature", r"benefit", r"advantage", r"capability"],
            "instruction": [r"step", r"\d+\.", r"first", r"then", r"finally"],
            "social": [r"facebook", r"twitter", r"instagram", r"linkedin", r"share"],
        }

        scores = {}
        for content_type, keywords in patterns.items():
            score = 0
            for item in items:
                for keyword in keywords:
                    if re.search(keyword, item.lower()):
                        score += 1
            scores[content_type] = score / len(items)

        # Determine primary content type
        if scores:
            primary_type = max(scores, key=scores.get)
            classification["content_type"] = primary_type
            classification["confidence"] = scores[primary_type]
            classification["characteristics"] = scores

        return classification

    def _detect_outliers(self, series: pd.Series) -> List[Any]:
        """Detect outliers in numeric series using IQR method."""
        if series.dtype not in ["int64", "float64"] or series.isnull().all():
            return []

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)].tolist()
        return outliers

    def _is_normal_distribution(self, series: pd.Series) -> bool:
        """Check if series follows normal distribution using skewness."""
        if series.dtype not in ["int64", "float64"] or len(series.dropna()) < 3:
            return False

        skewness = series.skew()
        return abs(skewness) < 0.5  # Roughly normal if skewness is close to 0

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality of DataFrame."""
        if df.empty:
            return {"score": 0, "issues": ["Empty dataset"]}

        quality_score = 100
        issues = []

        # Check completeness
        missing_percentage = (
            df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        ) * 100
        if missing_percentage > 20:
            quality_score -= 20
            issues.append(f"High missing data percentage: {missing_percentage:.1f}%")
        elif missing_percentage > 10:
            quality_score -= 10
            issues.append(f"Moderate missing data: {missing_percentage:.1f}%")

        # Check for duplicate rows
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 10:
            quality_score -= 15
            issues.append(f"High duplicate rows: {duplicate_percentage:.1f}%")

        # Check data consistency
        for col in df.select_dtypes(include=["object"]).columns:
            unique_ratio = df[col].nunique() / len(df.dropna())
            if unique_ratio > 0.9:  # Mostly unique values
                quality_score -= 5
                issues.append(f"Column '{col}' has mostly unique values")

        return {
            "score": max(0, quality_score),
            "issues": issues,
            "recommendations": self._generate_quality_recommendations(issues),
        }

    def _generate_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on data quality issues."""
        recommendations = []

        for issue in issues:
            if "missing data" in issue.lower():
                recommendations.append(
                    "Consider data imputation or collection improvement"
                )
            elif "duplicate" in issue.lower():
                recommendations.append("Remove duplicate rows to improve data quality")
            elif "unique values" in issue.lower():
                recommendations.append(
                    "Check if column should be an identifier or needs normalization"
                )

        return recommendations

    def _prepare_export_formats(self, df: pd.DataFrame) -> Dict[str, str]:
        """Prepare data in various export formats."""
        export_formats = {}

        try:
            export_formats["csv"] = df.to_csv(index=False)
            export_formats["json"] = df.to_json(orient="records", date_format="iso")
            export_formats["html"] = df.to_html(
                index=False, classes="table table-striped"
            )

            # Summary statistics
            if not df.select_dtypes(include=[np.number]).empty:
                export_formats["summary_stats"] = df.describe().to_dict()

        except Exception as e:
            logger.warning(f"Error preparing export formats: {e}")

        return export_formats

    def _comprehensive_table_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics for table."""
        if df.empty:
            return {}

        stats = {
            "basic_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "data_types": df.dtypes.value_counts().to_dict(),
            },
            "missing_data": {
                "total_missing_cells": df.isnull().sum().sum(),
                "missing_percentage": (
                    df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
                )
                * 100,
                "columns_with_missing": df.columns[df.isnull().any()].tolist(),
            },
        }

        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_analysis"] = {
                "columns": numeric_cols.tolist(),
                "statistics": df[numeric_cols].describe().to_dict(),
                "correlations": df[numeric_cols].corr().to_dict()
                if len(numeric_cols) > 1
                else {},
            }

        # Text column analysis
        text_cols = df.select_dtypes(include=["object"]).columns
        if len(text_cols) > 0:
            text_stats = {}
            for col in text_cols:
                valid_values = df[col].dropna()
                if not valid_values.empty:
                    text_stats[col] = {
                        "unique_count": valid_values.nunique(),
                        "most_common": valid_values.value_counts().head(3).to_dict(),
                        "avg_length": valid_values.astype(str).str.len().mean(),
                        "max_length": valid_values.astype(str).str.len().max(),
                    }
            stats["text_analysis"] = text_stats

        return stats

    def _detect_data_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in the data."""
        patterns = {
            "sequential_data": False,
            "time_series": False,
            "categorical_dominance": [],
            "structured_format": False,
        }

        # Check for sequential patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 1:
                diff = df[col].diff().dropna()
                if (diff == diff.iloc[0]).all():  # Constant difference
                    patterns["sequential_data"] = True
                    break

        # Check for time series data
        for col in df.columns:
            if df[col].dtype == "datetime64[ns]":
                patterns["time_series"] = True
                break

        # Check categorical dominance
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() < len(df) * 0.1:  # Less than 10% unique values
                patterns["categorical_dominance"].append(col)

        # Check for structured format
        if len(df.columns) > 2 and df.shape[0] > 5:
            # Simple heuristic: if most columns have data, it's structured
            completeness = 1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            patterns["structured_format"] = completeness > 0.7

        return patterns

    def _process_headers_data(self, headers_data: Dict) -> Dict[str, Any]:
        """Process headers data with pandas."""
        if "dataframe" not in headers_data or not headers_data["dataframe"]:
            return headers_data

        headers_df = pd.DataFrame(headers_data["dataframe"])

        if not headers_df.empty:
            # Analyze header hierarchy and structure
            hierarchy_analysis = {
                "total_headers": len(headers_df),
                "hierarchy_distribution": headers_df["level"].value_counts().to_dict()
                if "level" in headers_df.columns
                else {},
                "avg_length": headers_df["length"].mean()
                if "length" in headers_df.columns
                else 0,
                "structure_quality": self._assess_header_structure(headers_df),
            }

            return {
                **headers_data,
                "hierarchy_analysis": hierarchy_analysis,
                "processed_dataframe": headers_df.to_dict("records"),
            }

        return headers_data

    def _assess_header_structure(self, headers_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of header structure."""
        quality = {
            "has_h1": False,
            "logical_hierarchy": True,
            "consistent_style": True,
            "score": 100,
        }

        if "level" in headers_df.columns:
            levels = headers_df["level"].tolist()

            # Check for H1
            quality["has_h1"] = "h1" in levels
            if not quality["has_h1"]:
                quality["score"] -= 20

            # Check for logical hierarchy (H1 -> H2 -> H3, etc.)
            level_order = ["h1", "h2", "h3", "h4", "h5", "h6"]
            prev_level_index = -1

            for level in levels:
                if level in level_order:
                    current_level_index = level_order.index(level)
                    if current_level_index > prev_level_index + 1:
                        quality["logical_hierarchy"] = False
                        quality["score"] -= 15
                        break
                    prev_level_index = max(prev_level_index, current_level_index)

        return quality

    def _process_paragraphs_data(self, paragraphs_data: Dict) -> Dict[str, Any]:
        """Process paragraphs data with pandas."""
        if "dataframe" not in paragraphs_data or not paragraphs_data["dataframe"]:
            return paragraphs_data

        paragraphs_df = pd.DataFrame(paragraphs_data["dataframe"])

        if not paragraphs_df.empty:
            # Advanced text analysis
            readability_analysis = self._analyze_readability(paragraphs_df)
            content_structure = self._analyze_content_structure(paragraphs_df)

            return {
                **paragraphs_data,
                "readability_analysis": readability_analysis,
                "content_structure": content_structure,
                "processed_dataframe": paragraphs_df.to_dict("records"),
            }

        return paragraphs_data

    def _analyze_readability(self, paragraphs_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze text readability."""
        if "paragraph" not in paragraphs_df.columns:
            return {}

        readability = {
            "avg_paragraph_length": paragraphs_df["length"].mean()
            if "length" in paragraphs_df.columns
            else 0,
            "avg_words_per_paragraph": paragraphs_df["word_count"].mean()
            if "word_count" in paragraphs_df.columns
            else 0,
            "reading_difficulty": "medium",  # Simplified classification
        }

        # Simple readability classification based on paragraph length
        avg_length = readability["avg_paragraph_length"]
        if avg_length < 100:
            readability["reading_difficulty"] = "easy"
        elif avg_length > 300:
            readability["reading_difficulty"] = "hard"

        return readability

    def _analyze_content_structure(self, paragraphs_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content structure."""
        structure = {
            "total_paragraphs": len(paragraphs_df),
            "consistency_score": 0.0,
            "structure_type": "varied",
        }

        if "length" in paragraphs_df.columns:
            lengths = paragraphs_df["length"]
            coefficient_of_variation = (
                lengths.std() / lengths.mean() if lengths.mean() > 0 else 0
            )

            structure["consistency_score"] = max(
                0, 100 - (coefficient_of_variation * 100)
            )

            if coefficient_of_variation < 0.3:
                structure["structure_type"] = "consistent"
            elif coefficient_of_variation > 0.7:
                structure["structure_type"] = "highly_varied"

        return structure

    def _analyze_text_content(self, text: str) -> Dict[str, Any]:
        """Analyze text content using pandas."""
        if not text:
            return {}

        # Create word-level DataFrame
        words = re.findall(r"\b\w+\b", text.lower())
        words_df = pd.DataFrame({"word": words})

        # Create sentence-level DataFrame
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sentences_df = pd.DataFrame(
            {
                "sentence": sentences,
                "length": [len(s) for s in sentences],
                "word_count": [len(s.split()) for s in sentences],
            }
        )

        analysis = {
            "total_words": len(words),
            "unique_words": words_df["word"].nunique(),
            "total_sentences": len(sentences_df),
            "vocabulary_richness": words_df["word"].nunique() / len(words)
            if words
            else 0,
            "word_frequency": words_df["word"].value_counts().head(20).to_dict(),
            "sentence_statistics": sentences_df.describe().to_dict()
            if not sentences_df.empty
            else {},
            "readability_metrics": self._calculate_readability_metrics(
                words, sentences
            ),
        }

        return analysis

    def _calculate_readability_metrics(
        self, words: List[str], sentences: List[str]
    ) -> Dict[str, float]:
        """Calculate readability metrics."""
        if not words or not sentences:
            return {}

        avg_sentence_length = len(words) / len(sentences)

        # Count complex words (simplified: words with 3+ syllables approximated by length)
        complex_words = [w for w in words if len(w) > 6]
        complex_word_ratio = len(complex_words) / len(words)

        # Simplified Flesch-Kincaid Grade Level
        grade_level = 0.39 * avg_sentence_length + 11.8 * complex_word_ratio - 15.59
        grade_level = max(0, min(20, grade_level))

        return {
            "avg_sentence_length": avg_sentence_length,
            "complex_word_ratio": complex_word_ratio,
            "flesch_kincaid_grade": grade_level,
            "reading_ease": max(
                0,
                min(
                    100,
                    206.835 - 1.015 * avg_sentence_length - 84.6 * complex_word_ratio,
                ),
            ),
        }

    def _generate_comprehensive_stats(
        self, enhanced_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive statistics across all processed data."""
        stats = {
            "processing_summary": {
                "total_tables": len(enhanced_data.get("tables", [])),
                "total_lists": len(enhanced_data.get("lists", [])),
                "has_text_analysis": "text_analysis" in enhanced_data,
                "data_richness_score": 0,
            },
            "data_quality_overview": {},
            "actionable_insights": [],
        }

        # Calculate data richness score
        richness_score = 0
        if enhanced_data.get("tables"):
            richness_score += 30
        if enhanced_data.get("lists"):
            richness_score += 20
        if enhanced_data.get("text_analysis"):
            richness_score += 25
        if enhanced_data.get("headers"):
            richness_score += 15
        if enhanced_data.get("paragraphs"):
            richness_score += 10

        stats["processing_summary"]["data_richness_score"] = richness_score

        # Generate actionable insights
        insights = []

        if richness_score < 50:
            insights.append(
                "Page has limited structured data - consider using different extraction methods"
            )

        if enhanced_data.get("tables"):
            for table in enhanced_data["tables"]:
                if "data_quality" in table and table["data_quality"]["score"] < 70:
                    insights.append(
                        f"Table {table.get('table_index', 'unknown')} has quality issues"
                    )

        stats["actionable_insights"] = insights

        return stats

    def _generate_processing_metadata(
        self, original_data: Dict, enhanced_data: Dict
    ) -> Dict[str, Any]:
        """Generate metadata about the processing operation."""
        return {
            "original_data_size": len(str(original_data)),
            "enhanced_data_size": len(str(enhanced_data)),
            "enhancement_ratio": len(str(enhanced_data)) / len(str(original_data))
            if original_data
            else 0,
            "processing_features_used": list(enhanced_data.keys()),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }

    def _detect_list_patterns(self, items: List[str]) -> Dict[str, Any]:
        """Detect advanced patterns in list items."""
        if not items:
            return {}

        # Create DataFrame for pattern analysis
        df = pd.DataFrame({"item": items})

        patterns = {
            "numbering_pattern": self._detect_numbering_pattern(items),
            "bullet_pattern": self._detect_bullet_pattern(items),
            "structural_consistency": self._analyze_structural_consistency(items),
            "content_themes": self._detect_content_themes(items),
            "formatting_patterns": self._detect_formatting_patterns(df),
        }

        return patterns

    def _detect_numbering_pattern(self, items: List[str]) -> Dict[str, Any]:
        """Detect numbering patterns in list items."""
        numbered_items = [
            i for i, item in enumerate(items) if re.match(r"^\d+\.?\s", item)
        ]

        return {
            "has_numbering": len(numbered_items) > 0,
            "percentage_numbered": len(numbered_items) / len(items) * 100,
            "sequential": len(numbered_items) == len(items)
            and all(re.match(rf"^{i + 1}\.?\s", items[i]) for i in range(len(items)))
            if numbered_items
            else False,
        }

    def _detect_bullet_pattern(self, items: List[str]) -> Dict[str, Any]:
        """Detect bullet point patterns."""
        bullet_chars = ["•", "▪", "▫", "◦", "‣", "⁃", "-", "*"]
        bulleted_items = [
            i
            for i, item in enumerate(items)
            if any(item.startswith(char) for char in bullet_chars)
        ]

        return {
            "has_bullets": len(bulleted_items) > 0,
            "percentage_bulleted": len(bulleted_items) / len(items) * 100,
            "consistent_bullets": len(set(items[i][0] for i in bulleted_items)) == 1
            if bulleted_items
            else False,
        }

    def _analyze_structural_consistency(self, items: List[str]) -> Dict[str, float]:
        """Analyze structural consistency of list items."""
        if len(items) < 2:
            return {"score": 100.0}

        # Analyze length consistency
        lengths = [len(item) for item in items]
        length_cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0

        # Analyze word count consistency
        word_counts = [len(item.split()) for item in items]
        word_cv = (
            np.std(word_counts) / np.mean(word_counts)
            if np.mean(word_counts) > 0
            else 0
        )

        # Overall consistency score (lower coefficient of variation = more consistent)
        consistency_score = max(0, 100 - (length_cv + word_cv) * 50)

        return {
            "score": consistency_score,
            "length_consistency": max(0, 100 - length_cv * 100),
            "word_count_consistency": max(0, 100 - word_cv * 100),
        }

    def _detect_content_themes(self, items: List[str]) -> Dict[str, Any]:
        """Detect themes in content using simple keyword analysis."""
        all_text = " ".join(items).lower()
        words = re.findall(r"\b\w+\b", all_text)

        # Common theme keywords
        theme_keywords = {
            "technology": [
                "software",
                "app",
                "digital",
                "online",
                "tech",
                "system",
                "data",
            ],
            "business": [
                "company",
                "service",
                "customer",
                "sales",
                "marketing",
                "business",
            ],
            "product": ["product", "feature", "quality", "design", "price", "buy"],
            "education": [
                "learn",
                "course",
                "training",
                "education",
                "skill",
                "knowledge",
            ],
            "health": ["health", "medical", "doctor", "treatment", "wellness", "care"],
        }

        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(1 for word in words if word in keywords)
            theme_scores[theme] = score / len(words) * 100 if words else 0

        dominant_theme = (
            max(theme_scores, key=theme_scores.get) if theme_scores else "general"
        )

        return {
            "theme_scores": theme_scores,
            "dominant_theme": dominant_theme,
            "theme_confidence": theme_scores.get(dominant_theme, 0),
        }

    def _detect_formatting_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect formatting patterns in list items."""
        if "item" not in df.columns:
            return {}

        patterns = {
            "capitalization": {
                "all_caps": df["item"].str.isupper().sum(),
                "title_case": df["item"].str.istitle().sum(),
                "sentence_case": df["item"]
                .apply(lambda x: x[0].isupper() if x else False)
                .sum(),
            },
            "punctuation": {
                "ends_with_period": df["item"].str.endswith(".").sum(),
                "ends_with_colon": df["item"].str.endswith(":").sum(),
                "contains_quotes": df["item"].str.contains(r'["\']').sum(),
            },
            "special_formatting": {
                "contains_parentheses": df["item"].str.contains(r"\([^)]*\)").sum(),
                "contains_brackets": df["item"].str.contains(r"\[[^\]]*\]").sum(),
                "contains_emphasis": df["item"].str.contains(r"\*[^*]*\*").sum(),
            },
        }

        return patterns

    def _generate_list_insights(
        self, list_df: pd.DataFrame, items: List[str]
    ) -> Dict[str, Any]:
        """Generate insights from list data analysis."""
        insights = {
            "content_analysis": {},
            "structure_analysis": {},
            "recommendations": [],
        }

        if not list_df.empty:
            # Content insights
            insights["content_analysis"] = {
                "avg_complexity": list_df["word_count"].mean(),
                "content_variety": list_df["character_count"].std()
                / list_df["character_count"].mean()
                if list_df["character_count"].mean() > 0
                else 0,
                "information_density": (
                    list_df["has_numbers"].sum() + list_df["has_special_chars"].sum()
                )
                / len(list_df),
            }

            # Structure insights
            insights["structure_analysis"] = {
                "formatting_consistency": self._calculate_formatting_consistency(
                    list_df
                ),
                "length_distribution": list_df["character_count"].describe().to_dict(),
                "professional_appearance": self._assess_professional_appearance(
                    list_df
                ),
            }

            # Generate recommendations
            recommendations = []

            if insights["content_analysis"]["content_variety"] > 1.0:
                recommendations.append(
                    "Consider breaking down longer items for better readability"
                )

            if insights["structure_analysis"]["formatting_consistency"] < 70:
                recommendations.append(
                    "Improve formatting consistency across list items"
                )

            if list_df["contains_url"].sum() > 0:
                recommendations.append(
                    "List contains URLs - consider extracting for separate analysis"
                )

            insights["recommendations"] = recommendations

        return insights

    def _calculate_formatting_consistency(self, list_df: pd.DataFrame) -> float:
        """Calculate formatting consistency score."""
        if list_df.empty:
            return 0.0

        consistency_factors = []

        # Check capitalization consistency
        cap_patterns = [
            list_df["starts_with_capital"].sum() / len(list_df),
            1 - (list_df["starts_with_capital"].sum() / len(list_df)),
        ]
        consistency_factors.append(max(cap_patterns))

        # Check punctuation consistency
        punct_patterns = [
            list_df["ends_with_punctuation"].sum() / len(list_df),
            1 - (list_df["ends_with_punctuation"].sum() / len(list_df)),
        ]
        consistency_factors.append(max(punct_patterns))

        # Overall consistency score
        return np.mean(consistency_factors) * 100

    def _assess_professional_appearance(self, list_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess professional appearance of list formatting."""
        if list_df.empty:
            return {"score": 0}

        score = 100
        issues = []

        # Check for inconsistent capitalization
        if 0.2 < list_df["starts_with_capital"].mean() < 0.8:
            score -= 20
            issues.append("Inconsistent capitalization")

        # Check for mixed punctuation
        if 0.2 < list_df["ends_with_punctuation"].mean() < 0.8:
            score -= 15
            issues.append("Inconsistent punctuation")

        # Check for excessive special characters
        if list_df["has_special_chars"].mean() > 0.7:
            score -= 10
            issues.append("High special character usage")

        return {
            "score": max(0, score),
            "issues": issues,
            "strengths": self._identify_formatting_strengths(list_df),
        }

    def _identify_formatting_strengths(self, list_df: pd.DataFrame) -> List[str]:
        """Identify formatting strengths."""
        strengths = []

        if list_df["starts_with_capital"].mean() > 0.9:
            strengths.append("Consistent capitalization")

        if list_df["ends_with_punctuation"].mean() > 0.9:
            strengths.append("Consistent punctuation")

        cv = (
            list_df["character_count"].std() / list_df["character_count"].mean()
            if list_df["character_count"].mean() > 0
            else 0
        )
        if cv < 0.3:
            strengths.append("Consistent item length")

        return strengths


# Create global instance
web_data_processor = WebDataProcessor()


def process_web_data(raw_data: Dict[str, Any], url: str) -> Dict[str, Any]:
    """
    Convenience function to process web scraping data.

    Args:
        raw_data: Raw scraped data
        url: Source URL

    Returns:
        Enhanced data with pandas analysis
    """
    return web_data_processor.process_scraped_data(raw_data, url)
