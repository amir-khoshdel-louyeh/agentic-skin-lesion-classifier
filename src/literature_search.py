"""Real literature search integration with PubMed/Entrez API for lesion classification context."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PubMedArticle(BaseModel):
    """Structured PubMed article reference."""

    pmid: str
    title: str
    authors: str
    year: int
    relevance_score: float = 1.0

    def to_citation_string(self) -> str:
        """Format as a readable citation."""
        return f"PMID:{self.pmid} - {self.title} ({self.authors}, {self.year})"


class LiteratureSearchEngine:
    """Query PubMed for dermatology and skin lesion classification literature."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    DB = "pubmed"
    TOOL = "agentic-skin-lesion-classifier"
    EMAIL = "research@example.com"  # Required by NCBI
    REQUEST_TIMEOUT = 10.0  # seconds
    RATE_LIMIT_DELAY = 0.5  # seconds between requests
    LAST_REQUEST_TIME = 0.0

    def __init__(self):
        """Initialize the search engine."""
        self.client = httpx.Client(timeout=self.REQUEST_TIMEOUT)
        self.last_request_time = 0.0

    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass

    def _respect_rate_limit(self) -> None:
        """Enforce NCBI rate limiting (no more than 3 requests/second)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def _search(self, query: str, retmax: int = 20) -> list[str]:
        """Execute a PubMed search and return list of PMIDs.

        Args:
            query: Search query string
            retmax: Maximum number of results to return

        Returns:
            List of PMIDs
        """
        try:
            self._respect_rate_limit()
            params = {
                "db": self.DB,
                "tool": self.TOOL,
                "email": self.EMAIL,
                "term": query,
                "rettype": "json",
                "retmode": "json",
                "retmax": retmax,
            }
            response = self.client.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
            response.raise_for_status()

            # Check if response is empty
            if not response.text:
                logger.debug(f"Empty response from PubMed for query: {query}")
                return []

            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            return pmids[:retmax]

        except Exception as exc:
            logger.warning(f"PubMed search failed for query '{query}': {exc}")
            return []

    def _fetch_articles(self, pmids: list[str]) -> list[dict[str, Any]]:
        """Fetch full article summaries for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of article summaries
        """
        if not pmids:
            return []

        try:
            self._respect_rate_limit()
            params = {
                "db": self.DB,
                "tool": self.TOOL,
                "email": self.EMAIL,
                "id": ",".join(pmids),
                "rettype": "json",
                "retmode": "json",
            }
            response = self.client.get(f"{self.BASE_URL}/esummary.fcgi", params=params)
            response.raise_for_status()

            if not response.text:
                logger.debug(f"Empty response from PubMed for PMIDs: {pmids}")
                return []

            data = response.json()
            articles = []
            results = data.get("result", {})

            for pmid in pmids:
                if pmid not in results:
                    continue
                article = results[pmid]
                articles.append(
                    {
                        "pmid": pmid,
                        "title": article.get("title", "Untitled"),
                        "authors": self._extract_first_author(article.get("authors", [])),
                        "year": int(article.get("pubdate", "0000").split()[0][:4] or "0"),
                    }
                )

            return articles

        except Exception as exc:
            logger.warning(f"PubMed fetch failed for PMIDs {pmids}: {exc}")
            return []

    @staticmethod
    def _extract_first_author(authors: list[dict[str, str]]) -> str:
        """Extract first author name from author list."""
        if not authors:
            return "Unknown"
        first = authors[0].get("name", "Unknown")
        if len(authors) > 1:
            return f"{first} et al."
        return first

    def search_by_features(
        self,
        asymmetry: float,
        color_variation: float,
        border_irregularity: float,
        brightness: float,
    ) -> list[PubMedArticle]:
        """Generate search queries from clinical features and fetch relevant literature.

        Args:
            asymmetry: Asymmetry score (0-1)
            color_variation: Color variation score (0-1)
            border_irregularity: Border irregularity score (0-1)
            brightness: Brightness score (0-1)

        Returns:
            List of PubMedArticle objects
        """
        queries = self._build_search_queries(asymmetry, color_variation, border_irregularity, brightness)

        all_pmids = set()
        for query in queries:
            pmids = self._search(query, retmax=10)
            all_pmids.update(pmids)

        # Limit total results
        all_pmids = list(all_pmids)[:20]

        # Fetch article details
        articles_data = self._fetch_articles(all_pmids)

        # Convert to PubMedArticle objects
        articles = []
        for data in articles_data:
            try:
                article = PubMedArticle(**data)
                articles.append(article)
            except Exception as exc:
                logger.debug(f"Failed to parse article {data}: {exc}")

        return articles

    @staticmethod
    def _build_search_queries(
        asymmetry: float,
        color_variation: float,
        border_irregularity: float,
        brightness: float,
    ) -> list[str]:
        """Build intelligent PubMed search queries based on clinical features.

        Args:
            asymmetry: Asymmetry score (0-1)
            color_variation: Color variation score (0-1)
            border_irregularity: Border irregularity score (0-1)
            brightness: Brightness score (0-1)

        Returns:
            List of search query strings
        """
        queries = []

        # Base query: melanoma or skin cancer classification
        queries.append("melanoma diagnosis dermoscopy ABCD criteria")

        # High asymmetry: asymmetry is strong melanoma indicator
        if asymmetry > 0.5:
            queries.append("asymmetry melanoma dermatology dermoscopy")

        # High color variation: polychromatic lesions
        if color_variation > 0.5:
            queries.append("polychromatic lesion melanoma color variation")

        # High border irregularity: border is melanoma criterion
        if border_irregularity > 0.5:
            queries.append("irregular border melanoma dermoscopy classification")

        # Low brightness (dark lesion): common in melanoma
        if brightness < 0.4:
            queries.append("dark lesion melanoma pigmented dermoscopy")

        # High-risk combinations
        if asymmetry > 0.6 and color_variation > 0.6:
            queries.append("melanoma feature extraction asymmetry polychromia")

        # Edge case: very high scores across multiple features
        if (asymmetry + color_variation + border_irregularity) / 3 > 0.6:
            queries.append("high-risk melanoma classification clinical evaluation")

        # General dermatology context
        queries.append("skin lesion classification machine learning")

        return queries


def search_literature(
    asymmetry: float,
    color_variation: float,
    border_irregularity: float,
    brightness: float,
    max_results: int = 5,
) -> list[str]:
    """Convenience function to search PubMed and return formatted citations.

    Args:
        asymmetry: Asymmetry score (0-1)
        color_variation: Color variation score (0-1)
        border_irregularity: Border irregularity score (0-1)
        brightness: Brightness score (0-1)
        max_results: Maximum number of citations to return

    Returns:
        List of formatted citation strings
    """
    engine = LiteratureSearchEngine()
    try:
        articles = engine.search_by_features(asymmetry, color_variation, border_irregularity, brightness)
        # Return top results as formatted citations
        citations = [article.to_citation_string() for article in articles[:max_results]]
        return citations if citations else ["No literature references found (network or API error)"]
    except Exception as exc:
        logger.error(f"Literature search error: {exc}")
        return [f"Literature search failed: {exc}"]
    finally:
        engine.client.close()
