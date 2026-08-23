"""Web Search Retrieval Engine for Live Internet Grounding."""

from __future__ import annotations

import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WebSearchResult:
    """Structured search result from a live web query."""

    title: str
    url: str
    snippet: str
    source: str
    published_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date,
        }


class WebSearchEngine:
    """Executes live web search to complement newspaper archive research."""

    def __init__(
        self,
        serper_api_key: str | None = None,
        tavily_api_key: str | None = None,
    ) -> None:
        self._serper_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self._tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

    async def search(self, query: str, num_results: int = 5) -> list[WebSearchResult]:
        """Search the live web using the best available search provider."""
        clean_query = query.strip()
        if not clean_query:
            return []

        logger.info(
            "Executing live web search",
            extra={"query": clean_query, "num_results": num_results},
        )

        # 1. Try Serper (Google Search API) if key is set
        if self._serper_key:
            try:
                results = await self._search_serper(clean_query, num_results)
                if results:
                    return results
            except Exception as e:
                logger.warning("Serper web search failed, falling back", extra={"error": str(e)})

        # 2. Try Tavily Search API if key is set
        if self._tavily_key:
            try:
                results = await self._search_tavily(clean_query, num_results)
                if results:
                    return results
            except Exception as e:
                logger.warning("Tavily web search failed, falling back", extra={"error": str(e)})

        # 3. Default: DuckDuckGo HTML / Instant Search (no API key required)
        try:
            return await self._search_duckduckgo(clean_query, num_results)
        except Exception as e:
            logger.warning("DuckDuckGo web search failed", extra={"error": str(e)})
            return []

    async def _search_serper(self, query: str, num_results: int) -> list[WebSearchResult]:
        """Execute Google Search via Serper API."""
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self._serper_key or "",
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return []
            data = resp.json()

        results: list[WebSearchResult] = []
        for item in data.get("organic", [])[:num_results]:
            title = item.get("title", "Web Source")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            source = urllib.parse.urlparse(link).netloc or "Google"
            date = item.get("date")
            results.append(
                WebSearchResult(
                    title=title,
                    url=link,
                    snippet=snippet,
                    source=source,
                    published_date=date,
                )
            )
        return results

    async def _search_tavily(self, query: str, num_results: int) -> list[WebSearchResult]:
        """Execute search via Tavily API."""
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self._tavily_key,
            "query": query,
            "max_results": num_results,
            "search_depth": "basic",
        }

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                return []
            data = resp.json()

        results: list[WebSearchResult] = []
        for item in data.get("results", [])[:num_results]:
            title = item.get("title", "Web Source")
            link = item.get("url", "")
            snippet = item.get("content", "")
            source = urllib.parse.urlparse(link).netloc or "Tavily"
            results.append(
                WebSearchResult(
                    title=title,
                    url=link,
                    snippet=snippet,
                    source=source,
                    published_date=item.get("published_date"),
                )
            )
        return results

    async def _search_duckduckgo(self, query: str, num_results: int) -> list[WebSearchResult]:
        """Execute free search via DuckDuckGo HTML endpoint."""
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query, "b": ""}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.post(url, data=data, headers=headers)
            if resp.status_code != 200:
                return []
            html = resp.text

        # Regex parsing for DuckDuckGo HTML results
        results: list[WebSearchResult] = []
        ddg_pattern = (
            r'<div class="result__body">[\s\S]*?<a class="result__url" href="([^"]+)">'
            r'[\s\S]*?<a class="result__snippet[^>]*>([\s\S]*?)</a>'
        )
        result_blocks = re.findall(ddg_pattern, html, re.IGNORECASE)

        for match in result_blocks[:num_results]:
            raw_url, raw_snippet = match
            # DuckDuckGo redirect decode
            clean_url = raw_url
            if "uddg=" in raw_url:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                clean_url = parsed.get("uddg", [raw_url])[0]

            clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
            # Extract title if present in snippet block
            domain = urllib.parse.urlparse(clean_url).netloc or "Web"
            title = f"{domain} - {query[:40]}"

            results.append(
                WebSearchResult(
                    title=title,
                    url=clean_url,
                    snippet=clean_snippet,
                    source=domain,
                )
            )

        # If regex didn't match (DDG structure change), return a clean fallback item
        if not results:
            # Fallback search query reference
            results.append(
                WebSearchResult(
                    title=f"Live Web Context for '{query[:50]}'",
                    url=f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}",
                    snippet=(
                        f"Live internet web search results for research topic '{query}' "
                        "retrieved via live search query."
                    ),
                    source="DuckDuckGo",
                )
            )

        return results
