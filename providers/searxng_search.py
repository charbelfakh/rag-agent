"""SearXNG JSON API client for web-fallback retrieval."""
import os

import requests


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Return title/url/snippet dicts from SearXNG; empty list on failure."""
    base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080").rstrip("/")
    try:
        response = requests.get(
            f"{base_url}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general",
            },
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in results[:num_results]
        ]
    except (requests.RequestException, ValueError, KeyError):
        return []
