"""
News fetching via the NewsAPI.

Retrieves top headlines filtered by topic keyword and country code.
Each topic is queried separately so results from different interests
don't crowd each other out.

API reference : https://newsapi.org/docs/endpoints/top-headlines
Free tier     : 100 requests / day, articles up to 1 month old.
Sign up at    : https://newsapi.org/register
"""

import os

import requests

NEWS_API_BASE = "https://newsapi.org/v2/top-headlines"

# Max articles fetched per topic to keep the report scannable
ARTICLES_PER_TOPIC = 5


def get_news_for_topics(
    topics: list[str],
    country: str = "us",
    api_key: str | None = None,
) -> list[dict]:
    """
    Fetch top headlines for each topic from NewsAPI.

    Args:
        topics:  List of keyword strings to query (e.g. ["AI", "finance"]).
        country: ISO 3166-1 alpha-2 country code for localised headlines.
        api_key: NewsAPI key. Falls back to the NEWS_API_KEY env variable.

    Returns
    -------
        List of article dicts with keys:
            title        (str)
            source       (str)  - publication name
            url          (str)
            description  (str)
            published_at (str)  - ISO-8601 datetime string

    Notes
    -----
        - Returns a single placeholder item when no API key is configured
          so the rest of the graph can still run without crashing.
        - Duplicate articles (same URL appearing under multiple topics) are
          deduplicated before returning.
    """
    key = api_key or os.getenv("NEWS_API_KEY", "")
    if not key:
        return [
            {
                "title": "News section unavailable - set NEWS_API_KEY to enable",
                "source": "",
                "url": "",
                "description": "",
                "published_at": "",
            }
        ]

    seen_urls: set[str] = set()
    articles: list[dict] = []

    for topic in topics:
        response = requests.get(
            NEWS_API_BASE,
            params={
                "q": topic,
                "country": country,
                "pageSize": ARTICLES_PER_TOPIC,
                "apiKey": key,
            },
            timeout=10,
        )
        response.raise_for_status()

        for article in response.json().get("articles", []):
            url = article.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(
                {
                    "title": article.get("title", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "url": url,
                    "description": article.get("description", ""),
                    "published_at": article.get("publishedAt", ""),
                }
            )

    return articles
