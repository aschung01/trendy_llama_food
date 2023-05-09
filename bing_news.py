"""
Bing news API integration
"""

import os
import requests

SEARCH_TERM = "Microsoft"
SEARCH_URL = "https://api.bing.microsoft.com/v7.0/news/search"

req_headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_NEWS_SUBSCRIPTION_KEY"]}

def get_news(page):
    """Get news from Bing News API."""
    response = requests.get(
        SEARCH_URL,
        headers=req_headers,
        params={
            "q": SEARCH_TERM,
            "textFormat": "HTML",
            "count": 100,
            "offset": 100 * page,
        },
        timeout=1000,
    )
    response.raise_for_status()
    search_results = response.json()

    return [
        (
            article["description"],
            article["name"],
            article["datePublished"].split("T")[0],
            article["provider"][0]["name"],
        )
        for article in search_results["value"]
    ]
