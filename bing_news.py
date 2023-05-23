"""
Bing news API integration
"""

import os
import requests
from dotenv import load_dotenv
import utils
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import tqdm

load_dotenv()

SEARCH_TERM = ""
SEARCH_URL = "https://api.bing.microsoft.com/v7.0/news/search"
market_codes = [
    "es-AR",
    "en-AU",
    "de-AT",
    "nl-BE",
    "fr-BE",
    "pt-BR",
    "en-CA",
    "fr-CA",
    "es-CL",
    "da-DK",
    "fi-FI",
    "fr-FR",
    "de-DE",
    "zh-HK",
    "en-IN",
    "en-ID",
    "it-IT",
    "ja-JP",
    "ko-KR",
    "en-MY",
    "es-MX",
    "nl-NL",
    "en-NZ",
    "no-NO",
    "zh-CN",
    "pl-PL",
    "en-PH",
    "ru-RU",
    "en-ZA",
    "es-ES",
    "sv-SE",
    "fr-CH",
    "de-CH",
    "zh-TW",
    "tr-TR",
    "en-GB",
    "en-US",
    "es-US",
]
freshness_options = ["Day", "Week", "Month"]

req_headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_NEWS_SUBSCRIPTION_KEY"]}


def get_search_results(mkt, freshness):
    response = requests.get(
        SEARCH_URL,
        headers=req_headers,
        params={
            "q": SEARCH_TERM,
            "mkt": mkt,
            "textFormat": "HTML",
            "count": 100,
            "freshness": freshness,
        },
        timeout=5000,
    )
    response.raise_for_status()
    return response.json()


async def get_news(output_dir: str, request_new: bool = False):
    """Get news from Bing News API."""
    news_data = utils.load_jsonl(os.path.join(output_dir, "news_data.jsonl"))
    return_data = [
        (
            article["description"],
            article["name"],
            article["datePublished"].split("T")[0],
            article["provider"][0]["name"],
        )
        for article in news_data
    ]
    news_names = list(list(zip(*return_data))[1]) if len(return_data) > 0 else []

    # Iterate for all possible matches of market codes and freshness options
    if request_new:
        progress_bar = tqdm.tqdm(total=len(market_codes) * len(freshness_options))
        for mkt in market_codes:
            for freshness in freshness_options:
                search_results = get_search_results(mkt, freshness)
                for article in search_results["value"]:
                    if article["name"] not in news_names:
                        return_data.append(
                            (
                                article["description"],
                                article["name"],
                                article["datePublished"].split("T")[0],
                                article["provider"][0]["name"],
                            )
                        )
                        utils.append_to_jsonl(
                            article,
                            os.path.join(output_dir, "news_data.jsonl"),
                        )
                        news_names.append(article["name"])
                progress_bar.update(1)

    return return_data


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./")
    args = parser.parse_args()

    # run script
    asyncio.run(
        get_news(
            output_dir=args.output_dir,
        )
    )
