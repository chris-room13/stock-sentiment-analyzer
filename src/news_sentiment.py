# src/news_sentiment.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import config

from .sentiment import (
    analyze_sentiment,
    analyze_sentiment_bert,
    analyze_sentiment_zeroshot,
)

# 0) NewsAPI.org Configuration
NEWS_API_KEY = config.NEWS_API_KEY
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"


def fetch_news_for_ticker(
    ticker: str,
    days: int,
    analyzer_type: str,
    page_size: int = 100
) -> pd.DataFrame:
    """
    1) Hits NewsAPI.org to fetch up to `page_size` English‐language articles
       that mention `ticker` OR the resolved company name in the last `days` days.
    2) Runs the chosen sentiment analyzer on each article’s "title + description".
    3) Returns a DataFrame with columns:
       ["publishedAt", "source", "headline", "description", "url", "sentiment_score"].
    """

    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY is not set in config.py or .env")

    # 1a) Compute the ISO date string for "from=..." parameter
    today_utc = datetime.utcnow().date()
    from_date = (today_utc - timedelta(days=days)).isoformat()  # YYYY-MM-DD

    # 1b) Resolve the human-readable company name via yfinance
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
        company_name = name.split(",")[0].strip()
    except Exception:
        company_name = None

    # 1c) Build the NewsAPI "q" parameter to search for ticker OR company name
    query_terms = [ticker]
    if company_name and company_name.lower() != ticker.lower():
        query_terms.append(company_name)
    q_string = " OR ".join(query_terms)

    params = {
        "q":        q_string,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey":   NEWS_API_KEY,
    }

    # 1d) Perform the HTTP GET
    resp = requests.get(NEWS_API_ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    articles = payload.get("articles", [])

    # 1e) If no articles, return empty DataFrame with correct columns
    if not articles:
        return pd.DataFrame(columns=[
            "publishedAt",
            "source",
            "headline",
            "description",
            "url",
            "sentiment_score",
        ])

    # 1f) Loop through articles, analyze sentiment, build rows
    rows = []
    for art in articles:
        title = art.get("title") or ""
        desc  = art.get("description") or ""
        combined_text = f"{title} {desc}".strip()

        if analyzer_type == "VADER (fast)":
            sent = analyze_sentiment(combined_text)
            score = sent.get("compound", 0.0)

        elif analyzer_type == "BERT (accurate)":
            sent = analyze_sentiment_bert(combined_text)
            score = sent.get("score", 0.0)

        else:  # Zero‑Shot (HF)
            sent = analyze_sentiment_zeroshot(combined_text)
            score = sent.get("score", 0.0)

        rows.append({
            "publishedAt":     art.get("publishedAt"),
            "source":          art.get("source", {}).get("name", ""),
            "headline":        title,
            "description":     desc,
            "url":             art.get("url", ""),
            "sentiment_score": score,
        })

    # 1g) Convert to DataFrame & parse dates
    df = pd.DataFrame(rows)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
    return df


def aggregate_daily_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the DataFrame returned by fetch_news_for_ticker by UTC date
    and computes the average sentiment_score per day.
    """
    if news_df.empty:
        return pd.DataFrame(columns=["date", "avg_news_sentiment"])

    news_df["date"] = news_df["publishedAt"].dt.date
    daily = (
        news_df
        .groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_score": "avg_news_sentiment"})
    )
    return daily
