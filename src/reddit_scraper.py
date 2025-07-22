# src/reddit_scraper.py

import re
import datetime
import yfinance as yf
import praw
import config

reddit = praw.Reddit(
    client_id=config.REDDIT_CLIENT_ID,
    client_secret=config.REDDIT_CLIENT_SECRET,
    user_agent=config.REDDIT_USER_AGENT,
)

def fetch_reddit_posts(ticker, subreddits=config.DEFAULT_SUBREDDITS, limit=config.POST_LIMIT, days=7):
    """
    Holt Reddit-Posts, die in den letzten `days` Tagen erschienen sind,
    sucht in den Subreddits nach:
      • ticker (z.B. "BA")
      • $ticker (z.B. "$BA")
      • Firmenname (z.B. "Boeing")
    und bricht ab, sobald die PRAW-Such-API nur noch Ergebnisse > 1 Monat alt liefert.
    """

    # 0) Basis-Ticker ohne Exchange-Suffix ("RHM.DE" → "RHM")
    base = ticker.split(".")[0].upper()

    # 1) Firmenname via yfinance auflösen
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
        company = name.split(",")[0].strip()
    except Exception:
        company = None

    # 2) Baue Such-Queries + Regex Patterns
    queries = [base, f"${base}"]
    if company and company.lower() != base.lower():
        queries.append(company)

    patterns = [
        re.compile(rf"\b{re.escape(base)}\b", re.IGNORECASE),
        re.compile(rf"\${re.escape(base)}\b", re.IGNORECASE),
    ]
    if company:
        patterns.append(re.compile(rf"\b{re.escape(company)}\b", re.IGNORECASE))

    # 3) Date-cutoff für exaktes Tage-Fenster
    now    = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(days=days)

    posts, seen = [], set()

    for sub in subreddits:
        sr = reddit.subreddit(sub)

        # Durchlaufe jede Query, verwende PRAW-Search mit time_filter="month"
        for q in queries:
            for submission in sr.search(
                q,
                sort="new",
                time_filter="month",  # hol nur den letzten Monat
                limit=limit
            ):
                if submission.id in seen:
                    continue

                created = datetime.datetime.utcfromtimestamp(submission.created_utc)
                # wenn älter als unser Tage-Fenster, überspringen
                if created < cutoff:
                    continue

                text = f"{submission.title} {submission.selftext}"
                # exaktes Matching via Regex
                if not any(p.search(text) for p in patterns):
                    continue

                posts.append({
                    "id":           submission.id,
                    "subreddit":    sub,
                    "title":        submission.title,
                    "text":         submission.selftext,
                    "score":        submission.score,
                    "date":         created,
                    "url":          submission.url,
                    "num_comments": submission.num_comments
                })
                seen.add(submission.id)

    return posts
