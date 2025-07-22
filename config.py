# config.py

# Welche Subreddits sollen durchsucht werden?
DEFAULT_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "finance",  "financialindependence", "StockMarket"]

# Wie viele Beiträge sollen pro Subreddit geladen werden?
POST_LIMIT = 1000

# Wie viele Tage zurück sollen Posts berücksichtigt werden?
LOOKBACK_DAYS = 30

# Optional: Begriffe, die ignoriert werden sollen
MIN_TEXT_LENGTH = 20

from os import getenv
from dotenv import load_dotenv

load_dotenv() 

OPENAI_API_KEY       = getenv("OPENAI_API_KEY")
NEWS_API_KEY         = getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID     = getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT    = getenv("REDDIT_USER_AGENT")
