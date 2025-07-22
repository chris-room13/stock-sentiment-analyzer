# src/analyst_sentiment.py

import yfinance as yf
import pandas as pd
from typing import Tuple


def _compute_weighted_score(counts: pd.Series) -> float:
    """
    Berechnet aus einer Series mit den fünf Count-Werten
    ["strongBuy", "buy", "hold", "sell", "strongSell"] einen
    Score zwischen 0.0 und 1.0:

      strongBuy  → Gewicht 1.0
      buy        → Gewicht 0.8
      hold       → Gewicht 0.5
      sell       → Gewicht 0.2
      strongSell → Gewicht 0.0

    score = (∑ count_i * weight_i) / total_counts
    """
    sb = int(counts.get("strongBuy", 0))
    b  = int(counts.get("buy", 0))
    h  = int(counts.get("hold", 0))
    s  = int(counts.get("sell", 0))
    ss = int(counts.get("strongSell", 0))

    total = sb + b + h + s + ss
    if total == 0:
        return 0.5  # kein Analysten‐Input → neutral

    weighted_sum = sb * 1.0 + b * 0.8 + h * 0.5 + s * 0.2 + ss * 0.0
    return weighted_sum / total


def _map_score_to_label(score: float) -> str:
    """
    Wandelt einen numerischen Score [0.0–1.0] in ein Label um:
      - score >= 0.6 → "positive"
      - 0.4 <= score < 0.6 → "neutral"
      - score < 0.4 → "negative"
    """
    if score >= 0.6:
        return "positive"
    if score < 0.4:
        return "negative"
    return "neutral"


def fetch_analyst_sentiment(ticker_symbol: str) -> pd.DataFrame:
    """
    Holt die aggregierten Analysten‐Empfehlungen von YFinance (counts pro Periode).
    Liefert für die aktuellste Periode (z. B. "0m") genau eine Zeile mit:
      - Period      (String, z. B. "0m")
      - strongBuy   (int)
      - buy         (int)
      - hold        (int)
      - sell        (int)
      - strongSell  (int)
      - Score       (float zwischen 0.0 und 1.0)
      - Sentiment   (String, einer von "positive"/"neutral"/"negative")
    Falls YFinance keine `recommendations` zurückliefert oder das Format nicht passt,
    gibt es ein leeres DataFrame mit genau diesen Spalten.
    """
    # 1) Rufe YFinance‐Ticker‐Objekt ab
    ticker = yf.Ticker(ticker_symbol)

    # 2) Hole die aggregierten Recommendation‐Zahlen (DataFrame indexed by "period")
    df_reco = ticker.recommendations

    # 3) Wenn kein DataFrame oder leer, gib leeres Ergebnis‐Schema zurück
    if df_reco is None or df_reco.empty:
        return pd.DataFrame(
            columns=[
                "Period", "strongBuy", "buy", "hold", "sell", "strongSell", "Score", "Sentiment"
            ]
        )

    # 4) YFinance liefert per default Index=period (z. B. "0m", "-1m", "-2m", …)
    #    und Spalten ["strongBuy", "buy", "hold", "sell", "strongSell"].
    expected_cols = {"strongBuy", "buy", "hold", "sell", "strongSell"}
    if not expected_cols.issubset(set(df_reco.columns)):
        # Wenn das Format anders ist, liefere ein leeres DataFrame zurück
        return pd.DataFrame(
            columns=[
                "Period", "strongBuy", "buy", "hold", "sell", "strongSell", "Score", "Sentiment"
            ]
        )

    # 5) Wähle nur die erste (aktuellste) Zeile – das ist meistens Index "0m"
    period_label = df_reco.index[0]  # z. B. "0m"
    counts = df_reco.iloc[0][["strongBuy", "buy", "hold", "sell", "strongSell"]]

    # 6) Berechne den gewichteten Score (0.0–1.0)
    score = _compute_weighted_score(counts)

    # 7) Mappe Score → Label
    label = _map_score_to_label(score)

    # 8) Baue ein Ergebnis‐DataFrame mit genau einer Zeile
    result = pd.DataFrame([{
        "Period":     str(period_label),
        "strongBuy":  int(counts["strongBuy"]),
        "buy":        int(counts["buy"]),
        "hold":       int(counts["hold"]),
        "sell":       int(counts["sell"]),
        "strongSell": int(counts["strongSell"]),
        "Score":      round(score, 3),
        "Sentiment":  label
    }])

    return result
