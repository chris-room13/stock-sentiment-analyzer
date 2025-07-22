"""SentimentIQ – Streamlit app."""

from __future__ import annotations

# Standard library
import collections
import re
from typing import Tuple

# Third‑party
import matplotlib.pyplot as plt
import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import STOPWORDS, WordCloud
import yfinance as yf

# Local modules
import config
from src.analyst_sentiment import fetch_analyst_sentiment
from src.news_sentiment import aggregate_daily_news_sentiment, fetch_news_for_ticker
from src.reddit_scraper import fetch_reddit_posts
from src.sentiment import analyze_sentiment, analyze_sentiment_zeroshot

# ---------------------------------------------------------------------------
openai.api_key = config.OPENAI_API_KEY
COLORS = {"pos": "forestgreen", "neg": "firebrick", "neu": "gold"}
GAUGE_STEPS = 40


# ---------------------------------------------------------------------------
def get_descriptor(score: float) -> Tuple[str, str]:
    if score >= 0.66:
        return "Very Bullish", COLORS["pos"]
    if score >= 0.33:
        return "Bullish", COLORS["pos"]
    if score > 0.05:
        return "Slightly Bullish", COLORS["pos"]
    if score >= -0.05:
        return "Neutral", COLORS["neu"]
    if score > -0.33:
        return "Slightly Bearish", COLORS["neg"]
    if score > -0.66:
        return "Bearish", COLORS["neg"]
    return "Very Bearish", COLORS["neg"]


def make_gauge(score: float, title: str) -> go.Figure:
    step = 2 / GAUGE_STEPS
    steps_cfg = [
        {
            "range": [-1 + i * step, -1 + (i + 1) * step],
            "color": f"rgb({int(255 * (1 - i / (GAUGE_STEPS - 1)))}, {int(255 * (i / (GAUGE_STEPS - 1)))}, 0)",
        }
        for i in range(GAUGE_STEPS)
    ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 18}},
            gauge={
                "axis": {"range": [-1, 1], "tick0": -1, "dtick": 0.5},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": steps_cfg,
                "threshold": {"line": {"color": "black", "width": 4}, "value": score},
            },
            number={"valueformat": ".2f"},
        )
    )
    fig.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=300)
    return fig


@st.cache_data(show_spinner=False)
def load_reddit_sentiment(ticker: str, days: int, analyzer_key: str) -> pd.DataFrame:
    posts = fetch_reddit_posts(ticker, days=days)
    rows = []
    for p in posts:
        text = f"{p['title']} {p['text']}"
        if analyzer_key == "VADER (fast)":
            s = analyze_sentiment(text)
            score, label = s["compound"], s["label"].upper()
        else:
            s = analyze_sentiment_zeroshot(text)
            score, label = s["score"], s["label"].upper()
        rows.append(
            dict(
                Date=p["date"].strftime("%Y-%m-%d"),
                Subreddit=p["subreddit"],
                Title=p["title"],
                Text=p["text"][:300],
                Sentiment=label,
                Score=round(score, 3),
                Link=f"[View]({p['url']})",
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config("SentimentIQ", layout="wide", initial_sidebar_state="collapsed")
    st.title("📈 SentimentIQ – Stock Sentiment Analyzer")

    if "analyzed" not in st.session_state:
        st.session_state.update(analyzed=False, data_ready=False)

    ticker = st.text_input("Stock ticker").upper().strip()
    if not ticker:
        st.stop()

    lookback_map = {"1 Day": 1, "1 Week": 7, "2 Weeks": 14, "1 Month": 30}
    period = st.selectbox("Look‑back period", list(lookback_map))
    days = lookback_map[period]

    analyzer_choice = st.radio(
        "Sentiment model",
        ["⚡️ VADER (fast)", "🤖 Zero‑Shot LLM (accurate)"],
        horizontal=True,
    )
    analyzer_key = "VADER (fast)" if analyzer_choice.startswith("⚡️") else "Zero‑Shot (HF)"

    if st.button("Analyze"):
        st.session_state.update(analyzed=True, data_ready=False)

    if not st.session_state.analyzed:
        st.stop()

    # ----------------------------------------------------------------------
    # Fetch & compute once
    # ----------------------------------------------------------------------
    if not st.session_state.data_ready:
        with st.spinner("Crunching numbers…"):
            df_reddit = load_reddit_sentiment(ticker, days, analyzer_key)
            reddit_avg = df_reddit.Score.mean() if not df_reddit.empty else 0.0
            reddit_counts = df_reddit.Sentiment.value_counts().to_dict()

            df_analyst = fetch_analyst_sentiment(ticker)
            if not df_analyst.empty:
                raw_counts = df_analyst.loc[0, ["strongBuy", "buy", "hold", "sell", "strongSell"]].to_dict()
                analyst_scaled = (float(df_analyst.loc[0, "Score"]) - 0.5) * 2
            else:
                raw_counts, analyst_scaled = {}, 0.0

            df_news = fetch_news_for_ticker(ticker, days, analyzer_key)
            if not df_news.empty:
                daily = aggregate_daily_news_sentiment(df_news)
                news_avg = daily.avg_news_sentiment.mean()
                classify = lambda x: "POSITIVE" if x >= 0.05 else ("NEGATIVE" if x <= -0.05 else "NEUTRAL")
                news_counts = df_news.sentiment_score.apply(classify).value_counts().to_dict()
            else:
                news_avg, news_counts = 0.0, {}

            st.session_state.update(
                data_ready=True,
                df_reddit=df_reddit,
                reddit_avg=reddit_avg,
                reddit_counts=reddit_counts,
                df_news=df_news,
                news_avg=news_avg,
                news_counts=news_counts,
                raw_counts=raw_counts,
                analyst_scaled=analyst_scaled,
            )

    # Shortcuts
    df_reddit = st.session_state.df_reddit
    df_news = st.session_state.df_news
    reddit_avg = st.session_state.reddit_avg
    news_avg = st.session_state.news_avg
    analyst_scaled = st.session_state.analyst_scaled

    # ----------------------------------------------------------------------
    # Gauges
    # ----------------------------------------------------------------------
    st.markdown("#### Sentiment Overview")
    cols = st.columns(3)
    views = [
        ("Reddit", reddit_avg, st.session_state.reddit_counts),
        ("Analysts", analyst_scaled, st.session_state.raw_counts),
        ("News", news_avg, st.session_state.news_counts),
    ]
    for col, (title, score, _) in zip(cols, views):
        with col:
            st.plotly_chart(make_gauge(score, f"{title} Sentiment"), use_container_width=True)
            desc, clr = get_descriptor(score)
            st.markdown(f"<div style='text-align:center;color:{clr};font-weight:bold'>{desc}</div>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # Detailed Pie‑Charts
    # ----------------------------------------------------------------------
    with st.expander("Details"):
        pie_cols = st.columns(3)
        pie_cfg = [
            (st.session_state.reddit_counts, "Reddit Breakdown"),
            (st.session_state.raw_counts, "Analyst Recommendations"),
            (st.session_state.news_counts, "News Breakdown"),
        ]
        for pc, (counts, title) in zip(pie_cols, pie_cfg):
            with pc:
                st.markdown(f"**{title}**")
                if counts:
                    fig = px.pie(names=list(counts), values=list(counts.values()))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No data.")

    # ----------------------------------------------------------------------
    # Demo cards
    # ----------------------------------------------------------------------
    with st.expander("Example Posts"):
        yf_info = yf.Ticker(ticker).info
        company = (yf_info.get("shortName") or yf_info.get("longName") or "").split(",")[0]

        def filter_candidates(df, headline_col, text_col) -> pd.DataFrame:
            mask = df[headline_col].str.contains(ticker, case=False, na=False)
            if company:
                mask |= df[text_col].str.contains(company, case=False, na=False)
            cand = df[mask].copy()
            if cand.empty:
                cand = df.copy()
            cand["abs_score"] = cand["Score"].abs() if "Score" in cand else cand.sentiment_score.abs()
            return cand.sort_values("abs_score", ascending=False).reset_index(drop=True)

        reddit_cand = filter_candidates(df_reddit, "Title", "Text") if not df_reddit.empty else pd.DataFrame()
        news_cand = filter_candidates(df_news, "headline", "description") if not df_news.empty else pd.DataFrame()

        st.session_state.setdefault("idx_reddit", 0)
        st.session_state.setdefault("idx_news", 0)

        if not reddit_cand.empty:
            r = reddit_cand.loc[st.session_state.idx_reddit % len(reddit_cand)]
            score_col = COLORS["pos"] if r.Score > 0 else COLORS["neg"] if r.Score < 0 else COLORS["neu"]
            st.markdown(f"##### Reddit\n**{r.Title}**\n{r.Text}\n\nScore: `{r.Score:+.2f}`", unsafe_allow_html=True)
            if st.button("Next Reddit"):
                st.session_state.idx_reddit += 1

        if not news_cand.empty:
            n = news_cand.loc[st.session_state.idx_news % len(news_cand)]
            score_col = COLORS["pos"] if n.sentiment_score > 0 else COLORS["neg"] if n.sentiment_score < 0 else COLORS["neu"]
            st.markdown(f"##### News\n**{n.headline}**\n{n.description}\n\nScore: `{n.sentiment_score:+.2f}`", unsafe_allow_html=True)
            if st.button("Next News"):
                st.session_state.idx_news += 1

    # ----------------------------------------------------------------------
    # Word‑Cloud + Insight
    # ----------------------------------------------------------------------
    with st.expander("Word‑Cloud & Insight"):
        if st.button("Generate Word‑Clouds"):
            def wc_freq(text: str) -> dict[str, int]:
                words = re.findall(r"\b[a-z]{3,}\b", text.lower())
                words = [w for w in words if w not in STOPWORDS]
                return dict(collections.Counter(words).most_common(50))

            freq_reddit = wc_freq(" ".join(df_reddit.Title.fillna("") + " " + df_reddit.Text.fillna("")))
            freq_news = wc_freq(" ".join(df_news.headline.fillna("") + " " + df_news.description.fillna("")))

            wc_r = WordCloud(width=300, height=200, background_color="white").generate_from_frequencies(freq_reddit)
            wc_n = WordCloud(width=300, height=200, background_color="white").generate_from_frequencies(freq_news)

            c1, c2 = st.columns(2)
            with c1:
                st.image(wc_r.to_image(), caption="Reddit Word Cloud", use_column_width=True)
            with c2:
                st.image(wc_n.to_image(), caption="News Word Cloud", use_column_width=True)

            # GPT insight
            prompt = (
                f"Average sentiment for {ticker} – Reddit: {reddit_avg:.2f}, News: {news_avg:.2f}. "
                "Explain key themes driving any difference in 3‑4 sentences."
            )
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.6,
            )
            st.markdown(resp.choices[0].message.content.strip())

    # ----------------------------------------------------------------------
    # Advice Form
    # ----------------------------------------------------------------------
    with st.expander("Investment Advice"):
        with st.form("advice"):
            w_r = st.slider("Weight Reddit", 0, 10, 5)
            w_a = st.slider("Weight Analyst", 0, 10, 5)
            w_n = st.slider("Weight News", 0, 10, 5)
            submitted = st.form_submit_button("Calculate")

        if submitted:
            total = max(w_r + w_a + w_n, 1)
            combined = (w_r * reddit_avg + w_a * analyst_scaled + w_n * news_avg) / total
            if combined >= 0.33:
                label, clr = "Buy", COLORS["pos"]
            elif combined <= -0.33:
                label, clr = "Sell", COLORS["neg"]
            else:
                label, clr = "Hold", COLORS["neu"]
            st.markdown(f"<h2 style='color:{clr}'>{label} {ticker}</h2>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # Price chart
    # ----------------------------------------------------------------------
    st.markdown("#### Price Chart")
    tf_map = {"1 Week": "7d", "2 Weeks": "14d", "1 Month": "30d", "1 Year": "1y"}
    tf_choice = st.selectbox("Timeframe", list(tf_map), index=2)
    hist = yf.Ticker(ticker).history(period=tf_map[tf_choice])
    if hist.empty:
        st.warning("No price data.")
    else:
        fig = px.line(hist.reset_index(), x="Date", y="Close", title=f"{ticker} Close ({tf_choice})")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Key Metrics"):
            info = yf.Ticker(ticker).info

            def fmt(x) -> str:
                try:
                    x = float(x)
                    if x >= 1e12:
                        return f"{x / 1e12:,.2f} T"
                    if x >= 1e9:
                        return f"{x / 1e9:,.2f} B"
                    if x >= 1e6:
                        return f"{x / 1e6:,.2f} M"
                    return f"{x:,.2f}"
                except Exception:
                    return str(x)

            metrics = {
                "Market Cap": fmt(info.get("marketCap")),
                "Forward P/E": info.get("forwardPE"),
                "EPS (TTM)": info.get("trailingEps"),
                "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f} %" if info.get("dividendYield") else None,
                "52‑W High": fmt(info.get("fiftyTwoWeekHigh")),
                "52‑W Low": fmt(info.get("fiftyTwoWeekLow")),
                "Beta": info.get("beta"),
            }
            for name, val in metrics.items():
                if val:
                    st.metric(name, val)

    # ----------------------------------------------------------------------
    # Downloads
    # ----------------------------------------------------------------------
    if st.session_state.data_ready:
        c1, c2 = st.columns(2)
        with c1:
            if not df_reddit.empty:
                st.download_button(
                    "Download Reddit CSV",
                    df_reddit.to_csv(index=False).encode(),
                    f"{ticker}_reddit.csv",
                    "text/csv",
                )
        with c2:
            if not df_news.empty:
                st.download_button(
                    "Download News CSV",
                    df_news.to_csv(index=False).encode(),
                    f"{ticker}_news.csv",
                    "text/csv",
                )


if __name__ == "__main__":
    main()
