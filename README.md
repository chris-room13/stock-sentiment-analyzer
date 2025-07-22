# Stock Sentiment Analyzer (SentimentIQ)

**SentimentIQ** is a Streamlit app that analyzes real‑time market sentiment for any stock ticker using:

* **Reddit discussions** (via PRAW)
* **News articles** (via NewsAPI.org)
* **Analyst recommendations** (via Yahoo Finance’s `yfinance`)

<img width="1380" height="798" alt="Screenshot 2025-07-22 at 20 22 49" src="https://github.com/user-attachments/assets/eaa645fc-579e-4950-b3c6-2d0db979febf" />
---

## 🚀 Features

* **Multi‑source sentiment**: VADER (fast), BERT (optional), and zero‑shot LLM (accurate)
* **Visualizations**: Gauges, pie charts, word clouds, price charts
* **GPT‑powered insights**: Deeper analysis via OpenAI
* **Custom advice**: Weight each source to generate “Buy/Hold/Sell” suggestions
* **Exports**: Download CSVs of sentiment data

---

## 💾 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your‑username/stock-sentiment-analyzer.git
   cd stock-sentiment-analyzer
   ```
2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .\.venv\Scripts\activate  # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**

   * Copy `.env.example` to `.env` and fill in:

     ```dotenv
     OPENAI_API_KEY=...
     NEWS_API_KEY=...
     REDDIT_CLIENT_ID=...
     REDDIT_CLIENT_SECRET=...
     REDDIT_USER_AGENT=...
     ```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501) to explore.

---

## 🛠 Usage

1. **Enter a ticker** (e.g., `AAPL`, `TSLA`).
2. **Select look‑back period** (1 day, 1 week, 2 weeks, 1 month).
3. **Choose sentiment model** (fast vs. accurate).
4. **Analyze** to view gauges and charts.
5. **Expand sections** for breakdowns, word clouds, and advice.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

