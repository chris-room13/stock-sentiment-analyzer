# src/sentiment.py

import ssl
import certifi
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import sys
import os
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 1) SSL-Patch für nltk (verhindert Hänger beim Download auf macOS)
# ─────────────────────────────────────────────────────────────────────────────
ssl._create_default_https_context = lambda: ssl.create_default_https_context(
    cafile=certifi.where()
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) VADER-Initialisierung
# ─────────────────────────────────────────────────────────────────────────────
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
    sia = SentimentIntensityAnalyzer()
    print("✅ sentiment.py: VADER lexicon gefunden und SentimentIntensityAnalyzer initialisiert")
except Exception as e:
    print(f"❌ sentiment.py: VADER konnte nicht initialisiert werden: {e}", file=sys.stderr)
    sia = None

def analyze_sentiment(text: str) -> dict:
    """
    VADER-basierte Sentiment-Analyse.
    Gibt zurück: {"compound": <float>, "label": <"positive"/"neutral"/"negative">, "details": {...}}
    Falls VADER nicht initialisiert wurde, liefert nur neutral.
    """
    if sia is None:
        return {"compound": 0.0, "label": "neutral", "details": {}}

    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {"compound": compound, "label": label, "details": scores}


# ─────────────────────────────────────────────────────────────────────────────
# 3) BERT-Pipeline
# ─────────────────────────────────────────────────────────────────────────────
_LABEL_MAP = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive"
}

_bert_classifier = None

def analyze_sentiment_bert(text: str) -> dict:
    """
    BERT-basierte Sentiment-Analyse (lokal) mit cardiffnlp/twitter-roberta-base-sentiment.
    Rückgabe: {"label": <"positive"/"neutral"/"negative">, "score": <float>, "raw": <original>}.
    """
    global _bert_classifier

    # Lazy-Initialisierung der BERT-Pipeline (lokal)
    if _bert_classifier is None:
        try:
            print("🔍 sentiment.py: Initializing local Twitter-RoBERTa pipeline...")
            _bert_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
            )
            print("✅ sentiment.py: Lokale BERT pipeline (Twitter-RoBERTa) initialisiert")
        except Exception as e:
            print(f"❌ sentiment.py: Lokale BERT-Initialisierung fehlgeschlagen: {e}", file=sys.stderr)
            return {"label": "neutral", "score": 0.0, "raw": None}

    snippet = text[:512]
    try:
        result = _bert_classifier(snippet)[0]
    except Exception as e:
        print(f"❌ sentiment.py: Lokale Pipeline-Lauf fehlgeschlagen: {e}", file=sys.stderr)
        return {"label": "neutral", "score": 0.0, "raw": None}

    raw_label = result.get("label", "").lower()  # z.B. "label_2"
    raw_score = float(result.get("score", 0.0))
    mapped_label = _LABEL_MAP.get(raw_label, "neutral")

    return {"label": mapped_label, "score": raw_score, "raw": result}


# ─────────────────────────────────────────────────────────────────────────────
# 4) Zero-Shot-Klassifikation mit facebook/bart-large-mnli via HF Inference API
#    Caching mit @st.cache_data, damit jeder exakte Text nur einmal abgefragt wird
# ─────────────────────────────────────────────────────────────────────────────

import requests
import sys
import streamlit as st

# HF_API Token (bereits vorher definiert)
HF_TOKEN = "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"

@st.cache_data(show_spinner=False)
def analyze_sentiment_zeroshot(text: str) -> dict:
    """
    Zero-Shot-Sentimentklassifikation via Hugging Face Inference API
    mit 'facebook/bart-large-mnli'. Wir übergeben drei Kandidatenlabels:
      ["positive", "neutral", "negative"].

    Rückgabe:
      {
        "label":    "positive" | "neutral" | "negative",
        "score":    <signed float im Bereich –1.0 … +1.0>,   # Neu: ±Confidence
        "raw":      <komplette JSON-Antwort von HF>
      }

    Mapping-Logik:
      - Wenn das Top-Label "positive" ist, liefern wir +<Confidence>,
      - Wenn "negative", liefern wir –<Confidence>,
      - Wenn "neutral", liefern wir 0.0.
    """
    # 1) Falls kein Token gesetzt ist, einfach neutral zurückgeben
    if not HF_TOKEN:
        return {"label": "neutral", "score": 0.0, "raw": None}

    # 2) Baue die Anfrage-Payload
    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": ["positive", "neutral", "negative"],
            "multi_label": False
        }
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json"
    }

    # 3) Anfrage an HF Zero-Shot-API
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Zero-Shot-Inference-Fehler: {e}", file=sys.stderr)
        # Bei Fehlern geben wir neutral zurück
        return {"label": "neutral", "score": 0.0, "raw": None}

    res_json = r.json()
    # Beispiel-Antwort:
    # {
    #   "sequence": "...",
    #   "labels":   ["neutral","positive","negative"],
    #   "scores":   [0.75,       0.20,      0.05]
    # }

    labels = res_json.get("labels", [])
    scores = res_json.get("scores", [])

    if not labels or not scores:
        return {"label": "neutral", "score": 0.0, "raw": res_json}

    # 4) Top-Label + Confidence
    top_label = labels[0].lower()       # z.B. "positive", "neutral" oder "negative"
    top_score = float(scores[0])        # z.B. 0.82

    # 5) In einen signed Score umwandeln
    if top_label == "positive":
        signed_score = top_score        # bleibt positiv
    elif top_label == "negative":
        signed_score = -top_score       # wird negativ
    else:  # "neutral"
        signed_score = 0.0

    return {"label": top_label, "score": signed_score, "raw": res_json}
