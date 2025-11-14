# src/data/fetch_news_sentiment.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.utils.config import CONFIG


def fetch_news_articles(symbol: str, api_key: str, days_back: int, api_url: str):
    """
    Fetch recent news articles mentioning the stock symbol.
    """
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    params = {
        "q": symbol,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 100,
        "apiKey": api_key
    }

    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching news: {response.text}")

    data = response.json().get("articles", [])
    articles = pd.DataFrame([{
        "date": a["publishedAt"][:10],
        "title": a["title"],
        "description": a["description"],
        "source": a["source"]["name"]
    } for a in data if a.get("title")])

    return articles


def analyze_sentiment(articles: pd.DataFrame, model_name: str):
    """
    Analyze sentiment using FinBERT and add sentiment score.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sentiments = []
    for _, row in tqdm(articles.iterrows(), total=len(articles), desc="Analyzing Sentiment"):
        text = row["title"] or ""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs, dim=1).item()
        score = probs[0, label].item()

        sentiment_label = ["Negative", "Neutral", "Positive"][label]
        sentiments.append((sentiment_label, score))

    articles["sentiment_label"], articles["sentiment_score"] = zip(*sentiments)
    return articles


def aggregate_daily_sentiment(articles: pd.DataFrame):
    """
    Aggregate sentiment scores by day for feature use.
    """
    df_daily = (articles.groupby("date")
                        .agg(avg_sentiment_score=("sentiment_score", "mean"),
                             pos_ratio=("sentiment_label", lambda x: (x == "Positive").mean()),
                             neg_ratio=("sentiment_label", lambda x: (x == "Negative").mean()))
                        .reset_index())
    return df_daily


def save_sentiment(df_daily: pd.DataFrame, symbol: str):
    path = CONFIG["save_paths"]["raw_data"]
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"{symbol}_sentiment.csv")
    df_daily.to_csv(out_path, index=False)
    print(f"[SUCCESS] Sentiment data saved to {out_path}")


if __name__ == "__main__":
    cfg = CONFIG["sentiment"]
    symbol = cfg["symbol"]

    print(f"[INFO] Fetching news for {symbol}...")
    articles = fetch_news_articles(symbol, cfg["api_key"], cfg["days_back"], cfg["api_url"])
    print(f"[INFO] {len(articles)} articles fetched.")

    print("[INFO] Analyzing sentiment with FinBERT...")
    articles = analyze_sentiment(articles, cfg["model_name"])

    print("[INFO] Aggregating daily sentiment...")
    df_daily = aggregate_daily_sentiment(articles)

    save_sentiment(df_daily, symbol)
    print("[SUCCESS] Sentiment pipeline complete ✅")
