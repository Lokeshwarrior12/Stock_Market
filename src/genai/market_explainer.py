# src/genai/market_explainer.py

import os
import numpy as np
import pandas as pd
from src.utils.config import CONFIG

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def generate_prompt(symbol, current_price, predicted_price, indicators, fundamentals):
    """
    Compose a structured prompt for the GenAI model
    """
    price_change = ((predicted_price - current_price) / current_price) * 100
    direction = "rise" if predicted_price > current_price else "fall"

    prompt = f"""
You are an expert stock market analyst.
Analyze the stock {symbol} based on the given data and provide a concise summary.

Current Price: ${current_price:.2f}
Predicted Price (next period): ${predicted_price:.2f}
Expected Change: {price_change:.2f}% ({direction})

Technical Indicators:
{indicators}

Fundamental Summary:
{fundamentals}

Explain why this movement may occur, referencing both technical signals (momentum, volatility, support/resistance)
and fundamental aspects (growth, profitability, leverage, sentiment).

Keep tone analytical, not speculative. Avoid financial advice language.
"""
    return prompt


def generate_analysis(prompt: str):
    """
    Call OpenAI or local model to generate the analysis text.
    """
    genai_cfg = CONFIG["genai"]

    if genai_cfg["use_openai"]:
        if OpenAI is None:
            raise ImportError("Please install openai package: pip install openai")
        client = OpenAI(api_key=os.getenv(genai_cfg["api_key_env"]))

        response = client.chat.completions.create(
            model=genai_cfg["openai_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    else:
        # Placeholder for local model
        return "[Local model analysis placeholder: Use Llama or Hugging Face here.]"


if __name__ == "__main__":
    symbol = CONFIG["stock_symbol"]

    # Example data (you can connect this to your model output directly)
    current_price = 180.25
    predicted_price = 192.80
    indicators = {
        "RSI": 68.4,
        "MACD": "Positive crossover",
        "Volatility (ATR)": 2.6,
        "EMA Trend": "Short EMA above Long EMA"
    }
    fundamentals = {
        "EPS": 6.25,
        "P/E": 27.1,
        "ROE": 45.8,
        "Debt/Equity": 1.2,
        "Net Profit Margin": 22.5
    }

    prompt = generate_prompt(symbol, current_price, predicted_price, indicators, fundamentals)
    analysis = generate_analysis(prompt)

    print("--------- AI MARKET ANALYSIS ---------")
    print(analysis)
    print("-------------------------------------")
