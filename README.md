# 📈 Financial News Stock Sentiment Predictor

An NLP-powered Streamlit app that fetches **real-time financial news**, analyzes sentiment using **Groq (LLaMA-3.3-70B)**, and predicts **bullish / bearish / sideways** stock movement.

---

## 🚀 Features

- **Real-time news** via NewsAPI (last 7 days)
- **Live stock data** via yFinance (OHLCV + fundamentals)
- **LLM sentiment analysis** via Groq's LLaMA-3.3-70B
- Per-article sentiment: POSITIVE / NEGATIVE / NEUTRAL
- Overall sentiment score (−1.0 → +1.0)
- Predicted price movement with % estimate
- Interactive candlestick chart, gauge, pie & bar charts
- Dark-themed professional Streamlit UI

---

## 🛠️ Setup

### 1. Clone / Download the project
```bash
# Place all files in a folder, then:
cd financial_sentiment
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
NEWS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

**Get API keys:**
- **Groq** (free): https://console.groq.com → API Keys
- **NewsAPI** (free tier: 100 req/day): https://newsapi.org → Get API Key

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
financial_sentiment/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # Your API keys (create from .env.example)
├── .env.example        # API key template
└── README.md           # This file
```

---

## 🧠 How It Works

1. **User inputs** a stock ticker (e.g. AAPL) and company name
2. **yFinance** fetches OHLCV price data and stock fundamentals
3. **NewsAPI** fetches the latest 5–20 news articles mentioning the ticker/company
4. **Groq (LLaMA-3.3-70B)** receives all article headlines + descriptions and performs:
   - Per-article sentiment classification
   - Impact score (−1 to +1)
   - Overall sentiment score
   - Predicted price movement (BULLISH/BEARISH/SIDEWAYS)
   - Estimated % change over 1–3 days
   - Key themes and risk factors
5. **Streamlit** renders everything in an interactive dark-themed dashboard

---

## ⚠️ Disclaimer

This tool is built for **educational and NLP research purposes only**.  
It is **not financial advice**. Do not make investment decisions based on its output.
