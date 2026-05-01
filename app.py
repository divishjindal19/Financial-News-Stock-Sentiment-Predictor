import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
from groq import Groq
import json
import time

# ── Load env vars ──────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
NEWS_API_KEY  = os.getenv("NEWS_API_KEY")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c14;
    color: #e8eaf0;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 50%, #7b61ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #6b7a99;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Cards */
.metric-card {
    background: #0f1623;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00f5a0; }

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b7a99;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
}
.positive  { color: #00f5a0; }
.negative  { color: #ff4c6a; }
.neutral   { color: #f5c400; }

/* News cards */
.news-card {
    background: #0f1623;
    border: 1px solid #1e2d45;
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.news-card.positive { border-left-color: #00f5a0; }
.news-card.negative { border-left-color: #ff4c6a; }
.news-card.neutral  { border-left-color: #f5c400; }

.news-title {
    font-weight: 600;
    font-size: 0.92rem;
    margin-bottom: 0.3rem;
    line-height: 1.4;
}
.news-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #6b7a99;
    letter-spacing: 0.06em;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-left: 8px;
}
.badge-positive { background: #00f5a020; color: #00f5a0; border: 1px solid #00f5a040; }
.badge-negative { background: #ff4c6a20; color: #ff4c6a; border: 1px solid #ff4c6a40; }
.badge-neutral  { background: #f5c40020; color: #f5c400; border: 1px solid #f5c40040; }

/* Prediction box */
.prediction-box {
    background: linear-gradient(135deg, #0f1623 0%, #121c2e 100%);
    border: 1.5px solid #1e2d45;
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.prediction-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00f5a0, #00d9f5, #7b61ff);
}
.prediction-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #6b7a99;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.6rem;
}
.prediction-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.6;
    color: #d8dce8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] * { color: #c8cdd8 !important; }

.stButton>button {
    background: linear-gradient(135deg, #00f5a0, #00d9f5) !important;
    color: #080c14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.04em !important;
    width: 100%;
}
.stButton>button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: #1e2d45 !important; }

/* Score bar */
.score-bar-container { margin: 0.5rem 0 1rem; }
.score-bar-bg {
    background: #1e2d45;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def get_stock_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)
        return hist, stock.info
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return pd.DataFrame(), {}


def fetch_news(ticker: str, company_name: str, num_articles: int = 10):
    """Fetch latest news via NewsAPI."""
    if not NEWS_API_KEY:
        st.error("NEWS_API_KEY not found in .env")
        return []
    try:
        client   = NewsApiClient(api_key=NEWS_API_KEY)
        query    = f"{ticker} OR {company_name} stock"
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=num_articles,
            from_param=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        )
        return response.get("articles", [])
    except Exception as e:
        st.error(f"NewsAPI error: {e}")
        return []


def analyze_sentiment_with_groq(articles: list, ticker: str, stock_info: dict) -> dict:
    """
    Send article headlines + descriptions to Groq (llama-3.3-70b-versatile).
    Returns structured JSON with per-article sentiment + overall prediction.
    """
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env")
        return {}

    client = Groq(api_key=GROQ_API_KEY)

    # Build article list for the prompt
    article_texts = []
    for i, a in enumerate(articles[:10], 1):
        title = a.get("title", "")
        desc  = a.get("description", "") or ""
        article_texts.append(f"{i}. TITLE: {title}\n   DESC: {desc[:200]}")

    articles_str = "\n\n".join(article_texts)

    company_name = stock_info.get("longName", ticker)
    current_price = stock_info.get("currentPrice", stock_info.get("regularMarketPrice", "N/A"))
    sector = stock_info.get("sector", "N/A")

    system_prompt = """You are a professional financial NLP analyst specializing in stock sentiment analysis.
Your task is to analyze news articles and predict stock movement sentiment.
Always respond with valid JSON only — no markdown, no explanation outside JSON."""

    user_prompt = f"""Analyze the following {len(articles[:10])} news articles about {company_name} (ticker: {ticker}).
Current price: {current_price} | Sector: {sector}

ARTICLES:
{articles_str}

Return a JSON object with EXACTLY this structure:
{{
  "ticker": "{ticker}",
  "company": "{company_name}",
  "overall_sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "sentiment_score": <float -1.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "predicted_movement": "BULLISH" | "BEARISH" | "SIDEWAYS",
  "predicted_change_pct": <float, estimated % price change in next 1-3 days>,
  "key_themes": [<list of 3-5 key themes found in news>],
  "risk_factors": [<list of 1-3 risk factors>],
  "summary": "<2-3 sentence summary of overall news sentiment and what it means for the stock>",
  "articles": [
    {{
      "index": <int, 1-based>,
      "title": "<article title>",
      "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
      "impact_score": <float -1.0 to 1.0>,
      "reason": "<one sentence why this sentiment>"
    }}
    ... for each article
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        # Strip possible markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error from Groq: {e}")
        return {}
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return {}


def make_candlestick_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """Build a dark-themed candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        increasing_line_color="#00f5a0",
        decreasing_line_color="#ff4c6a",
        increasing_fillcolor="#00f5a030",
        decreasing_fillcolor="#ff4c6a30",
    )])

    # Volume bars
    colors = ["#00f5a0" if c >= o else "#ff4c6a"
              for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"],
        marker_color=colors, opacity=0.3,
        yaxis="y2", name="Volume",
    ))

    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0a0f1a",
        font=dict(family="Space Mono", color="#6b7a99", size=11),
        title=dict(text=f"{ticker} — Price & Volume", font=dict(family="Syne", size=16, color="#e8eaf0")),
        xaxis=dict(gridcolor="#1e2d45", showgrid=True, rangeslider_visible=False),
        yaxis=dict(gridcolor="#1e2d45", showgrid=True, side="right"),
        yaxis2=dict(overlaying="y", side="left", showgrid=False, showticklabels=False),
        legend=dict(bgcolor="#0f1623", bordercolor="#1e2d45"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
    )
    return fig


def make_candlestick_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """Build a dark-themed candlestick chart."""
    
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],

        increasing_line_color="#00f5a0",
        decreasing_line_color="#ff4c6a",

        # FIXED HERE
        increasing_fillcolor="rgba(0,245,160,0.18)",
        decreasing_fillcolor="rgba(255,76,106,0.18)",
    )])

    # Volume bars
    colors = [
        "#00f5a0" if c >= o else "#ff4c6a"
        for c, o in zip(hist["Close"], hist["Open"])
    ]

    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist["Volume"],
        marker_color=colors,
        opacity=0.3,
        yaxis="y2",
        name="Volume",
    ))

    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0a0f1a",
        font=dict(
            family="Space Mono",
            color="#6b7a99",
            size=11
        ),
        title=dict(
            text=f"{ticker} — Price & Volume",
            font=dict(
                family="Syne",
                size=16,
                color="#e8eaf0"
            )
        ),
        xaxis=dict(
            gridcolor="#1e2d45",
            showgrid=True,
            rangeslider_visible=False
        ),
        yaxis=dict(
            gridcolor="#1e2d45",
            showgrid=True,
            side="right"
        ),
        yaxis2=dict(
            overlaying="y",
            side="left",
            showgrid=False,
            showticklabels=False
        ),
        legend=dict(
            bgcolor="#0f1623",
            bordercolor="#1e2d45"
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
    )

    return fig

def make_sentiment_gauge(score: float) -> go.Figure:
    """Render a gauge chart for sentiment score."""

    color = "#00f5a0" if score > 0.1 else (
        "#ff4c6a" if score < -0.1 else "#f5c400"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,

        number={
            "font": {
                "family": "Syne",
                "size": 28,
                "color": color
            }
        },

        gauge={
            "axis": {
                "range": [-1, 1],
                "tickcolor": "#6b7a99",
                "tickfont": {
                    "family": "Space Mono",
                    "size": 9
                }
            },

            "bar": {
                "color": color,
                "thickness": 0.25
            },

            "bgcolor": "#0f1623",
            "borderwidth": 0,

            "steps": [
                {"range": [-1, -0.1], "color": "rgba(255,76,106,0.15)"},
                {"range": [-0.1, 0.1], "color": "rgba(245,196,0,0.15)"},
                {"range": [0.1, 1], "color": "rgba(0,245,160,0.15)"}
            ],

            "threshold": {
                "line": {
                    "color": color,
                    "width": 3
                },
                "value": score
            }
        },

        title={
            "text": "Sentiment Score",
            "font": {
                "family": "Space Mono",
                "size": 11,
                "color": "#6b7a99"
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0f1623",
        height=220,
        margin=dict(l=20, r=20, t=40, b=10)
    )

    return fig

def sentiment_color(s: str) -> str:
    s = s.upper()
    if s == "POSITIVE": return "positive"
    if s == "NEGATIVE": return "negative"
    return "neutral"


def movement_emoji(m: str) -> str:
    m = m.upper()
    if m == "BULLISH":  return "🚀"
    if m == "BEARISH":  return "🔻"
    return "↔️"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    ticker_input = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g. TSLA, NVDA, MSFT").upper().strip()
    company_input = st.text_input("Company Name (for news search)", value="Apple", placeholder="e.g. Tesla")
    num_articles  = st.slider("Number of Articles", min_value=5, max_value=20, value=10, step=1)
    period_map    = {"1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo"}
    period_label  = st.selectbox("Chart Period", list(period_map.keys()), index=1)
    chart_period  = period_map[period_label]

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Sentiment")

    st.markdown("---")
    st.markdown("""
<div style='font-family: Space Mono; font-size: 0.6rem; color: #3a4a66; line-height: 1.8;'>
DATA SOURCES<br>
• NewsAPI — live headlines<br>
• yFinance — price & fundamentals<br>
• Groq / LLaMA-3.3-70B — NLP<br><br>
⚠️ Not financial advice.
</div>
""", unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Stock Sentiment Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Financial NLP · Real-time News · AI-Powered Analysis</div>', unsafe_allow_html=True)

if not analyze_btn:
    st.markdown("""
<div class="metric-card" style="text-align:center; padding: 3rem;">
<div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
<div style="font-family: Syne; font-size: 1.1rem; font-weight: 600; color: #c8cdd8; margin-bottom: 0.5rem;">
  Enter a ticker and click Analyze Sentiment
</div>
<div style="font-family: Space Mono; font-size: 0.7rem; color: #6b7a99;">
  The app will fetch live news, run LLaMA-3.3-70B sentiment analysis,<br>
  and predict bullish / bearish / sideways movement.
</div>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── Run analysis ───────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker_input}**…"):
    hist, stock_info = get_stock_data(ticker_input, chart_period)
    articles         = fetch_news(ticker_input, company_input, num_articles)

if hist.empty:
    st.error("Could not fetch stock data. Check the ticker symbol.")
    st.stop()

if not articles:
    st.warning("No news articles found. Try a different company name or check your API key.")

with st.spinner("Running Groq NLP sentiment analysis…"):
    result = analyze_sentiment_with_groq(articles, ticker_input, stock_info)

if not result:
    st.error("Sentiment analysis failed. Check your Groq API key.")
    st.stop()


# ── Top KPI row ────────────────────────────────────────────────────────────────
current_price   = stock_info.get("currentPrice") or stock_info.get("regularMarketPrice") or (hist["Close"].iloc[-1] if not hist.empty else "N/A")
prev_close      = stock_info.get("previousClose") or (hist["Close"].iloc[-2] if len(hist) > 1 else None)
price_change    = ((current_price - prev_close) / prev_close * 100) if prev_close and isinstance(current_price, float) else None
overall_sent    = result.get("overall_sentiment", "NEUTRAL")
movement        = result.get("predicted_movement", "SIDEWAYS")
confidence      = result.get("confidence", 0)
predicted_chg   = result.get("predicted_change_pct", 0)
sent_score      = result.get("sentiment_score", 0)

col1, col2, col3, col4 = st.columns(4)

with col1:
    chg_color = "positive" if (price_change or 0) >= 0 else "negative"
    chg_str   = f"{price_change:+.2f}%" if price_change is not None else "N/A"
    st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Current Price</div>
  <div class="metric-value">${current_price:.2f}</div>
  <div class="{chg_color}" style="font-family:Space Mono;font-size:0.75rem;">{chg_str} today</div>
</div>""", unsafe_allow_html=True)

with col2:
    sc = sentiment_color(overall_sent)
    st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Overall Sentiment</div>
  <div class="metric-value {sc}">{overall_sent}</div>
  <div style="font-family:Space Mono;font-size:0.75rem;color:#6b7a99;">score: {sent_score:+.2f}</div>
</div>""", unsafe_allow_html=True)

with col3:
    mc = sentiment_color(overall_sent) if movement != "SIDEWAYS" else "neutral"
    st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Predicted Movement</div>
  <div class="metric-value {mc}">{movement_emoji(movement)} {movement}</div>
  <div style="font-family:Space Mono;font-size:0.75rem;color:#6b7a99;">est. {predicted_chg:+.2f}% (1-3d)</div>
</div>""", unsafe_allow_html=True)

with col4:
    conf_pct = int(confidence * 100)
    st.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Confidence</div>
  <div class="metric-value">{conf_pct}%</div>
  <div class="score-bar-container">
    <div class="score-bar-bg">
      <div class="score-bar-fill" style="width:{conf_pct}%; background: linear-gradient(90deg,#00f5a0,#00d9f5);"></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ── Prediction summary ─────────────────────────────────────────────────────────
summary = result.get("summary", "")
if summary:
    st.markdown(f"""
<div class="prediction-box">
  <div class="prediction-label">🤖 AI Prediction Summary</div>
  <div class="prediction-text">{summary}</div>
</div>""", unsafe_allow_html=True)


# ── Chart + Gauge ──────────────────────────────────────────────────────────────
chart_col, gauge_col = st.columns([2, 1])

with chart_col:
    st.plotly_chart(make_candlestick_chart(hist, ticker_input), use_container_width=True)

with gauge_col:
    st.plotly_chart(make_sentiment_gauge(sent_score), use_container_width=True)

    # Key themes
    themes = result.get("key_themes", [])
    if themes:
        st.markdown('<div class="metric-label" style="margin-top:0.5rem;">Key Themes</div>', unsafe_allow_html=True)
        for t in themes:
            st.markdown(f'<span class="badge badge-neutral" style="margin:2px 3px;display:inline-block;">{t}</span>', unsafe_allow_html=True)

    # Risk factors
    risks = result.get("risk_factors", [])
    if risks:
        st.markdown('<div class="metric-label" style="margin-top:0.8rem;">Risk Factors</div>', unsafe_allow_html=True)
        for r in risks:
            st.markdown(f'<span class="badge badge-negative" style="margin:2px 3px;display:inline-block;">{r}</span>', unsafe_allow_html=True)


# ── Article sentiment breakdown ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="font-family:Syne;font-weight:800;font-size:1.2rem;margin-bottom:1rem;">📰 News Article Sentiment Breakdown</div>', unsafe_allow_html=True)

analyzed_articles = result.get("articles", [])

# Merge Groq results back with original NewsAPI articles
news_col, bar_col = st.columns([3, 2])

with news_col:
    for item in analyzed_articles:
        idx  = item.get("index", 1) - 1
        orig = articles[idx] if idx < len(articles) else {}
        sent = item.get("sentiment", "NEUTRAL")
        sc   = sentiment_color(sent)
        impact = item.get("impact_score", 0)
        title  = item.get("title") or orig.get("title", "No title")
        source = orig.get("source", {}).get("name", "Unknown")
        pub_at = orig.get("publishedAt", "")[:10]
        reason = item.get("reason", "")
        url    = orig.get("url", "#")

        st.markdown(f"""
<div class="news-card {sc}">
  <div class="news-title">
    {title}
    <span class="badge badge-{sc}">{sent}</span>
  </div>
  <div class="news-meta">{source} · {pub_at} · impact: {impact:+.2f}</div>
  <div style="font-size:0.8rem;color:#8892a4;margin-top:0.35rem;">{reason}</div>
</div>""", unsafe_allow_html=True)

with bar_col:
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for item in analyzed_articles:
        s = item.get("sentiment", "NEUTRAL").upper()
        if s in counts:
            counts[s] += 1

    fig_pie = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.55,
        marker=dict(colors=["#00f5a0", "#ff4c6a", "#f5c400"],
                    line=dict(color="#080c14", width=2)),
        textfont=dict(family="Space Mono", size=10, color="#e8eaf0"),
    ))
    fig_pie.update_layout(
        paper_bgcolor="#0f1623",
        plot_bgcolor="#0f1623",
        font=dict(family="Space Mono", color="#6b7a99"),
        legend=dict(bgcolor="#0f1623", font=dict(color="#c8cdd8", size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
        title=dict(text="Sentiment Distribution", font=dict(family="Syne", size=13, color="#e8eaf0")),
        annotations=[dict(text=f"{len(analyzed_articles)}<br>articles",
                          x=0.5, y=0.5, font_size=12, showarrow=False,
                          font=dict(family="Syne", color="#c8cdd8"))],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Impact bar chart
    impact_df = pd.DataFrame([
        {"Article": f"#{a['index']}", "Impact": a.get("impact_score", 0),
         "Color": "#00f5a0" if a.get("impact_score", 0) > 0 else "#ff4c6a"}
        for a in analyzed_articles
    ])
    if not impact_df.empty:
        fig_bar = go.Figure(go.Bar(
            x=impact_df["Article"],
            y=impact_df["Impact"],
            marker_color=impact_df["Color"],
            marker_line_width=0,
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0f1623",
            plot_bgcolor="#0a0f1a",
            font=dict(family="Space Mono", color="#6b7a99", size=10),
            title=dict(text="Article Impact Scores", font=dict(family="Syne", size=13, color="#e8eaf0")),
            xaxis=dict(gridcolor="#1e2d45"),
            yaxis=dict(gridcolor="#1e2d45", range=[-1, 1]),
            margin=dict(l=10, r=10, t=40, b=10),
            height=220,
        )
        fig_bar.add_hline(y=0, line_color="#2a3a55", line_width=1)
        st.plotly_chart(fig_bar, use_container_width=True)


# ── Stock fundamentals ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="font-family:Syne;font-weight:800;font-size:1.2rem;margin-bottom:1rem;">📊 Stock Fundamentals</div>', unsafe_allow_html=True)

keys = {
    "Market Cap":       stock_info.get("marketCap"),
    "P/E Ratio":        stock_info.get("trailingPE"),
    "52W High":         stock_info.get("fiftyTwoWeekHigh"),
    "52W Low":          stock_info.get("fiftyTwoWeekLow"),
    "Avg Volume":       stock_info.get("averageVolume"),
    "Beta":             stock_info.get("beta"),
    "Revenue":          stock_info.get("totalRevenue"),
    "Profit Margin":    stock_info.get("profitMargins"),
}

cols = st.columns(4)
for i, (label, val) in enumerate(keys.items()):
    with cols[i % 4]:
        if val is None:
            display = "N/A"
        elif isinstance(val, float) and val < 1:
            display = f"{val:.1%}"
        elif isinstance(val, (int, float)) and val > 1e9:
            display = f"${val/1e9:.1f}B"
        elif isinstance(val, (int, float)) and val > 1e6:
            display = f"${val/1e6:.1f}M"
        else:
            display = f"{val:.2f}" if isinstance(val, float) else str(val)

        st.markdown(f"""
<div class="metric-card" style="padding:0.9rem 1rem;">
  <div class="metric-label">{label}</div>
  <div style="font-family:Syne;font-weight:700;font-size:1.1rem;color:#c8cdd8;">{display}</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;font-family:Space Mono;font-size:0.58rem;color:#3a4a66;margin-top:2rem;">
  ⚠️ This tool is for educational and NLP research purposes only. Not financial advice.
  Data: NewsAPI · yFinance · Groq/LLaMA-3.3-70B
</div>
""", unsafe_allow_html=True)
