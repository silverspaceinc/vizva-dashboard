import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import requests
import io
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from collections import Counter
import numpy as np

# ── NLTK bootstrap (download once, cached) ──────────────────────
@st.cache_resource
def _download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("vader_lexicon", quiet=True)

_download_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Vizva Interview Dashboard", page_icon="chart_with_upwards_trend",
                   layout="wide", initial_sidebar_state="expanded")

API_KEY = st.secrets["API_KEY"]
BASE_URL = st.secrets["BASE_URL"]

TASK_ORDER = ["completed", "rescheduled", "cancelled", "pending"]
TASK_LABEL = {"completed": "Completed", "rescheduled": "Rescheduled",
              "cancelled": "Cancelled", "pending": "Pending"}
CLR = {"completed": "#2ecc71", "rescheduled": "#f39c12",
       "cancelled": "#e74c3c", "pending": "#3498db"}

SUPPORT_TYPES = ["Interview Support", "Assessment Support", "Mock Interview", "Resume Understanding"]

ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

HIST = {
    "Interview Support": {
        "2025-07": {"completed": 54, "rescheduled": 14, "cancelled": 11, "candidates": 17},
        "2025-08": {"completed": 56, "rescheduled": 5,  "cancelled": 17, "candidates": 26},
        "2025-09": {"completed": 50, "rescheduled": 2,  "cancelled": 24, "candidates": 35},
        "2025-10": {"completed": 37, "rescheduled": 10, "cancelled": 19, "candidates": 24},
        "2025-11": {"completed": 32, "rescheduled": 2,  "cancelled": 9,  "candidates": 19},
        "2025-12": {"completed": 37, "rescheduled": 1,  "cancelled": 19, "candidates": 29},
        "2026-01": {"completed": 49, "rescheduled": 3,  "cancelled": 17, "candidates": 29},
        "2026-02": {"completed": 66, "rescheduled": 6,  "cancelled": 11, "candidates": 28},
        "2026-03": {"completed": 81, "rescheduled": 6,  "cancelled": 21, "candidates": 37},
        "2026-04": {"completed": 71, "rescheduled": 9,  "cancelled": 24, "candidates": 52},
    },
}

# ── Domain-specific stopwords to exclude from analysis ───────────
DOMAIN_STOPWORDS = {
    "interview", "assessment", "mock", "resume", "understanding",
    "support", "interviewer", "interviews", "assessments",
    "candidate", "candidates", "expert", "experts",
    "also", "would", "could", "get", "got", "go", "went",
    "one", "two", "good", "well", "done", "like", "need",
    "much", "many", "even", "still", "really", "thing", "things",
    "make", "made", "take", "took", "give", "gave", "come", "came",
    "know", "said", "say", "asked", "question", "questions",
    "answer", "answered", "round", "rounds", "company", "name",
    "time", "day", "date", "month", "yes", "no",
}

# ── Sentiment colour helpers ─────────────────────────────────────
SENT_CLR = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}


def sentiment_color(score):
    if score >= 20:
        return "#2ecc71"
    elif score <= -20:
        return "#e74c3c"
    return "#f39c12"


def sentiment_label(score):
    if score >= 20:
        return "Positive"
    elif score <= -20:
        return "Negative"
    return "Neutral"


# ═══════════════════════════════════════════════════════════════════
#  VADER SENTIMENT UTILITIES (used by Dashboard views)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_vader():
    return SentimentIntensityAnalyzer()


def compute_sentiment_score(text):
    if not isinstance(text, str) or not text.strip():
        return None
    sia = _get_vader()
    compound = sia.polarity_scores(text)["compound"]
    return round(compound * 100, 1)


def add_sentiment_column(df, feedback_col="feedback"):
    candidates = [feedback_col, "feedback", "Feedback", "feedback_text",
                  "case_feedback", "expert_feedback", "comments", "remark", "remarks"]
    found_col = None
    for c in candidates:
        if c in df.columns:
            found_col = c
            break
    if found_col is None:
        df["sentiment_score"] = None
        df["sentiment_label"] = None
        return df, None

    df["sentiment_score"] = df[found_col].apply(compute_sentiment_score)
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda x: sentiment_label(x) if pd.notna(x) else None
    )
    return df, found_col


def get_sentiment_stats(df):
    valid = df["sentiment_score"].dropna()
    if valid.empty:
        return None
    stats = {
        "avg": round(valid.mean(), 1),
        "min": round(valid.min(), 1),
        "max": round(valid.max(), 1),
        "count": len(valid),
        "positive": int((valid >= 20).sum()),
        "neutral": int(((valid > -20) & (valid < 20)).sum()),
        "negative": int((valid <= -20).sum()),
    }
    return stats


def render_sentiment_kpi(stats, title="Feedback Sentiment"):
    if stats is None:
        st.info("No feedback available for sentiment analysis.")
        return

    st.subheader(title)

    avg = stats["avg"]
    clr = sentiment_color(avg)
    lbl = sentiment_label(avg)

    c = st.columns(6)
    c[0].metric("Avg Sentiment", f"{avg:+.1f}%",
                delta=lbl, delta_color="normal" if avg >= 0 else "inverse")
    c[1].metric("Feedbacks Analyzed", stats["count"])
    c[2].metric("Positive (≥20%)", stats["positive"])
    c[3].metric("Neutral (-20% to 20%)", stats["neutral"])
    c[4].metric("Negative (≤-20%)", stats["negative"])
    c[5].metric("Range", f"{stats['min']:+.0f}% to {stats['max']:+.0f}%")


def render_sentiment_donut(stats, title="Sentiment Split"):
    if stats is None:
        return None
    labels = ["Positive", "Neutral", "Negative"]
    values = [stats["positive"], stats["neutral"], stats["negative"]]
    colors = [SENT_CLR[l] for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=colors),
        textinfo="label+value+percent",
    ))
    fig.update_layout(title=title, height=380, showlegend=False)
    return fig


def render_sentiment_histogram(df, title="Sentiment Score Distribution"):
    valid = df["sentiment_score"].dropna()
    if valid.empty:
        return None
    fig = go.Figure(go.Histogram(
        x=valid, nbinsx=20,
        marker_color="#3498db",
        marker_line=dict(color="white", width=1),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    fig.add_vline(x=valid.mean(), line_dash="dot", line_color="#e74c3c",
                  annotation_text=f"Avg: {valid.mean():+.1f}%")
    fig.update_layout(
        title=title, height=380,
        xaxis_title="Sentiment Score (%)",
        yaxis_title="Count",
        xaxis=dict(range=[-105, 105]),
    )
    return fig


def render_sentiment_section(df, section_title="Feedback Sentiment Analysis"):
    stats = get_sentiment_stats(df)
    if stats is None:
        st.info("No feedback available for sentiment analysis in this selection.")
        return

    render_sentiment_kpi(stats, title=section_title)

    col1, col2 = st.columns(2)
    with col1:
        fig = render_sentiment_donut(stats, "Sentiment Split")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = render_sentiment_histogram(df, "Score Distribution")
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def monthly_sentiment_trend(df):
    if df.empty or "date" not in df.columns or "sentiment_score" not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=["sentiment_score"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    agg = d.groupby("month")["sentiment_score"].agg(
        avg_sentiment="mean",
        feedbacks="count",
        min_sentiment="min",
        max_sentiment="max",
    ).reset_index()
    agg["avg_sentiment"] = agg["avg_sentiment"].round(1)
    agg["min_sentiment"] = agg["min_sentiment"].round(1)
    agg["max_sentiment"] = agg["max_sentiment"].round(1)

    pos = d[d["sentiment_score"] >= 20].groupby(d["date"].dt.to_period("M").astype(str)).size().rename("positive")
    neu = d[(d["sentiment_score"] > -20) & (d["sentiment_score"] < 20)].groupby(d["date"].dt.to_period("M").astype(str)).size().rename("neutral")
    neg = d[d["sentiment_score"] <= -20].groupby(d["date"].dt.to_period("M").astype(str)).size().rename("negative")

    agg = agg.merge(pos, left_on="month", right_index=True, how="left")
    agg = agg.merge(neu, left_on="month", right_index=True, how="left")
    agg = agg.merge(neg, left_on="month", right_index=True, how="left")
    agg = agg.fillna(0)
    for c in ["positive", "neutral", "negative"]:
        agg[c] = agg[c].astype(int)

    return agg


def render_sentiment_trend_chart(sent_monthly, title="Monthly Avg Sentiment Trend"):
    if sent_monthly.empty:
        return None
    colors = [sentiment_color(v) for v in sent_monthly["avg_sentiment"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sent_monthly["month"],
        y=sent_monthly["avg_sentiment"],
        mode="lines+markers+text",
        text=sent_monthly["avg_sentiment"].apply(lambda v: f"{v:+.1f}%"),
        textposition="top center",
        line=dict(color="#8e44ad", width=3),
        marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=20, line_dash="dot", line_color="#2ecc71", opacity=0.3,
                  annotation_text="Positive threshold")
    fig.add_hline(y=-20, line_dash="dot", line_color="#e74c3c", opacity=0.3,
                  annotation_text="Negative threshold")
    fig.update_layout(
        title=title, height=420,
        yaxis_title="Avg Sentiment (%)",
        yaxis=dict(range=[
            min(-50, sent_monthly["avg_sentiment"].min() - 15),
            max(50, sent_monthly["avg_sentiment"].max() + 15)
        ]),
    )
    return fig


def render_sentiment_stacked_bar(sent_monthly, title="Monthly Sentiment Breakdown"):
    if sent_monthly.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sent_monthly["month"], y=sent_monthly["positive"],
                         name="Positive", marker_color="#2ecc71",
                         text=sent_monthly["positive"], textposition="inside"))
    fig.add_trace(go.Bar(x=sent_monthly["month"], y=sent_monthly["neutral"],
                         name="Neutral", marker_color="#f39c12",
                         text=sent_monthly["neutral"], textposition="inside"))
    fig.add_trace(go.Bar(x=sent_monthly["month"], y=sent_monthly["negative"],
                         name="Negative", marker_color="#e74c3c",
                         text=sent_monthly["negative"], textposition="inside"))
    fig.update_layout(barmode="stack", title=title, height=420,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


def daily_sentiment_agg(df):
    if df.empty or "date" not in df.columns or "sentiment_score" not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=["sentiment_score"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["day"] = d["date"].dt.date
    agg = d.groupby("day")["sentiment_score"].agg(
        avg_sentiment="mean",
        feedbacks="count",
    ).reset_index()
    agg["avg_sentiment"] = agg["avg_sentiment"].round(1)
    return agg


# ═══════════════════════════════════════════════════════════════════
#  START TIME UTILITIES
#  Treats "noon" (case-insensitive) as 12:00 PM
#  Used ONLY when Interview Support is selected
# ═══════════════════════════════════════════════════════════════════

def add_start_time_columns(df):
    """Add start_hour and start_hour_label columns.
    Modifies df in-place and returns it."""
    if "start_time" not in df.columns:
        return df

    df = df.copy()

    # ── Parse start_time ─────────────────────────────────────────
    st_raw = df["start_time"].astype(str).str.strip()
    noon_s = st_raw.str.lower() == "noon"
    st_raw = st_raw.where(~noon_s, "12:00 PM")
    # Try 12-hour format first
    start_parsed = pd.to_datetime(st_raw, errors="coerce", format="%I:%M %p")
    # Fallback: 24-hour format
    mask_nat = start_parsed.isna() & st_raw.notna() & (st_raw != "") & (st_raw.str.lower() != "nan")
    if mask_nat.any():
        fallback1 = pd.to_datetime(st_raw[mask_nat], errors="coerce", format="%H:%M")
        start_parsed.loc[mask_nat] = fallback1
    # Fallback: mixed format
    mask_nat2 = start_parsed.isna() & st_raw.notna() & (st_raw != "") & (st_raw.str.lower() != "nan")
    if mask_nat2.any():
        fallback2 = pd.to_datetime(st_raw[mask_nat2], errors="coerce", format="mixed")
        start_parsed.loc[mask_nat2] = fallback2

    df["start_hour"] = start_parsed.dt.hour
    df["start_hour_label"] = start_parsed.dt.strftime("%I %p")

    return df


def render_start_time_insights(df, title_suffix=""):
    """Render the Start Time Insights section: hourly distribution,
    peak-hour KPIs, and mean interview start time."""
    if "start_hour" not in df.columns:
        st.info("No start_time data available for time-of-day analysis.")
        return

    valid = df.dropna(subset=["start_hour"]).copy()
    if valid.empty:
        st.info("No valid start_time entries found" + title_suffix + ".")
        return

    st.subheader("Start Time Insights" + title_suffix)

    # ── KPI row ──────────────────────────────────────────────────
    total_with_time = len(valid)
    hour_counts = valid["start_hour"].value_counts().sort_index()
    peak_hour = int(hour_counts.idxmax())
    peak_count = int(hour_counts.max())
    peak_label = datetime(2000, 1, 1, peak_hour).strftime("%I:%M %p")
    mean_hour = valid["start_hour"].mean()
    mean_label = datetime(2000, 1, 1, int(mean_hour), int((mean_hour % 1) * 60)).strftime("%I:%M %p")

    k_cols = st.columns(4)
    k_cols[0].metric("Interviews with Time", total_with_time)
    k_cols[1].metric("Peak Hour", peak_label)
    k_cols[2].metric("Interviews at Peak", peak_count)
    k_cols[3].metric("Mean Start Time", mean_label)

    # ── Hourly distribution bar chart ────────────────────────────
    all_hours = list(range(0, 24))
    hour_labels = [datetime(2000, 1, 1, h).strftime("%I %p").lstrip("0") for h in all_hours]
    counts = [int(hour_counts.get(h, 0)) for h in all_hours]

    bar_colors = ["#e74c3c" if h == peak_hour else "#3498db" for h in all_hours]

    fig_hourly = go.Figure(go.Bar(
        x=hour_labels, y=counts,
        marker_color=bar_colors,
        text=counts, textposition="outside",
    ))
    fig_hourly.update_layout(
        title="Interview Count by Hour of Day" + title_suffix,
        height=420,
        xaxis_title="Hour of Day",
        yaxis_title="Number of Interviews",
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

    # ── Time-slot breakdown table ────────────────────────────────
    slots = {
        "Early Morning (6-9 AM)": (6, 9),
        "Morning (9 AM-12 PM)": (9, 12),
        "Afternoon (12-3 PM)": (12, 15),
        "Late Afternoon (3-6 PM)": (15, 18),
        "Evening (6-9 PM)": (18, 21),
        "Night (9 PM-6 AM)": None,
    }
    slot_rows = []
    for slot_name, rng in slots.items():
        if rng:
            cnt = int(((valid["start_hour"] >= rng[0]) & (valid["start_hour"] < rng[1])).sum())
        else:
            cnt = int(((valid["start_hour"] >= 21) | (valid["start_hour"] < 6)).sum())
        pct = round(cnt / total_with_time * 100, 1) if total_with_time > 0 else 0
        slot_rows.append({"Time Slot": slot_name, "Count": cnt, "% of Total": pct})
    slot_df = pd.DataFrame(slot_rows)

    with st.expander("Time Slot Breakdown" + title_suffix):
        st.dataframe(slot_df, use_container_width=True, hide_index=True)


def render_monthly_start_time_trend(df, title_suffix=""):
    """Render a monthly x hour-of-day heatmap and monthly mean start time line."""
    if "start_hour" not in df.columns or "date" not in df.columns:
        return

    valid = df.dropna(subset=["start_hour"]).copy()
    if valid.empty:
        return

    valid["month"] = valid["date"].dt.to_period("M").astype(str)

    st.subheader("Monthly Start Time Trends" + title_suffix)

    # ── Heatmap: month x hour ────────────────────────────────────
    pivot = valid.groupby(["month", "start_hour"]).size().reset_index(name="count")
    pivot_wide = pivot.pivot(index="start_hour", columns="month", values="count").fillna(0).astype(int)
    pivot_wide = pivot_wide.reindex(range(24), fill_value=0)
    pivot_wide.index = [datetime(2000, 1, 1, h).strftime("%I %p").lstrip("0") for h in range(24)]

    fig_heat = px.imshow(
        pivot_wide, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        title="Interviews by Hour & Month" + title_suffix,
        labels=dict(x="Month", y="Hour of Day", color="Count"),
    )
    fig_heat.update_layout(height=550)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Monthly mean start time line chart ───────────────────────
    monthly_agg = valid.groupby("month").agg(
        interviews=("start_hour", "size"),
        mean_start_hour=("start_hour", "mean"),
    ).reset_index()
    monthly_agg["mean_start_hour"] = monthly_agg["mean_start_hour"].round(2)
    monthly_agg["mean_start_label"] = monthly_agg["mean_start_hour"].apply(
        lambda h: datetime(2000, 1, 1, int(h), int((h % 1) * 60)).strftime("%I:%M %p")
    )

    fig_mean = go.Figure()
    fig_mean.add_trace(go.Scatter(
        x=monthly_agg["month"],
        y=monthly_agg["mean_start_hour"],
        mode="lines+markers+text",
        text=monthly_agg["mean_start_label"],
        textposition="top center",
        line=dict(color="#e67e22", width=3),
        marker=dict(size=10),
    ))
    fig_mean.update_layout(
        title="Mean Interview Start Time by Month" + title_suffix,
        height=400, xaxis_title="Month",
        yaxis_title="Hour of Day (24h)",
        yaxis=dict(range=[
            max(0, monthly_agg["mean_start_hour"].min() - 2),
            min(24, monthly_agg["mean_start_hour"].max() + 2),
        ]),
    )
    st.plotly_chart(fig_mean, use_container_width=True)

    with st.expander("Monthly Start Time Data" + title_suffix):
        st.dataframe(monthly_agg[["month", "interviews", "mean_start_label"]],
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  RoBERTa ONNX SENTIMENT (used ONLY by Transcript Analyzer view)
#  Model: cardiffnlp/twitter-roberta-base-sentiment-latest
#  ONNX weights: Xenova/twitter-roberta-base-sentiment-latest
#  Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
#  NO PyTorch — uses onnxruntime + quantized ONNX (~126MB)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def _load_roberta_onnx():
    """Load tokenizer + ONNX session once, cache across reruns."""
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    model_id = "Xenova/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_path = hf_hub_download(repo_id=model_id, filename="onnx/model_quantized.onnx")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 2
    session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
    return tokenizer, session


def _roberta_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def _roberta_preprocess(text):
    tokens = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        tokens.append(t)
    return " ".join(tokens)


def roberta_score_text(text):
    """Score a single text with RoBERTa ONNX. Returns dict."""
    tokenizer, session = _load_roberta_onnx()
    cleaned = _roberta_preprocess(text)
    encoded = tokenizer(cleaned, truncation=True, max_length=512, return_tensors="np")
    feeds = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    logits = session.run(None, feeds)[0]
    probs = _roberta_softmax(logits)[0]
    score = round((float(probs[2]) - float(probs[0])) * 100, 1)
    return {
        "score": score,
        "p_negative": round(float(probs[0]) * 100, 1),
        "p_neutral": round(float(probs[1]) * 100, 1),
        "p_positive": round(float(probs[2]) * 100, 1),
    }


def roberta_score_chunks(text, chunk_mode="paragraph"):
    """Split text into chunks and score each with RoBERTa ONNX."""
    if chunk_mode == "paragraph":
        chunks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    elif chunk_mode == "sentence":
        chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    else:
        chunks = [text.strip()]
    if not chunks:
        return []

    tokenizer, session = _load_roberta_onnx()
    results = []
    for chunk in chunks:
        cleaned = _roberta_preprocess(chunk)
        try:
            encoded = tokenizer(cleaned, truncation=True, max_length=512, return_tensors="np")
            feeds = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            logits = session.run(None, feeds)[0]
            probs = _roberta_softmax(logits)[0]
            score = round((float(probs[2]) - float(probs[0])) * 100, 1)
            results.append({
                "text": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                "full_text": chunk,
                "score": score,
                "p_negative": round(float(probs[0]) * 100, 1),
                "p_neutral": round(float(probs[1]) * 100, 1),
                "p_positive": round(float(probs[2]) * 100, 1),
                "label": "Positive" if score >= 20 else ("Negative" if score <= -20 else "Neutral"),
            })
        except Exception:
            results.append({
                "text": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                "full_text": chunk,
                "score": None, "p_negative": None, "p_neutral": None,
                "p_positive": None, "label": "Error",
            })
    return results


# ═══════════════════════════════════════════════════════════════════
#  TEXT PREPROCESSING + WORDCLOUD UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _nltk_pos_to_wordnet(tag):
    from nltk.corpus import wordnet
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_feedback_texts(texts):
    if not texts:
        return []
    english_stops = set(stopwords.words("english"))
    all_stops = english_stops | DOMAIN_STOPWORDS
    lemmatizer = WordNetLemmatizer()
    all_tokens = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        for word, tag in tagged:
            lemma = lemmatizer.lemmatize(word, _nltk_pos_to_wordnet(tag))
            if lemma not in all_stops and len(lemma) > 1:
                all_tokens.append(lemma)
    return all_tokens


def get_top_words(tokens, n=10):
    if not tokens:
        return []
    return Counter(tokens).most_common(n)


def render_wordcloud_section(feedback_texts, section_title="Feedback Word Cloud"):
    if not feedback_texts:
        st.info("No feedback text available for word cloud analysis.")
        return
    tokens = preprocess_feedback_texts(feedback_texts)
    if not tokens:
        st.info("No meaningful words found after preprocessing the feedback.")
        return
    top10 = get_top_words(tokens, 10)
    freq = Counter(tokens)

    st.subheader(section_title)
    st.caption(
        f"Preprocessing: lowercased → punctuation/numbers removed → English stopwords removed → "
        f"domain words removed (interview, assessment, mock …) → lemmatized. "
        f"**{len(feedback_texts)}** feedback entries → **{len(tokens)}** tokens → "
        f"**{len(freq)}** unique words."
    )

    top_df = pd.DataFrame(top10, columns=["Word", "Count"])
    fig_top = go.Figure(go.Bar(
        x=top_df["Count"], y=top_df["Word"], orientation="h",
        marker_color="#2980b9", text=top_df["Count"], textposition="outside",
    ))
    fig_top.update_layout(
        title="Top 10 Words in Feedback", height=400,
        yaxis=dict(autorange="reversed"), xaxis_title="Count", yaxis_title="",
    )

    wc = WordCloud(
        width=1000, height=500, background_color="white",
        colormap="viridis", max_words=120, min_font_size=10, prefer_horizontal=0.7,
    ).generate_from_frequencies(freq)

    fig_wc, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)

    col_bar, col_wc = st.columns([1, 2])
    with col_bar:
        st.plotly_chart(fig_top, use_container_width=True)
    with col_wc:
        st.pyplot(fig_wc)
    plt.close(fig_wc)


def extract_feedback_texts(df, col="feedback"):
    candidates = [col, "feedback", "Feedback", "feedback_text",
                  "case_feedback", "expert_feedback", "comments", "remark", "remarks"]
    for c in candidates:
        if c in df.columns:
            series = df[c].dropna().astype(str).str.strip()
            series = series[series != ""]
            series = series[series.str.lower() != "nan"]
            series = series[series.str.lower() != "none"]
            return series.tolist()
    return []


# ═══════════════════════════════════════════════════════════════════
#  ORIGINAL HELPERS (unchanged)
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def fetch_all_data():
    headers = {"x-api-key": API_KEY}
    all_records = []
    offset = 0
    limit = 500
    while True:
        response = requests.get(
            BASE_URL + "/api/app-case",
            headers=headers,
            params={"limit": limit, "offset": offset}
        )
        response.raise_for_status()
        batch = response.json()["data"]
        if not batch:
            break
        all_records.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    df = pd.DataFrame(all_records)
    return df


def normalize(df):
    cols_to_drop = ["case_candidate_phone", "status", "filled_by_username", "candidate_resume",
                    "case_candidate_email", "candidate_phone", "candidate_email", "candidate_status_flag",
                    "expert_is_team_lead", "expert_date_of_joining", "filled_by_first_name",
                    "filled_by_last_name", "filled_by_email"]
    id_cols = [c for c in df.columns if c.endswith("_id") or c == "id"]
    cols_to_drop = cols_to_drop + id_cols
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "task_status" in df.columns:
        df["task_status"] = df["task_status"].fillna("pending")
        df["task_status"] = df["task_status"].astype(str).str.strip().str.lower()
        df["task_status"] = df["task_status"].replace("not done", "pending")
        df.loc[~df["task_status"].isin(["completed", "rescheduled", "cancelled", "pending"]), "task_status"] = "pending"
    if "support_name" in df.columns:
        df["support_name"] = df["support_name"].astype(str).str.strip()
    return df


def filter_current_year(df):
    if "date" not in df.columns:
        return df
    current_year = datetime.now().year
    return df[df["date"].dt.year == current_year].copy()


def filter_active_experts(df):
    if "expert_status_flag" not in df.columns:
        return df
    filtered = df[df["expert_status_flag"] == True].copy()
    filtered = filtered.drop(columns=["expert_status_flag"], errors="ignore")
    return filtered


def get_by_support(df, support_type):
    if df.empty or "support_name" not in df.columns:
        return df
    return df[df["support_name"].str.lower() == support_type.lower()].copy()


def hist_monthly_df(support_type):
    if support_type not in HIST:
        return pd.DataFrame()
    rows = []
    for m, d in HIST[support_type].items():
        total = d["completed"] + d["rescheduled"] + d["cancelled"]
        rows.append({"month": m, "completed": d["completed"], "rescheduled": d["rescheduled"],
                      "cancelled": d["cancelled"], "pending": 0,
                      "total": total, "candidates": d["candidates"]})
    return pd.DataFrame(rows)


def live_monthly(idf, from_date="2026-05-01"):
    if idf.empty or "date" not in idf.columns:
        return pd.DataFrame()
    today = date.today()
    f = idf[(idf["date"] >= pd.Timestamp(from_date)) & (idf["date"].dt.date <= today)].copy()
    if f.empty:
        return pd.DataFrame()
    f["month"] = f["date"].dt.to_period("M").astype(str)
    rows = []
    for m, g in f.groupby("month"):
        tc = g["task_status"].value_counts()
        rows.append({"month": m, "completed": int(tc.get("completed", 0)),
                      "rescheduled": int(tc.get("rescheduled", 0)),
                      "cancelled": int(tc.get("cancelled", 0)),
                      "pending": int(tc.get("pending", 0)),
                      "total": len(g),
                      "candidates": g["candidate_name"].nunique() if "candidate_name" in g.columns else 0})
    return pd.DataFrame(rows)


def expert_monthly(idf):
    if idf.empty or "date" not in idf.columns or "expert_name" not in idf.columns:
        return pd.DataFrame()
    d = idf.copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    rows = []
    for (month, expert), g in d.groupby(["month", "expert_name"]):
        tc = g["task_status"].value_counts()
        rows.append({"month": month, "expert_name": expert,
                      "completed": int(tc.get("completed", 0)),
                      "rescheduled": int(tc.get("rescheduled", 0)),
                      "cancelled": int(tc.get("cancelled", 0)),
                      "pending": int(tc.get("pending", 0)),
                      "total": len(g),
                      "candidates": g["candidate_name"].nunique() if "candidate_name" in g.columns else 0})
    return pd.DataFrame(rows)


def candidate_monthly_support(df):
    if df.empty or "date" not in df.columns or "candidate_name" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    rows = []
    for (month, cand), g in d.groupby(["month", "candidate_name"]):
        counts = {}
        for st_name in SUPPORT_TYPES:
            counts[st_name] = len(g[g["support_name"].str.lower() == st_name.lower()])
        row = {"month": month, "candidate_name": cand}
        row.update(counts)
        row["total"] = sum(counts.values())
        rows.append(row)
    return pd.DataFrame(rows)


def daily_agg(idf):
    if idf.empty or "date" not in idf.columns:
        return pd.DataFrame()
    d = idf.copy()
    d["day"] = d["date"].dt.date
    rows = []
    for day_val, g in d.groupby("day"):
        tc = g["task_status"].value_counts()
        rows.append({"day": day_val, "completed": int(tc.get("completed", 0)),
                      "rescheduled": int(tc.get("rescheduled", 0)),
                      "cancelled": int(tc.get("cancelled", 0)),
                      "pending": int(tc.get("pending", 0)),
                      "total": len(g),
                      "candidates": g["candidate_name"].nunique() if "candidate_name" in g.columns else 0})
    return pd.DataFrame(rows)


def to_excel_bytes(df):
    clean = df.copy()
    for col in clean.columns:
        try:
            clean[col] = clean[col].apply(
                lambda x: re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufeff\ufffe\uffff]', '', str(x)) if isinstance(x, str) else x
            )
        except Exception:
            pass
    return clean.to_csv(index=False).encode("utf-8")


def kpi_row(data):
    c = st.columns(5)
    c[0].metric("Completed", int(data.get("completed", 0)))
    c[1].metric("Rescheduled", int(data.get("rescheduled", 0)))
    c[2].metric("Cancelled", int(data.get("cancelled", 0)))
    c[3].metric("Pending", int(data.get("pending", 0)))
    c[4].metric("Candidates", int(data.get("candidates", 0)))


def stacked_bar(df, x="month", title="Monthly Counts"):
    fig = go.Figure()
    for s in TASK_ORDER:
        if s in df.columns:
            fig.add_trace(go.Bar(x=df[x], y=df[s], name=TASK_LABEL[s],
                                 marker_color=CLR[s], text=df[s], textposition="inside"))
    fig.update_layout(barmode="stack", title=title, height=460,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


def donut(data, title=""):
    labels = [TASK_LABEL[s] for s in TASK_ORDER]
    vals = [data.get(s, 0) for s in TASK_ORDER]
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=.5,
                           marker=dict(colors=[CLR[s] for s in TASK_ORDER]),
                           textinfo="label+value+percent"))
    fig.update_layout(title=title, height=400, showlegend=False)
    return fig


def trend_line(df, y, title, color="#3498db"):
    fig = go.Figure(go.Scatter(x=df["month"], y=df[y], mode="lines+markers+text",
                               text=df[y], textposition="top center",
                               line=dict(color=color, width=3), marker=dict(size=10)))
    fig.update_layout(title=title, height=400)
    return fig


def pct_line(df, num, title, color):
    d = df.copy()
    d["pct"] = (d[num] / d["total"] * 100).round(1)
    fig = go.Figure(go.Scatter(x=d["month"], y=d["pct"], mode="lines+markers+text",
                               text=d["pct"].astype(str) + "%", textposition="top center",
                               line=dict(color=color, width=3), marker=dict(size=10)))
    fig.update_layout(title=title, height=400, yaxis=dict(range=[0, max(d["pct"].max() + 10, 50)]))
    return fig


def h_bar_by_task(df, col, n=15, title=None):
    if col not in df.columns or "task_status" not in df.columns:
        return None
    top_items = df[col].value_counts().head(n).index
    filtered = df[df[col].isin(top_items)]
    pivot = filtered.groupby([col, "task_status"]).size().unstack(fill_value=0)
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=True).drop(columns="_total")
    fig = go.Figure()
    for s in TASK_ORDER:
        if s in pivot.columns:
            fig.add_trace(go.Bar(y=pivot.index, x=pivot[s], name=TASK_LABEL[s],
                                 orientation="h", marker_color=CLR[s],
                                 text=pivot[s], textposition="inside"))
    fig.update_layout(barmode="stack", title=title or "Top " + str(n) + " - " + col.title(),
                      height=max(400, n * 40),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


def h_bar(df, col, n=15, title=None):
    if col not in df.columns:
        return None
    c = df[col].value_counts().head(n).sort_values()
    if c.empty:
        return None
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                           marker_color="#3498db", text=c.values, textposition="outside"))
    fig.update_layout(title=title or "Top " + str(n) + " - " + col.title(), height=max(360, n * 38))
    return fig


def expert_stack(idf):
    if "expert_name" not in idf.columns or idf.empty:
        return None
    s = idf.groupby("expert_name")["task_status"].value_counts().unstack(fill_value=0)
    s["_t"] = s.sum(axis=1)
    s = s.sort_values("_t", ascending=False).head(15).drop(columns="_t")
    fig = go.Figure()
    for t in TASK_ORDER:
        if t in s.columns:
            fig.add_trace(go.Bar(y=s.index, x=s[t], name=TASK_LABEL[t],
                                 orientation="h", marker_color=CLR[t]))
    fig.update_layout(barmode="stack", title="Top 15 Experts", height=550,
                      yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


def day_of_week_chart(idf):
    if idf.empty or "date" not in idf.columns:
        return None
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    c = idf["date"].dt.day_name().value_counts().reindex(order, fill_value=0)
    fig = go.Figure(go.Bar(x=c.index, y=c.values, marker_color="#9b59b6",
                           text=c.values, textposition="outside"))
    fig.update_layout(title="By Day of Week", height=400)
    return fig


def round_charts(df, title_suffix=""):
    if "round_name" not in df.columns or df.empty:
        return
    rc = df["round_name"].value_counts()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(labels=rc.index, values=rc.values, hole=.4,
                              textinfo="label+value+percent"))
        fig.update_layout(title="Round Distribution" + title_suffix, height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rt = df.groupby(["round_name", "task_status"]).size().unstack(fill_value=0)
        fig2 = go.Figure()
        for s in TASK_ORDER:
            if s in rt.columns:
                fig2.add_trace(go.Bar(x=rt.index, y=rt[s], name=TASK_LABEL[s],
                                      marker_color=CLR[s], text=rt[s], textposition="inside"))
        fig2.update_layout(barmode="stack", title="Round x Task" + title_suffix, height=420)
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  TRANSCRIPT ANALYZER VIEW (RoBERTa ONNX)
# ═══════════════════════════════════════════════════════════════════

def transcript_analyzer_view():
    st.header("Transcript Sentiment Analyzer")
    st.caption("Powered by RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) via ONNX Runtime")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Settings")
    chunk_mode = st.sidebar.radio(
        "Split transcript into:",
        ["Full Text (single score)", "Paragraphs", "Sentences"],
        index=1,
        key="ta_chunk_mode",
    )
    chunk_map = {
        "Full Text (single score)": "full",
        "Paragraphs": "paragraph",
        "Sentences": "sentence",
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Score Range**\n\n"
        "- **Positive**: score >= +20%\n"
        "- **Neutral**: -20% < score < +20%\n"
        "- **Negative**: score <= -20%\n\n"
        "Score = (P(pos) - P(neg)) x 100"
    )

    transcript = st.text_area(
        "Paste your transcript below:",
        height=300,
        placeholder="Paste the full interview transcript, feedback, or any text here...\n\nSeparate paragraphs with blank lines for paragraph-level analysis.",
        key="ta_transcript",
    )

    analyze_btn = st.button("Analyze Sentiment", type="primary", key="ta_analyze_btn")

    if analyze_btn and transcript.strip():
        with st.spinner("Loading RoBERTa ONNX model & analyzing..."):
            mode = chunk_map[chunk_mode]

            overall = roberta_score_text(transcript)
            ov_score = overall["score"]
            ov_color = sentiment_color(ov_score)
            ov_label = sentiment_label(ov_score)

            st.markdown("---")
            st.subheader("Overall Sentiment: " + ov_label + " (" + f"{ov_score:+.1f}%" + ")")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Overall Score", f"{ov_score:+.1f}%")
            k2.metric("P(Positive)", f"{overall['p_positive']:.1f}%")
            k3.metric("P(Neutral)", f"{overall['p_neutral']:.1f}%")
            k4.metric("P(Negative)", f"{overall['p_negative']:.1f}%")

            fig_donut = go.Figure(go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[overall["p_positive"], overall["p_neutral"], overall["p_negative"]],
                hole=0.55,
                marker=dict(colors=["#2ecc71", "#f39c12", "#e74c3c"]),
                textinfo="label+percent",
            ))
            fig_donut.update_layout(
                title="Probability Distribution",
                height=350, showlegend=False,
                annotations=[dict(text=f"{ov_score:+.1f}%", x=0.5, y=0.5,
                                  font_size=24, font_color=ov_color, showarrow=False)],
            )

            if mode != "full":
                chunks = roberta_score_chunks(transcript, chunk_mode=mode)

                if chunks:
                    valid_chunks = [c for c in chunks if c["score"] is not None]
                    scores = [c["score"] for c in valid_chunks]

                    st.markdown("---")
                    unit_name = "Paragraph" if mode == "paragraph" else "Sentence"
                    st.subheader(unit_name + "-Level Breakdown (" + str(len(chunks)) + " " + unit_name.lower() + "s)")

                    if scores:
                        pos_count = sum(1 for s in scores if s >= 20)
                        neu_count = sum(1 for s in scores if -20 < s < 20)
                        neg_count = sum(1 for s in scores if s <= -20)
                        avg_score = round(np.mean(scores), 1)

                        s1, s2, s3, s4, s5 = st.columns(5)
                        s1.metric("Avg " + unit_name + " Score", f"{avg_score:+.1f}%")
                        s2.metric("Total " + unit_name + "s", len(chunks))
                        s3.metric("Positive", pos_count)
                        s4.metric("Neutral", neu_count)
                        s5.metric("Negative", neg_count)

                    col_donut, col_bar = st.columns(2)
                    with col_donut:
                        st.plotly_chart(fig_donut, use_container_width=True)
                    with col_bar:
                        if scores:
                            bar_colors = [sentiment_color(s) for s in scores]
                            bar_labels = [unit_name[0] + str(i + 1) for i in range(len(scores))]
                            fig_bar = go.Figure(go.Bar(
                                x=bar_labels, y=scores,
                                marker_color=bar_colors,
                                text=[f"{s:+.1f}%" for s in scores],
                                textposition="outside",
                            ))
                            fig_bar.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                            fig_bar.add_hline(y=20, line_dash="dot", line_color="#2ecc71", opacity=0.3)
                            fig_bar.add_hline(y=-20, line_dash="dot", line_color="#e74c3c", opacity=0.3)
                            fig_bar.update_layout(
                                title="Sentiment by " + unit_name,
                                height=350,
                                yaxis_title="Sentiment Score (%)",
                                yaxis=dict(range=[
                                    min(-60, min(scores) - 15),
                                    max(60, max(scores) + 15),
                                ]),
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                    if len(scores) > 1:
                        st.markdown("---")
                        st.subheader("Sentiment Flow Through Transcript")
                        flow_colors = [sentiment_color(s) for s in scores]
                        fig_flow = go.Figure()
                        fig_flow.add_trace(go.Scatter(
                            x=list(range(1, len(scores) + 1)),
                            y=scores,
                            mode="lines+markers+text",
                            text=[f"{s:+.1f}%" for s in scores],
                            textposition="top center",
                            line=dict(color="#8e44ad", width=3),
                            marker=dict(size=12, color=flow_colors,
                                        line=dict(width=2, color="white")),
                        ))
                        fig_flow.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig_flow.add_hline(y=20, line_dash="dot", line_color="#2ecc71", opacity=0.3)
                        fig_flow.add_hline(y=-20, line_dash="dot", line_color="#e74c3c", opacity=0.3)
                        fig_flow.update_layout(
                            height=400,
                            xaxis_title=unit_name + " Number",
                            yaxis_title="Sentiment Score (%)",
                            yaxis=dict(range=[
                                min(-60, min(scores) - 15),
                                max(60, max(scores) + 15),
                            ]),
                        )
                        st.plotly_chart(fig_flow, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Detailed " + unit_name + " Scores")
                    for i, chunk in enumerate(chunks):
                        c_label = chunk["label"]
                        score_str = f"{chunk['score']:+.1f}%" if chunk["score"] is not None else "N/A"

                        with st.expander(
                            unit_name + " " + str(i + 1) + ": " + c_label + " (" + score_str + ") — " + chunk["text"]
                        ):
                            if chunk["score"] is not None:
                                mc = st.columns(4)
                                mc[0].metric("Score", score_str)
                                mc[1].metric("P(Positive)", f"{chunk['p_positive']:.1f}%")
                                mc[2].metric("P(Neutral)", f"{chunk['p_neutral']:.1f}%")
                                mc[3].metric("P(Negative)", f"{chunk['p_negative']:.1f}%")
                            st.markdown("**Full text:**")
                            st.text(chunk["full_text"])

            else:
                col_donut, col_info = st.columns(2)
                with col_donut:
                    st.plotly_chart(fig_donut, use_container_width=True)
                with col_info:
                    st.markdown(
                        "### Interpretation\n\n"
                        "The overall sentiment score is **" + f"{ov_score:+.1f}%" + "** (" + ov_label + ").\n\n"
                        "- **P(Positive)**: " + f"{overall['p_positive']:.1f}%" + "\n"
                        "- **P(Neutral)**: " + f"{overall['p_neutral']:.1f}%" + "\n"
                        "- **P(Negative)**: " + f"{overall['p_negative']:.1f}%" + "\n\n"
                        "Score = (P(positive) - P(negative)) x 100, range -100% to +100%."
                    )

    elif analyze_btn:
        st.warning("Please paste some text before clicking Analyze.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════

def main():
    title_col, dl_col = st.columns([4, 1])
    with title_col:
        st.title("Vizva Support Dashboard")

    try:
        with st.spinner("Fetching latest data from API..."):
            raw = fetch_all_data()
            raw = normalize(raw)
    except Exception as e:
        st.error("Failed to fetch data: " + str(e))
        st.stop()

    raw = filter_current_year(raw)
    if raw.empty:
        st.error("No data found for current year.")
        st.stop()

    all_case_df = raw.copy()
    active_expert_df = filter_active_experts(raw)

    # ── Add sentiment scores to the active expert data ONCE ──────
    active_expert_df, _fb_col = add_sentiment_column(active_expert_df)

    with dl_col:
        st.write("")
        st.write("")
        excel_data = to_excel_bytes(all_case_df)
        st.download_button(
            label="Download Raw Data",
            data=excel_data,
            file_name="vizva_raw_data_" + date.today().strftime("%Y%m%d") + ".csv",
            mime="text/csv"
        )

    st.sidebar.header("Support Type")
    selected_support = st.sidebar.selectbox("Select Support Type", SUPPORT_TYPES, index=0)
    support_label = selected_support

    support_df = get_by_support(active_expert_df, selected_support)

    # ── Add start_time columns ONCE for Interview Support ────────
    if selected_support == "Interview Support" and "start_time" in support_df.columns:
        support_df = add_start_time_columns(support_df)

    st.sidebar.markdown("---")
    st.sidebar.metric("Total Cases (This Year)", len(all_case_df))
    for stype in SUPPORT_TYPES:
        count = len(get_by_support(active_expert_df, stype))
        st.sidebar.metric(stype, count)
    st.sidebar.caption("Only active experts shown")
    st.sidebar.caption("Data refreshes every 10 minutes")

    if st.sidebar.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    if support_df.empty:
        st.warning("No " + support_label + " rows found for active experts.")
        st.stop()

    st.sidebar.success("Showing " + str(len(support_df)) + " " + support_label + " rows")

    # ── Sidebar: Overall sentiment gauge for selected support type ─
    overall_stats = get_sentiment_stats(support_df)
    if overall_stats:
        avg = overall_stats["avg"]
        st.sidebar.markdown("---")
        st.sidebar.metric(
            "Avg Sentiment (" + support_label + ")",
            f"{avg:+.1f}%",
            delta=sentiment_label(avg),
            delta_color="normal" if avg >= 0 else "inverse",
        )

    # ── Sidebar: Start Time summary for Interview Support ────────
    if selected_support == "Interview Support" and "start_hour" in support_df.columns:
        valid_st = support_df["start_hour"].dropna()
        if not valid_st.empty:
            peak_h = int(valid_st.value_counts().idxmax())
            peak_lbl = datetime(2000, 1, 1, peak_h).strftime("%I:%M %p")
            st.sidebar.metric("Peak Interview Hour", peak_lbl)

    hist = hist_monthly_df(selected_support)
    live = live_monthly(support_df)
    if not hist.empty and not live.empty:
        monthly = pd.concat([hist, live], ignore_index=True).drop_duplicates("month", keep="last")
    elif not hist.empty:
        monthly = hist
    elif not live.empty:
        monthly = live
    else:
        monthly = pd.DataFrame()

    st.sidebar.markdown("---")
    st.sidebar.header("View")
    view = st.sidebar.radio("Navigation", ["Todays Snapshot", "Monthly Overview",
                                  "Daily Drill-Down", "Deep-Dive Analytics",
                                  "Transcript Analyzer"],
                            label_visibility="collapsed")

    # ======= TRANSCRIPT ANALYZER =======
    if view == "Transcript Analyzer":
        transcript_analyzer_view()
        st.sidebar.markdown("---")
        st.sidebar.caption("Vizva Dashboard v17.0 | API-powered | Active Experts Only | RoBERTa ONNX | Start Time Analytics")
        return

    # ======= TODAY =======
    if view == "Todays Snapshot":
        st.header("Todays Snapshot - " + support_label)
        today = date.today()
        today_df = support_df[support_df["date"].dt.date == today]

        st.caption(today.strftime("%A, %B %d, %Y"))

        if today_df.empty:
            st.info("No " + support_label + " scheduled for today.")
            kd = {"completed": 0, "rescheduled": 0, "cancelled": 0, "pending": 0, "candidates": 0}
        else:
            tc = today_df["task_status"].value_counts()
            kd = {s: int(tc.get(s, 0)) for s in TASK_ORDER}
            kd["candidates"] = today_df["candidate_name"].nunique()

        kpi_row(kd)

        # ── TODAY'S SENTIMENT KPI (compact, right after task KPIs) ─
        if not today_df.empty:
            today_stats = get_sentiment_stats(today_df)
            if today_stats:
                st.markdown("---")
                render_sentiment_kpi(today_stats, title="Today's Feedback Sentiment")

        # ── TODAY'S START TIME INSIGHTS (Interview Support only) ──
        if selected_support == "Interview Support" and not today_df.empty and "start_hour" in today_df.columns:
            st.markdown("---")
            render_start_time_insights(today_df, title_suffix=" - " + str(today))

        if not today_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(donut(kd, "Task Split - " + str(today)), use_container_width=True)
            with c2:
                fig = h_bar_by_task(today_df, "company_name", 10, "Companies - " + str(today))
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig = h_bar_by_task(today_df, "candidate_name", 10, "Candidates - " + str(today))
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with c4:
                fig = h_bar_by_task(today_df, "expert_name", 10, "Experts - " + str(today))
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            if selected_support == "Interview Support":
                st.markdown("---")
                st.subheader("Round Breakdown - " + str(today))
                round_charts(today_df, " - " + str(today))

            # ── SENTIMENT DONUT + HISTOGRAM — TODAY ───────────────
            st.markdown("---")
            render_sentiment_section(today_df, section_title="Sentiment Analysis - " + str(today))

            # ── FEEDBACK WORD CLOUD — TODAY ───────────────────────
            st.markdown("---")
            feedback_texts = extract_feedback_texts(today_df)
            render_wordcloud_section(
                feedback_texts,
                section_title="Feedback Word Cloud - " + str(today),
            )

            with st.expander("Raw Data (includes Sentiment Score)"):
                st.dataframe(today_df, use_container_width=True, height=400)

        # ── ABOUT TO MOVE TO MARKET (Resume Understanding with pending status) ──
        st.markdown("---")
        st.subheader("About to Move to Market")
        st.caption("Resume Understanding candidates with pending status — ready to enter the market")

        resume_df = get_by_support(active_expert_df, "Resume Understanding")
        if not resume_df.empty and "task_status" in resume_df.columns:
            market_df = resume_df[resume_df["task_status"] == "pending"].copy()

            if not market_df.empty:
                mc = st.columns(3)
                mc[0].metric("Total Pending", len(market_df))
                mc[1].metric("Candidates", market_df["candidate_name"].nunique() if "candidate_name" in market_df.columns else 0)
                mc[2].metric("Experts", market_df["expert_name"].nunique() if "expert_name" in market_df.columns else 0)

                if "candidate_name" in market_df.columns:
                    cand_counts = market_df["candidate_name"].value_counts().head(15)
                    fig = go.Figure(go.Bar(
                        y=cand_counts.index, x=cand_counts.values,
                        orientation="h", marker_color="#1abc9c",
                        text=cand_counts.values, textposition="outside"
                    ))
                    fig.update_layout(title="Candidates - About to Move to Market",
                                      height=max(400, len(cand_counts) * 35),
                                      yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

                if "candidate_technology" in market_df.columns:
                    tech_counts = market_df["candidate_technology"].value_counts().head(15)
                    if not tech_counts.empty:
                        fig = go.Figure(go.Bar(
                            y=tech_counts.index, x=tech_counts.values,
                            orientation="h", marker_color="#9b59b6",
                            text=tech_counts.values, textposition="outside"
                        ))
                        fig.update_layout(title="Technologies - About to Move to Market",
                                          height=max(400, len(tech_counts) * 35),
                                          yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

                with st.expander("Full List - About to Move to Market"):
                    display_cols = [c for c in ["candidate_name", "expert_name",
                                                "candidate_technology", "date", "task_status"]
                                    if c in market_df.columns]
                    st.dataframe(market_df[display_cols] if display_cols else market_df,
                                 use_container_width=True, height=400)
            else:
                st.info("No Resume Understanding candidates with pending status found.")
        else:
            st.info("No Resume Understanding data available.")

    # ======= MONTHLY =======
    elif view == "Monthly Overview":
        if monthly.empty:
            st.warning("No monthly data available for " + support_label)
            st.stop()

        has_hist = selected_support in HIST
        title_range = "(Jul 2025 - Present)" if has_hist else "(2026 - Present)"
        st.header("Monthly Overview - " + support_label + " " + title_range)

        current_month_str = date.today().strftime("%Y-%m")
        current_month_row = monthly[monthly["month"] == current_month_str]
        if not current_month_row.empty:
            latest = current_month_row.iloc[0]
        else:
            latest = monthly.iloc[-1]

        c = st.columns(6)
        c[0].metric("Month", latest["month"])
        c[1].metric("Completed", int(latest["completed"]))
        c[2].metric("Rescheduled", int(latest["rescheduled"]))
        c[3].metric("Cancelled", int(latest["cancelled"]))
        c[4].metric("Pending", int(latest["pending"]))
        c[5].metric("Candidates", int(latest["candidates"]))

        # ── CURRENT MONTH SENTIMENT KPI ──────────────────────────
        current_month_data = support_df[support_df["date"].dt.to_period("M").astype(str) == current_month_str]
        if not current_month_data.empty:
            cm_stats = get_sentiment_stats(current_month_data)
            if cm_stats:
                avg = cm_stats["avg"]
                st.columns(6)[0].empty()  # spacer
                sent_cols = st.columns(4)
                sent_cols[0].metric("Avg Sentiment (This Month)", f"{avg:+.1f}%",
                                    delta=sentiment_label(avg),
                                    delta_color="normal" if avg >= 0 else "inverse")
                sent_cols[1].metric("Positive", cm_stats["positive"])
                sent_cols[2].metric("Neutral", cm_stats["neutral"])
                sent_cols[3].metric("Negative", cm_stats["negative"])

        st.plotly_chart(stacked_bar(monthly, title="Monthly " + support_label + " Counts"), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(trend_line(monthly, "candidates", "Unique Candidates per Month"), use_container_width=True)
        with c2:
            st.plotly_chart(pct_line(monthly, "completed", "Completion Rate %", "#2ecc71"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(pct_line(monthly, "cancelled", "Cancellation Rate %", "#e74c3c"), use_container_width=True)
        with c4:
            st.plotly_chart(donut(latest.to_dict(), "Task Split - " + str(latest["month"])), use_container_width=True)

        st.plotly_chart(trend_line(monthly, "total", "Total " + support_label + " per Month", "#8e44ad"), use_container_width=True)

        # ── START TIME INSIGHTS — OVERALL (Interview Support) ────
        if selected_support == "Interview Support" and "start_hour" in support_df.columns:
            st.markdown("---")
            render_start_time_insights(support_df, title_suffix=" - All Months")

            st.markdown("---")
            render_monthly_start_time_trend(support_df, title_suffix=" - " + support_label)

        # ── MONTHLY SENTIMENT TREND ──────────────────────────────
        st.markdown("---")
        st.subheader("Sentiment Trend Across Months")

        sent_monthly = monthly_sentiment_trend(support_df)
        if not sent_monthly.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = render_sentiment_trend_chart(sent_monthly, "Monthly Avg Sentiment - " + support_label)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = render_sentiment_stacked_bar(sent_monthly, "Monthly Sentiment Breakdown")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("Monthly Sentiment Data"):
                st.dataframe(sent_monthly, use_container_width=True)
        else:
            st.info("No feedback data available for sentiment trend.")

        with st.expander("Monthly Data Table"):
            st.dataframe(monthly, use_container_width=True)

        # ── FEEDBACK ANALYSIS (Word Cloud + Sentiment per month) ──
        st.markdown("---")
        st.subheader("Feedback Analysis")

        if "date" in support_df.columns:
            fb_df = support_df.copy()
            fb_df["month"] = fb_df["date"].dt.to_period("M").astype(str)
            fb_months = sorted(fb_df["month"].unique())

            fb_month_options = ["All Months"] + fb_months
            sel_fb_month = st.selectbox(
                "Select Month for Feedback Analysis",
                fb_month_options,
                index=len(fb_month_options) - 1,
                key="fb_month_sel",
            )

            if sel_fb_month == "All Months":
                fb_subset = fb_df
            else:
                fb_subset = fb_df[fb_df["month"] == sel_fb_month]

            render_sentiment_section(fb_subset,
                                     section_title="Sentiment - " + support_label + " - " + sel_fb_month)

            st.markdown("")

            feedback_texts = extract_feedback_texts(fb_subset)
            render_wordcloud_section(
                feedback_texts,
                section_title="Feedback Word Cloud - " + support_label + " - " + sel_fb_month,
            )

        # -- ROUND WISE MONTHLY (Interview Support Only) --
        if selected_support == "Interview Support" and "round_name" in support_df.columns:
            st.markdown("---")
            st.subheader("Round-wise Monthly Breakdown")

            month_round_df = support_df.copy()
            month_round_df["month"] = month_round_df["date"].dt.to_period("M").astype(str)

            round_months = sorted(month_round_df["month"].unique())
            sel_round_month = st.selectbox("Select Month  ", round_months, index=len(round_months) - 1, key="round_month")

            round_month_data = month_round_df[month_round_df["month"] == sel_round_month]

            if not round_month_data.empty:
                round_charts(round_month_data, " - " + sel_round_month)
            else:
                st.info("No data for " + sel_round_month)

        # -- EXPERT WISE MONTHLY --
        st.markdown("---")
        st.subheader("Expert-wise Monthly Breakdown")

        exp_monthly = expert_monthly(support_df)
        if not exp_monthly.empty:
            months_available = sorted(exp_monthly["month"].unique())
            selected_month = st.selectbox("Select Month", months_available, index=len(months_available) - 1)

            month_exp = exp_monthly[exp_monthly["month"] == selected_month].sort_values("total", ascending=False)

            if not month_exp.empty:
                fig = go.Figure()
                for s in TASK_ORDER:
                    if s in month_exp.columns:
                        fig.add_trace(go.Bar(x=month_exp["expert_name"], y=month_exp[s],
                                             name=TASK_LABEL[s], marker_color=CLR[s],
                                             text=month_exp[s], textposition="inside"))
                fig.update_layout(barmode="stack", title="Expert Counts - " + selected_month, height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(month_exp[["expert_name", "completed", "rescheduled", "cancelled",
                                        "pending", "total", "candidates"]],
                             use_container_width=True)

            st.subheader("Expert Trend Across Months")
            experts_list = sorted(exp_monthly["expert_name"].unique())
            selected_expert = st.selectbox("Select Expert", experts_list)

            expert_trend = exp_monthly[exp_monthly["expert_name"] == selected_expert].sort_values("month")
            if not expert_trend.empty:
                fig = go.Figure()
                for s in TASK_ORDER:
                    if s in expert_trend.columns:
                        fig.add_trace(go.Scatter(x=expert_trend["month"], y=expert_trend[s],
                                                 mode="lines+markers", name=TASK_LABEL[s],
                                                 line=dict(color=CLR[s])))
                fig.update_layout(title=selected_expert + " - Monthly Trend", height=400)
                st.plotly_chart(fig, use_container_width=True)

        # -- CANDIDATE WISE MONTHLY --
        st.markdown("---")
        st.subheader("Candidate-wise Monthly Counts (All Support Types)")

        cand_monthly = candidate_monthly_support(active_expert_df)
        if not cand_monthly.empty:
            cand_months = sorted(cand_monthly["month"].unique())
            sel_cand_month = st.selectbox("Select Month ", cand_months, index=len(cand_months) - 1, key="cand_month")

            cand_month_data = cand_monthly[cand_monthly["month"] == sel_cand_month].sort_values("total", ascending=False)

            if not cand_month_data.empty:
                support_colors = {"Interview Support": "#3498db", "Assessment Support": "#e67e22",
                                  "Mock Interview": "#9b59b6", "Resume Understanding": "#1abc9c"}
                fig = go.Figure()
                for stype in SUPPORT_TYPES:
                    if stype in cand_month_data.columns:
                        fig.add_trace(go.Bar(x=cand_month_data["candidate_name"],
                                             y=cand_month_data[stype],
                                             name=stype, marker_color=support_colors.get(stype, "#95a5a6"),
                                             text=cand_month_data[stype], textposition="inside"))
                fig.update_layout(barmode="stack",
                                  title="Candidate Counts - " + sel_cand_month,
                                  height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(cand_month_data, use_container_width=True)

    # ======= DAILY =======
    elif view == "Daily Drill-Down":
        st.header("Daily Drill-Down - " + support_label)
        min_d = support_df["date"].dt.date.min()
        max_d = support_df["date"].dt.date.max()

        ca, cb = st.sidebar.columns(2)
        start = ca.date_input("From", value=max(min_d, date(2026, 5, 1)),
                              min_value=min_d, max_value=max_d)
        end = cb.date_input("To", value=max_d, min_value=min_d, max_value=max_d)

        mask = (support_df["date"].dt.date >= start) & (support_df["date"].dt.date <= end)
        period = support_df[mask]
        dd = daily_agg(period)

        if dd.empty:
            st.info("No data in selected range.")
        else:
            kd = {s: dd[s].sum() for s in TASK_ORDER}
            kd["candidates"] = period["candidate_name"].nunique()
            kpi_row(kd)

            # ── PERIOD SENTIMENT KPI ─────────────────────────────
            period_stats = get_sentiment_stats(period)
            if period_stats:
                render_sentiment_kpi(period_stats,
                                     title="Feedback Sentiment (" + str(start) + " to " + str(end) + ")")

            dd_plot = dd.copy()
            dd_plot["day"] = pd.to_datetime(dd_plot["day"])
            fig = go.Figure()
            for s in TASK_ORDER:
                fig.add_trace(go.Scatter(x=dd_plot["day"], y=dd_plot[s], mode="lines+markers",
                                         name=TASK_LABEL[s], line=dict(color=CLR[s])))
            fig.update_layout(title="Daily Trend (" + str(start) + " to " + str(end) + ")", height=450)
            st.plotly_chart(fig, use_container_width=True)

            st.plotly_chart(stacked_bar(dd_plot, x="day", title="Daily Stacked View"), use_container_width=True)

            # ── DAILY SENTIMENT TREND LINE ───────────────────────
            daily_sent = daily_sentiment_agg(period)
            if not daily_sent.empty:
                st.markdown("---")
                st.subheader("Daily Sentiment Trend")
                d_colors = [sentiment_color(v) for v in daily_sent["avg_sentiment"]]
                fig_ds = go.Figure()
                fig_ds.add_trace(go.Scatter(
                    x=pd.to_datetime(daily_sent["day"]),
                    y=daily_sent["avg_sentiment"],
                    mode="lines+markers+text",
                    text=daily_sent["avg_sentiment"].apply(lambda v: f"{v:+.1f}%"),
                    textposition="top center",
                    line=dict(color="#8e44ad", width=2),
                    marker=dict(size=10, color=d_colors, line=dict(width=1, color="white")),
                ))
                fig_ds.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_ds.add_hline(y=20, line_dash="dot", line_color="#2ecc71", opacity=0.3)
                fig_ds.add_hline(y=-20, line_dash="dot", line_color="#e74c3c", opacity=0.3)
                fig_ds.update_layout(
                    title="Daily Avg Sentiment (" + str(start) + " to " + str(end) + ")",
                    height=420, yaxis_title="Avg Sentiment (%)",
                    yaxis=dict(range=[
                        min(-50, daily_sent["avg_sentiment"].min() - 15),
                        max(50, daily_sent["avg_sentiment"].max() + 15)
                    ]),
                )
                st.plotly_chart(fig_ds, use_container_width=True)

            with st.expander("Daily Table"):
                disp = dd.copy()
                disp["day"] = pd.to_datetime(disp["day"]).dt.strftime("%Y-%m-%d")
                st.dataframe(disp, use_container_width=True)

            if selected_support == "Interview Support" and "round_name" in period.columns and not period.empty:
                st.markdown("---")
                st.subheader("Round Breakdown (" + str(start) + " to " + str(end) + ")")
                round_charts(period, " (" + str(start) + " to " + str(end) + ")")

        st.sidebar.markdown("---")
        single = st.sidebar.date_input("Inspect single day", value=max_d,
                                       min_value=min_d, max_value=max_d, key="single")
        sdf = support_df[support_df["date"].dt.date == single]
        if not sdf.empty:
            st.subheader("Details - " + str(single))
            tc = sdf["task_status"].value_counts()
            kd2 = {s: int(tc.get(s, 0)) for s in TASK_ORDER}
            kd2["candidates"] = sdf["candidate_name"].nunique()
            kpi_row(kd2)

            # ── SINGLE DAY SENTIMENT KPI ─────────────────────────
            single_stats = get_sentiment_stats(sdf)
            if single_stats:
                render_sentiment_kpi(single_stats, title="Sentiment - " + str(single))

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(donut(kd2, "Split - " + str(single)), use_container_width=True)
            with c2:
                fig = h_bar_by_task(sdf, "company_name", 10, "Companies - " + str(single))
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig = h_bar_by_task(sdf, "candidate_name", 10, "Candidates - " + str(single))
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with c4:
                fig = h_bar_by_task(sdf, "expert_name", 10, "Experts - " + str(single))
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # ── SINGLE DAY SENTIMENT DONUT + HISTOGRAM ───────────
            if single_stats:
                st.markdown("---")
                col_d, col_h = st.columns(2)
                with col_d:
                    fig = render_sentiment_donut(single_stats, "Sentiment Split - " + str(single))
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with col_h:
                    fig = render_sentiment_histogram(sdf, "Score Distribution - " + str(single))
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            if selected_support == "Interview Support" and "round_name" in sdf.columns:
                st.markdown("---")
                st.subheader("Round Breakdown - " + str(single))
                round_charts(sdf, " - " + str(single))

            st.dataframe(sdf, use_container_width=True)
        elif single:
            st.info("No " + support_label + " on " + str(single))

    # ======= DEEP DIVE =======
    elif view == "Deep-Dive Analytics":
        st.header("Deep-Dive Analytics - " + support_label)

        tab_names = ["Experts", "Companies", "Rounds",
                     "Day of Week", "All Support Types",
                     "Candidates", "Technology"]
        if selected_support == "Interview Support" and "start_hour" in support_df.columns:
            tab_names.append("Start Time Analysis")

        tabs = st.tabs(tab_names)

        with tabs[0]:
            fig = expert_stack(support_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if "expert_name" in support_df.columns:
                ea = support_df.groupby("expert_name").agg(
                    Total=("task_status", "size"),
                    Completed=("task_status", lambda x: (x == "completed").sum()),
                    Rescheduled=("task_status", lambda x: (x == "rescheduled").sum()),
                    Cancelled=("task_status", lambda x: (x == "cancelled").sum()),
                    Pending=("task_status", lambda x: (x == "pending").sum()),
                    Candidates=("candidate_name", "nunique"),
                    Companies=("company_name", "nunique"),
                    Avg_Sentiment=("sentiment_score", lambda x: round(x.dropna().mean(), 1) if x.dropna().any() else None),
                ).reset_index()
                ea["Completion_Pct"] = (ea["Completed"] / ea["Total"] * 100).round(1)
                st.dataframe(ea.sort_values("Total", ascending=False), use_container_width=True)

        with tabs[1]:
            fig = h_bar_by_task(support_df, "company_name", 20, "Top 20 Companies by Volume")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if "company_name" in support_df.columns:
                top = support_df["company_name"].value_counts().head(15).index
                ct = support_df[support_df["company_name"].isin(top)]
                pv = ct.groupby(["company_name", "task_status"]).size().unstack(fill_value=0)
                fig2 = px.imshow(pv, text_auto=True, aspect="auto",
                                color_continuous_scale="Blues", title="Company x Task Heatmap")
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[2]:
            if "round_name" in support_df.columns:
                rc = support_df["round_name"].value_counts()
                fig = go.Figure(go.Pie(labels=rc.index, values=rc.values, hole=.4,
                                      textinfo="label+value+percent"))
                fig.update_layout(title="Round Distribution", height=420)
                st.plotly_chart(fig, use_container_width=True)

                rt = support_df.groupby(["round_name", "task_status"]).size().unstack(fill_value=0)
                fig2 = px.imshow(rt, text_auto=True, aspect="auto",
                                color_continuous_scale="Oranges", title="Round x Task Heatmap")
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[3]:
            fig = day_of_week_chart(support_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if "date" in support_df.columns:
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                tmp = support_df.copy()
                tmp["dow"] = tmp["date"].dt.day_name()
                pv = tmp.groupby(["dow", "task_status"]).size().unstack(fill_value=0).reindex(order)
                fig2 = px.imshow(pv, text_auto=True, aspect="auto",
                                color_continuous_scale="Purples", title="Day x Task Heatmap")
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[4]:
            if "support_name" in all_case_df.columns:
                sc = all_case_df["support_name"].value_counts()
                fig = go.Figure(go.Bar(x=sc.index, y=sc.values, marker_color="#1abc9c",
                                       text=sc.values, textposition="outside"))
                fig.update_layout(title="All Support Types (Current Year)", height=400)
                st.plotly_chart(fig, use_container_width=True)

                st_task = all_case_df.groupby(["support_name", "task_status"]).size().unstack(fill_value=0)
                fig2 = px.imshow(st_task, text_auto=True, aspect="auto",
                                color_continuous_scale="Teal", title="Support Type x Task")
                fig2.update_layout(height=350)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[5]:
            if "candidate_name" in support_df.columns:
                ca_agg = support_df.groupby("candidate_name").agg(
                    Total=("task_status", "size"),
                    Completed=("task_status", lambda x: (x == "completed").sum()),
                    Rescheduled=("task_status", lambda x: (x == "rescheduled").sum()),
                    Cancelled=("task_status", lambda x: (x == "cancelled").sum()),
                    Pending=("task_status", lambda x: (x == "pending").sum()),
                    Companies=("company_name", "nunique"),
                    Experts=("expert_name", "nunique"),
                    Avg_Sentiment=("sentiment_score", lambda x: round(x.dropna().mean(), 1) if x.dropna().any() else None),
                ).reset_index()
                ca_agg["Completion_Pct"] = (ca_agg["Completed"] / ca_agg["Total"] * 100).round(1)
                ca_agg = ca_agg.sort_values("Total", ascending=False)
                st.subheader("Candidate Performance Table")
                st.dataframe(ca_agg, use_container_width=True)

                top30 = ca_agg.head(20).sort_values("Total")
                fig = go.Figure()
                fig.add_trace(go.Bar(y=top30["candidate_name"], x=top30["Completed"],
                                     name="Completed", orientation="h", marker_color="#2ecc71"))
                fig.add_trace(go.Bar(y=top30["candidate_name"], x=top30["Rescheduled"],
                                     name="Rescheduled", orientation="h", marker_color="#f39c12"))
                fig.add_trace(go.Bar(y=top30["candidate_name"], x=top30["Cancelled"],
                                     name="Cancelled", orientation="h", marker_color="#e74c3c"))
                fig.add_trace(go.Bar(y=top30["candidate_name"], x=top30["Pending"],
                                     name="Pending", orientation="h", marker_color="#3498db"))
                fig.update_layout(barmode="stack", title="Top 20 Candidates", height=600,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        with tabs[6]:
            if "candidate_technology" in support_df.columns:
                fig = h_bar_by_task(support_df, "candidate_technology", 20, "Top 20 Technologies")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                top_tech = support_df["candidate_technology"].value_counts().head(15).index
                tt = support_df[support_df["candidate_technology"].isin(top_tech)]
                pv = tt.groupby(["candidate_technology", "task_status"]).size().unstack(fill_value=0)
                fig2 = px.imshow(pv, text_auto=True, aspect="auto",
                                color_continuous_scale="Greens", title="Technology x Task Heatmap")
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No technology column found.")

        # ── START TIME ANALYSIS TAB (Interview Support only) ─────
        if selected_support == "Interview Support" and "start_hour" in support_df.columns and len(tabs) > 7:
            with tabs[7]:
                render_start_time_insights(support_df, title_suffix=" - All Data")
                st.markdown("---")
                render_monthly_start_time_trend(support_df, title_suffix="")

    st.sidebar.markdown("---")
    st.sidebar.caption("Vizva Dashboard v17.0 | API-powered | Active Experts Only | RoBERTa ONNX | Start Time Analytics")


# ================================================================
# AUTHENTICATION LAYER
# ================================================================

def login():
    st.title("Vizva Secure Access")
    with st.container():
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if user == st.secrets["VIZVA_USERNAME"] and pw == st.secrets["VIZVA_PASSWORD"]:
                st.session_state["authenticated"] = True
                st.session_state["login_time"] = datetime.now()
                st.rerun()
            else:
                st.error("Invalid Username or Password")


def check_timeout():
    if "login_time" in st.session_state:
        delta = datetime.now() - st.session_state["login_time"]
        if delta.total_seconds() > 3600:
            st.session_state["authenticated"] = False
            st.warning("Session expired. Please log in again.")
            st.rerun()


if __name__ == "__main__":
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login()
    else:
        check_timeout()
        if st.sidebar.button("Logout"):
            st.session_state["authenticated"] = False
            st.rerun()
        main()
