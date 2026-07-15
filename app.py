import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
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
    """Add sentiment_score and sentiment_label columns.
    Sentiment is computed ONLY for rows where:
      - task_status == 'completed'
      - expert_name is NOT 'Self' (case-insensitive)
    All other rows get None for sentiment columns.
    """
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

    # Default all rows to None
    df["sentiment_score"] = None
    df["sentiment_label"] = None

    # Build mask: completed AND expert is not Self
    mask = pd.Series(True, index=df.index)

    if "task_status" in df.columns:
        mask = mask & (df["task_status"].astype(str).str.strip().str.lower() == "completed")

    if "expert_name" in df.columns:
        mask = mask & (df["expert_name"].astype(str).str.strip().str.lower() != "self")

    # Compute sentiment only for qualifying rows
    if mask.any():
        df.loc[mask, "sentiment_score"] = df.loc[mask, found_col].apply(compute_sentiment_score)
        df.loc[mask, "sentiment_label"] = df.loc[mask, "sentiment_score"].apply(
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
    """Add start_hour, start_hour_label, and _parsed_start columns."""
    if "start_time" not in df.columns:
        return df

    df = df.copy()

    st_raw = df["start_time"].astype(str).str.strip()
    noon_s = st_raw.str.lower() == "noon"
    st_raw = st_raw.where(~noon_s, "12:00 PM")
    start_parsed = pd.to_datetime(st_raw, errors="coerce", format="%I:%M %p")
    mask_nat = start_parsed.isna() & st_raw.notna() & (st_raw != "") & (st_raw.str.lower() != "nan")
    if mask_nat.any():
        fallback1 = pd.to_datetime(st_raw[mask_nat], errors="coerce", format="%H:%M")
        start_parsed.loc[mask_nat] = fallback1
    mask_nat2 = start_parsed.isna() & st_raw.notna() & (st_raw != "") & (st_raw.str.lower() != "nan")
    if mask_nat2.any():
        fallback2 = pd.to_datetime(st_raw[mask_nat2], errors="coerce", format="mixed")
        start_parsed.loc[mask_nat2] = fallback2

    df["start_hour"] = start_parsed.dt.hour
    df["start_hour_label"] = start_parsed.dt.strftime("%I %p")
    df["_parsed_start"] = start_parsed

    return df


def render_start_time_insights(df, title_suffix=""):
    """Render the Start Time Insights section."""
    if "start_hour" not in df.columns:
        st.info("No start_time data available for time-of-day analysis.")
        return

    valid = df.dropna(subset=["start_hour"]).copy()
    if valid.empty:
        st.info("No valid start_time entries found" + title_suffix + ".")
        return

    st.subheader("Start Time Insights" + title_suffix)

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

    slots = {
        "Early Morning (7-9 AM)": (6, 9),
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
    """Render a monthly × hour-of-day heatmap and monthly mean start time line."""
    if "start_hour" not in df.columns or "date" not in df.columns:
        return

    valid = df.dropna(subset=["start_hour"]).copy()
    if valid.empty:
        return

    valid["month"] = valid["date"].dt.to_period("M").astype(str)

    st.subheader("Monthly Start Time Trends" + title_suffix)

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
#  SCHEDULING CLASH DETECTION
# ═══════════════════════════════════════════════════════════════════

def _build_clash_groups(minutes_list):
    """Given a sorted list of (index, minutes) tuples, find connected
    components where any two nodes within 30 min are connected."""
    if len(minutes_list) < 2:
        return []
    n = len(minutes_list)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(minutes_list[j][1] - minutes_list[i][1]) <= 30:
                adj[i].add(j)
                adj[j].add(i)
    visited = set()
    groups = []
    for i in range(n):
        if i in visited or not adj[i]:
            continue
        component = []
        queue = [i]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(minutes_list[node])
            for nb in adj[node]:
                if nb not in visited:
                    queue.append(nb)
        if len(component) >= 2:
            groups.append(component)
    return groups


def detect_expert_clashes(df):
    """Detect scheduling clash GROUPS for experts.

    Returns two DataFrames:
      1) clash_groups_df
      2) clash_pairs_df
    """
    empty_groups = pd.DataFrame()
    empty_pairs = pd.DataFrame()

    if "_parsed_start" not in df.columns or "expert_name" not in df.columns or "date" not in df.columns:
        return empty_groups, empty_pairs

    valid = df.dropna(subset=["_parsed_start", "expert_name", "date"]).copy()
    if valid.empty:
        return empty_groups, empty_pairs

    valid["_day"] = valid["date"].dt.date
    valid["_start_minutes"] = valid["_parsed_start"].dt.hour * 60 + valid["_parsed_start"].dt.minute

    group_rows = []
    pair_rows = []

    for (expert, day), grp in valid.groupby(["expert_name", "_day"]):
        if len(grp) < 2:
            continue
        sorted_grp = grp.sort_values("_start_minutes")
        minutes_indexed = list(enumerate(zip(
            sorted_grp["_start_minutes"].values,
            sorted_grp["_parsed_start"].dt.strftime("%I:%M %p").values
        )))
        minutes_list = [(i, int(m)) for i, (m, _) in minutes_indexed]
        time_labels = {i: lbl for i, (_, lbl) in minutes_indexed}

        groups = _build_clash_groups(minutes_list)

        for group in groups:
            size = len(group)
            times = [time_labels[idx] for idx, _ in group]
            mins_vals = [m for _, m in group]
            mean_min = sum(mins_vals) / len(mins_vals)
            mean_h = int(mean_min // 60)
            mean_m = int(mean_min % 60)
            mean_label = datetime(2000, 1, 1, mean_h, mean_m).strftime("%I:%M %p")

            group_rows.append({
                "expert_name": expert,
                "date": day,
                "group_size": size,
                "interviews_str": ", ".join(times),
                "mean_start_minutes": round(mean_min, 1),
                "mean_start_label": mean_label,
                "start_times_list": times,
            })

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    diff = abs(group[j][1] - group[i][1])
                    if diff <= 30:
                        pair_rows.append({
                            "expert_name": expert,
                            "date": day,
                            "start_time_1": time_labels[group[i][0]],
                            "start_time_2": time_labels[group[j][0]],
                            "time_diff_min": diff,
                        })

    if not group_rows:
        return empty_groups, empty_pairs

    clash_groups_df = pd.DataFrame(group_rows)
    clash_groups_df["date"] = pd.to_datetime(clash_groups_df["date"])
    clash_groups_df["month"] = clash_groups_df["date"].dt.to_period("M").astype(str)

    clash_pairs_df = pd.DataFrame(pair_rows) if pair_rows else empty_pairs
    if not clash_pairs_df.empty:
        clash_pairs_df["date"] = pd.to_datetime(clash_pairs_df["date"])
        clash_pairs_df["month"] = clash_pairs_df["date"].dt.to_period("M").astype(str)

    return clash_groups_df, clash_pairs_df


def _mean_start_label_from_minutes(minutes_val):
    """Convert float minutes-since-midnight to HH:MM AM/PM label."""
    h = int(minutes_val // 60) % 24
    m = int(minutes_val % 60)
    return datetime(2000, 1, 1, h, m).strftime("%I:%M %p")


def render_clash_summary(df, title_suffix=""):
    """Render the full Scheduling Clash Summary section."""
    clash_groups, clash_pairs = detect_expert_clashes(df)

    st.subheader("Scheduling Clash Detection" + title_suffix)
    st.caption(
        "A clash occurs when the same expert has 2+ interviews within a "
        "30-minute window on the same day. A '3-interview clash' means 3 "
        "interviews form a connected overlap group."
    )

    if clash_groups.empty:
        st.success("No scheduling clashes detected" + title_suffix + ".")
        return

    # ── KPI row ──────────────────────────────────────────────────
    total_groups = len(clash_groups)
    total_interviews_in_clashes = int(clash_groups["group_size"].sum())
    experts_with_clashes = clash_groups["expert_name"].nunique()
    days_with_clashes = clash_groups["date"].dt.date.nunique()
    overall_mean_min = clash_groups["mean_start_minutes"].mean()
    overall_mean_label = _mean_start_label_from_minutes(overall_mean_min)

    k = st.columns(5)
    k[0].metric("Clash Groups", total_groups)
    k[1].metric("Interviews in Clashes", total_interviews_in_clashes)
    k[2].metric("Experts with Clashes", experts_with_clashes)
    k[3].metric("Days with Clashes", days_with_clashes)
    k[4].metric("Mean Clash Start Time", overall_mean_label)

    # ── Clash Size Distribution ──────────────────────────────────
    st.markdown("---")
    st.subheader("Clash Size Distribution" + title_suffix)
    st.caption("How many interviews overlap in each clash group")

    size_counts = clash_groups["group_size"].value_counts().sort_index()
    size_labels = [str(s) + "-Interview Clash" for s in size_counts.index]
    size_colors = ["#f39c12" if s == 2 else "#e74c3c" if s == 3 else "#8e44ad"
                   for s in size_counts.index]

    col_size1, col_size2 = st.columns(2)
    with col_size1:
        fig_size = go.Figure(go.Bar(
            x=size_labels, y=size_counts.values,
            marker_color=size_colors,
            text=size_counts.values, textposition="outside",
        ))
        fig_size.update_layout(
            title="Overall Clash Size Distribution",
            height=400,
            xaxis_title="Clash Type",
            yaxis_title="Number of Clash Groups",
        )
        st.plotly_chart(fig_size, use_container_width=True)

    with col_size2:
        fig_pie = go.Figure(go.Pie(
            labels=size_labels, values=size_counts.values.tolist(),
            hole=0.45,
            marker=dict(colors=size_colors),
            textinfo="label+value+percent",
        ))
        fig_pie.update_layout(title="Clash Size Split", height=400, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Expert-wise Clash Size Breakdown ─────────────────────────
    st.markdown("---")
    st.subheader("Expert-wise Clash Breakdown" + title_suffix)

    expert_size = clash_groups.groupby(["expert_name", "group_size"]).size().reset_index(name="count")
    expert_size["size_label"] = expert_size["group_size"].apply(lambda s: str(s) + "-Interview")

    expert_totals = expert_size.groupby("expert_name")["count"].sum().sort_values(ascending=False)
    top_experts = expert_totals.head(20).index.tolist()
    expert_size_top = expert_size[expert_size["expert_name"].isin(top_experts)]

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        pivot_es = expert_size_top.pivot_table(
            index="expert_name", columns="size_label", values="count", fill_value=0
        )
        pivot_es["_total"] = pivot_es.sum(axis=1)
        pivot_es = pivot_es.sort_values("_total", ascending=True).drop(columns="_total")

        fig_es = go.Figure()
        color_map = {"2-Interview": "#f39c12", "3-Interview": "#e74c3c",
                     "4-Interview": "#8e44ad", "5-Interview": "#2c3e50"}
        for col_name in sorted(pivot_es.columns):
            clr = color_map.get(col_name, "#95a5a6")
            fig_es.add_trace(go.Bar(
                y=pivot_es.index, x=pivot_es[col_name],
                name=col_name, orientation="h",
                marker_color=clr,
                text=pivot_es[col_name], textposition="inside",
            ))
        fig_es.update_layout(
            barmode="stack",
            title="Clash Groups by Expert & Size",
            height=max(420, len(top_experts) * 35),
            xaxis_title="Clash Groups",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_es, use_container_width=True)

    with col_e2:
        expert_agg = clash_groups.groupby("expert_name").agg(
            clash_groups_count=("group_size", "size"),
            total_interviews=("group_size", "sum"),
            clash_days=("date", lambda x: x.dt.date.nunique()),
            mean_start=("mean_start_minutes", "mean"),
        ).reset_index().sort_values("clash_groups_count", ascending=False)
        expert_agg["mean_start_label"] = expert_agg["mean_start"].apply(_mean_start_label_from_minutes)

        fig_e2 = go.Figure()
        fig_e2.add_trace(go.Bar(
            y=expert_agg["expert_name"].head(15),
            x=expert_agg["clash_groups_count"].head(15),
            orientation="h",
            marker_color="#e74c3c",
            text=expert_agg["clash_groups_count"].head(15),
            textposition="outside",
            name="Clash Groups",
        ))
        fig_e2.update_layout(
            title="Top 15 Experts by Clash Groups",
            height=max(420, 15 * 35),
            yaxis=dict(autorange="reversed"),
            xaxis_title="Clash Groups",
        )
        st.plotly_chart(fig_e2, use_container_width=True)

    # ── Monthly Clash Trend with Size Breakdown ──────────────────
    st.markdown("---")
    st.subheader("Monthly Clash Trends" + title_suffix)

    monthly_size = clash_groups.groupby(["month", "group_size"]).size().reset_index(name="count")
    monthly_size["size_label"] = monthly_size["group_size"].apply(lambda s: str(s) + "-Interview")

    pivot_ms = monthly_size.pivot_table(
        index="month", columns="size_label", values="count", fill_value=0
    ).reset_index()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_ms = go.Figure()
        for col_name in sorted([c for c in pivot_ms.columns if c != "month"]):
            clr = color_map.get(col_name, "#95a5a6")
            fig_ms.add_trace(go.Bar(
                x=pivot_ms["month"], y=pivot_ms[col_name],
                name=col_name, marker_color=clr,
                text=pivot_ms[col_name], textposition="inside",
            ))
        fig_ms.update_layout(
            barmode="stack",
            title="Monthly Clash Groups by Size",
            height=420,
            yaxis_title="Clash Groups",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_ms, use_container_width=True)

    with col_m2:
        monthly_mean = clash_groups.groupby("month").agg(
            mean_start=("mean_start_minutes", "mean"),
        ).reset_index()
        monthly_mean["mean_start_label"] = monthly_mean["mean_start"].apply(
            _mean_start_label_from_minutes
        )
        monthly_mean["mean_start_hour"] = (monthly_mean["mean_start"] / 60).round(2)

        fig_mm = go.Figure()
        fig_mm.add_trace(go.Scatter(
            x=monthly_mean["month"],
            y=monthly_mean["mean_start_hour"],
            mode="lines+markers+text",
            text=monthly_mean["mean_start_label"],
            textposition="top center",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=10),
        ))
        fig_mm.update_layout(
            title="Mean Clash Start Time by Month",
            height=420,
            xaxis_title="Month",
            yaxis_title="Hour of Day (24h)",
            yaxis=dict(range=[
                max(0, monthly_mean["mean_start_hour"].min() - 2),
                min(24, monthly_mean["mean_start_hour"].max() + 2),
            ]),
        )
        st.plotly_chart(fig_mm, use_container_width=True)

    # ── Time-of-Day Distribution ─────────────────────────────────
    st.markdown("---")
    st.subheader("Clash Time-of-Day Distribution" + title_suffix)

    clash_hours = (clash_groups["mean_start_minutes"] // 60).astype(int)
    hour_counts = clash_hours.value_counts().sort_index()
    all_hours = list(range(0, 24))
    hour_labels = [datetime(2000, 1, 1, h).strftime("%I %p").lstrip("0") for h in all_hours]
    counts = [int(hour_counts.get(h, 0)) for h in all_hours]
    peak_h = int(hour_counts.idxmax()) if not hour_counts.empty else 0
    bar_colors = ["#e74c3c" if h == peak_h else "#f39c12" for h in all_hours]

    fig_hour = go.Figure(go.Bar(
        x=hour_labels, y=counts,
        marker_color=bar_colors,
        text=counts, textposition="outside",
    ))
    fig_hour.update_layout(
        title="Clash Groups by Hour of Day" + title_suffix,
        height=420,
        xaxis_title="Hour of Day",
        yaxis_title="Clash Groups",
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # ── Expert-wise summary table ────────────────────────────────
    with st.expander("Expert Clash Summary Table" + title_suffix):
        expert_agg_display = expert_agg.copy()
        expert_agg_display = expert_agg_display[["expert_name", "clash_groups_count",
                                                  "total_interviews", "clash_days",
                                                  "mean_start_label"]]
        expert_agg_display.columns = ["Expert", "Clash Groups", "Interviews in Clashes",
                                      "Days with Clashes", "Mean Clash Start Time"]
        st.dataframe(expert_agg_display, use_container_width=True, hide_index=True)

    # ── Detailed Clash Groups table ──────────────────────────────
    with st.expander("Detailed Clash Groups" + title_suffix):
        detail = clash_groups[["expert_name", "date", "group_size",
                               "interviews_str", "mean_start_label", "month"]].copy()
        detail.columns = ["Expert", "Date", "Group Size", "Overlapping Times",
                          "Mean Start", "Month"]
        detail["Date"] = detail["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(detail, use_container_width=True, hide_index=True)

    if not clash_pairs.empty:
        with st.expander("Detailed Clash Pairs" + title_suffix):
            pairs_disp = clash_pairs[["expert_name", "date", "start_time_1",
                                      "start_time_2", "time_diff_min", "month"]].copy()
            pairs_disp.columns = ["Expert", "Date", "Time 1", "Time 2",
                                  "Diff (min)", "Month"]
            pairs_disp["Date"] = pairs_disp["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(pairs_disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  TODAY'S CLASH SUMMARY (compact view for Today's Snapshot)
# ═══════════════════════════════════════════════════════════════════

def render_today_clash_summary(df):
    """Compact clash summary for today's snapshot."""
    clash_groups, clash_pairs = detect_expert_clashes(df)

    if clash_groups.empty:
        st.success("No scheduling clashes detected today.")
        return

    total_groups = len(clash_groups)
    total_interviews = int(clash_groups["group_size"].sum())
    experts_with = clash_groups["expert_name"].nunique()
    overall_mean_min = clash_groups["mean_start_minutes"].mean()
    overall_mean_label = _mean_start_label_from_minutes(overall_mean_min)

    k = st.columns(4)
    k[0].metric("⚠️ Clash Groups", total_groups)
    k[1].metric("Interviews in Clashes", total_interviews)
    k[2].metric("Experts with Clashes", experts_with)
    k[3].metric("Mean Clash Start Time", overall_mean_label)

    with st.expander("Clash Details"):
        display = clash_groups[["expert_name", "group_size", "interviews_str",
                                "mean_start_label"]].copy()
        display.columns = ["Expert", "Group Size", "Overlapping Times", "Mean Start"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        if not clash_pairs.empty:
            st.caption("Pairwise Clashes")
            pairs_display = clash_pairs[["expert_name", "start_time_1",
                                         "start_time_2", "time_diff_min"]].copy()
            pairs_display.columns = ["Expert", "Time 1", "Time 2", "Diff (min)"]
            st.dataframe(pairs_display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  BLOCKAGE DETECTION
#  A blockage occurs in a 30-minute bracket on a given day when:
#    1) ALL active experts have at least one interview in that bracket
#    2) At least one expert has a clash (2+ interviews) in that bracket
#
#  Active experts = all unique expert names from the last 5 calendar
#  days of data PLUS any new experts appearing on the target day.
# ═══════════════════════════════════════════════════════════════════

def _get_active_experts_for_day(df, target_day, lookback_working_days=7):
    """Return the set of active expert names based on the past N working days
    of data relative to target_day, plus any new experts on target_day itself.
    
    Working days = days that actually have interview data (not calendar days).
    """
    if df.empty or "expert_name" not in df.columns or "date" not in df.columns:
        return set()
    
    # Get all unique dates with data, before the target day
    all_dates = sorted(df["date"].dt.date.unique())
    past_dates = [d for d in all_dates if d < target_day]
    
    # Take the last N working days (days with actual data)
    recent_dates = set(past_dates[-lookback_working_days:]) if past_dates else set()
    
    # Experts from recent working days
    if recent_dates:
        mask = df["date"].dt.date.isin(recent_dates)
        recent_experts = set(df.loc[mask, "expert_name"].dropna().unique())
    else:
        recent_experts = set()
    
    # Experts appearing on the target day itself
    target_mask = df["date"].dt.date == target_day
    target_experts = set(df.loc[target_mask, "expert_name"].dropna().unique())
    
    # Combine both
    return recent_experts | target_experts


def detect_blockages(df, active_expert_count=None, all_experts_df=None):
    """Detect blockage events across all days in df.

    New logic: A blockage occurs in a 1-hour window on a given day when
    the number of interviews in that window >= number of active experts
    for that specific day.
    
    Active experts per day = unique experts from past 7 working days + 
    any new expert on that day.

    Returns a DataFrame with one row per blockage bracket, columns:
      date, day_name, bracket_start_min, bracket_end_min, bracket_label,
      active_experts, interviews_in_bracket, overflow,
      involved_experts, mean_start_minutes, mean_start_label, month
    """
    empty = pd.DataFrame()
    if "_parsed_start" not in df.columns or "expert_name" not in df.columns or "date" not in df.columns:
        return empty

    valid = df.dropna(subset=["_parsed_start", "expert_name", "date"]).copy()
    if valid.empty:
        return empty

    valid["_day"] = valid["date"].dt.date
    valid["_start_minutes"] = valid["_parsed_start"].dt.hour * 60 + valid["_parsed_start"].dt.minute

    blockage_rows = []

    for day_val, day_grp in valid.groupby("_day"):
        # Calculate active experts specifically for this day
        if active_expert_count is not None:
            day_active_count = active_expert_count
        else:
            lookback_data = all_experts_df if all_experts_df is not None else valid
            day_active_experts = _get_active_experts_for_day(lookback_data, day_val, lookback_working_days=7)
            day_active_count = len(day_active_experts)

        if day_active_count == 0:
            continue

        min_start = int(day_grp["_start_minutes"].min())
        max_start = int(day_grp["_start_minutes"].max())

        # Slide a 1-hour (60 min) window in 15-min steps
        for bracket_start in range(max(0, min_start - 60), min(max_start + 15, 24 * 60), 15):
            bracket_end = bracket_start + 60

            in_bracket = day_grp[
                (day_grp["_start_minutes"] >= bracket_start) &
                (day_grp["_start_minutes"] < bracket_end)
            ]

            if in_bracket.empty:
                continue

            interview_count = len(in_bracket)

            # BLOCKAGE: interviews in this hour >= active experts for this day
            if interview_count > day_active_count:
                experts_in_bracket = sorted(in_bracket["expert_name"].unique())
                mean_min = in_bracket["_start_minutes"].mean()
                mean_h = int(mean_min // 60) % 24
                mean_m = int(mean_min % 60)
                mean_label = datetime(2000, 1, 1, mean_h, mean_m).strftime("%I:%M %p")

                bracket_h = int(bracket_start // 60) % 24
                bracket_m = int(bracket_start % 60)
                bracket_label = datetime(2000, 1, 1, bracket_h, bracket_m).strftime("%I:%M %p")

                bracket_end_h = int(bracket_end // 60) % 24
                bracket_end_m = int(bracket_end % 60)
                bracket_end_label = datetime(2000, 1, 1, bracket_end_h, bracket_end_m).strftime("%I:%M %p")

                overflow = interview_count - day_active_count

                blockage_rows.append({
                    "date": pd.Timestamp(day_val),
                    "day_name": pd.Timestamp(day_val).day_name(),
                    "bracket_start_min": bracket_start,
                    "bracket_end_min": bracket_end,
                    "bracket_label": bracket_label + " - " + bracket_end_label,
                    "active_experts": day_active_count,
                    "interviews_in_bracket": interview_count,
                    "overflow": overflow,
                    "involved_experts": ", ".join(experts_in_bracket),
                    "mean_start_minutes": round(mean_min, 1),
                    "mean_start_label": mean_label,
                })

    if not blockage_rows:
        return empty

    result = pd.DataFrame(blockage_rows)
    result["month"] = result["date"].dt.to_period("M").astype(str)

    # Deduplicate overlapping windows per day
    result = result.sort_values(["date", "interviews_in_bracket"], ascending=[True, False])
    result = result.drop_duplicates(subset=["date", "bracket_start_min"], keep="first")

    return result

    result = pd.DataFrame(blockage_rows)
    result["month"] = result["date"].dt.to_period("M").astype(str)
    return result


# ═══════════════════════════════════════════════════════════════════
#  TODAY'S BLOCKAGE INDICATOR
# ═══════════════════════════════════════════════════════════════════

def render_today_blockage(df, all_experts_df=None):
    """Compact blockage indicator for Today's Snapshot."""
    blockages = detect_blockages(df, all_experts_df=all_experts_df)

    if blockages.empty:
        st.success("No blockage detected today. Interview load is within expert capacity in every hour.")
        return

    today_val = date.today()
    today_blockages = blockages[blockages["date"].dt.date == today_val]

    if today_blockages.empty:
        st.success("No blockage detected today.")
        return

    total_brackets = len(today_blockages)
    active_count = int(today_blockages["active_experts"].iloc[0])
    max_overflow = int(today_blockages["overflow"].max())
    max_interviews = int(today_blockages["interviews_in_bracket"].max())
    mean_min = today_blockages["mean_start_minutes"].mean()
    mean_label = _mean_start_label_from_minutes(mean_min)

    st.warning(f"🚨 **{total_brackets} Blockage Hour(s) Detected Today!** (Interviews > {active_count} active experts)")

    k = st.columns(5)
    k[0].metric("Blockage Hours", total_brackets)
    k[1].metric("Active Experts (7-day)", active_count)
    k[2].metric("Max Overflow", f"+{max_overflow}")
    k[3].metric("Peak Blockage Time", mean_label)
    k[4].metric("Peak Interviews in 1 Hour", max_interviews)

    with st.expander("Blockage Details"):
        display = today_blockages[["bracket_label", "active_experts",
                                    "interviews_in_bracket", "overflow",
                                    "involved_experts", "mean_start_label"]].copy()
        display.columns = ["Hour Window", "Active Experts", "Interviews",
                           "Overflow", "Experts Involved", "Mean Start"]
        st.dataframe(display, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
#  BLOCKAGE ANALYTICS (full view for Monthly / Deep-Dive)
# ═══════════════════════════════════════════════════════════════════

def render_blockage_summary(df, title_suffix="", all_experts_df=None):
    """Full blockage analytics section."""
    blockages = detect_blockages(df, all_experts_df=all_experts_df)

    st.subheader("Blockage Analysis" + title_suffix)
    st.caption(
        "A blockage occurs in any 1-hour window when the number of interviews "
        "exceeds the number of active experts for that day. "
        "Active experts = unique experts (including now-inactive) from the past 7 working days + any new expert on that day. "
        "This means all experts are occupied and no capacity remains."
    )

    if blockages.empty:
        st.success("No blockage events detected" + title_suffix + ".")
        return

    # ── KPIs ─────────────────────────────────────────────────────
    total_events = len(blockages)
    days_with = blockages["date"].dt.date.nunique()
    months_with = blockages["month"].nunique()
    max_overflow = int(blockages["overflow"].max())
    avg_overflow = round(blockages["overflow"].mean(), 1)
    mean_min = blockages["mean_start_minutes"].mean()
    mean_label = _mean_start_label_from_minutes(mean_min)
    avg_active = round(blockages["active_experts"].mean(), 1)

    k = st.columns(7)
    k[0].metric("Total Blockage Events", total_events)
    k[1].metric("Days with Blockage", days_with)
    k[2].metric("Months with Blockage", months_with)
    k[3].metric("Max Overflow", f"+{max_overflow}")
    k[4].metric("Avg Overflow", f"+{avg_overflow}")
    k[5].metric("Avg Active Experts", f"{avg_active:.1f}")
    k[6].metric("Mean Blockage Time", mean_label)

    # ── Monthly Blockage Trend ───────────────────────────────────
    st.markdown("---")
    st.subheader("Monthly Blockage Trends" + title_suffix)

    monthly_counts = blockages.groupby("month").agg(
        blockage_events=("overflow", "size"),
        avg_overflow=("overflow", "mean"),
        max_overflow=("overflow", "max"),
        avg_interviews=("interviews_in_bracket", "mean"),
        avg_active=("active_experts", "mean"),
    ).reset_index()
    monthly_counts["avg_overflow"] = monthly_counts["avg_overflow"].round(1)
    monthly_counts["avg_interviews"] = monthly_counts["avg_interviews"].round(1)
    monthly_counts["avg_active"] = monthly_counts["avg_active"].round(1)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly_counts["month"],
            y=monthly_counts["blockage_events"],
            marker_color="#e74c3c",
            text=monthly_counts["blockage_events"],
            textposition="outside",
            name="Blockage Events",
        ))
        fig_monthly.update_layout(
            title="Blockage Events by Month",
            height=420,
            xaxis_title="Month",
            yaxis_title="Blockage Events",
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_m2:
        fig_ov = go.Figure()
        fig_ov.add_trace(go.Scatter(
            x=monthly_counts["month"],
            y=monthly_counts["avg_overflow"],
            mode="lines+markers+text",
            text=monthly_counts["avg_overflow"].apply(lambda v: f"+{v:.1f}"),
            textposition="top center",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=10),
            name="Avg Overflow",
        ))
        fig_ov.add_trace(go.Scatter(
            x=monthly_counts["month"],
            y=monthly_counts["avg_active"],
            mode="lines+markers",
            line=dict(color="#3498db", width=2, dash="dot"),
            marker=dict(size=8),
            name="Avg Active Experts",
        ))
        fig_ov.update_layout(
            title="Monthly Overflow vs Active Experts",
            height=420,
            xaxis_title="Month",
            yaxis_title="Count",
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_ov, use_container_width=True)

    # ── Expert-wise Blockage Involvement ─────────────────────────
    st.markdown("---")
    st.subheader("Expert-wise Blockage Involvement" + title_suffix)

    all_involved = blockages["involved_experts"].str.split(", ").explode()
    expert_involvement = all_involved.value_counts().head(20)

    if not expert_involvement.empty:
        fig_exp = go.Figure(go.Bar(
            y=expert_involvement.index,
            x=expert_involvement.values,
            orientation="h",
            marker_color="#e74c3c",
            text=expert_involvement.values,
            textposition="outside",
        ))
        fig_exp.update_layout(
            title="Top 20 Experts in Blockage Hours",
            height=max(420, len(expert_involvement) * 35),
            yaxis=dict(autorange="reversed"),
            xaxis_title="Blockage Hours Involved",
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    # ── Time-of-Day Distribution ─────────────────────────────────
    st.markdown("---")
    st.subheader("Blockage Time-of-Day Distribution" + title_suffix)

    blockage_hours = (blockages["bracket_start_min"] // 60).astype(int)
    hour_counts = blockage_hours.value_counts().sort_index()
    all_hours = list(range(0, 24))
    hour_labels = [datetime(2000, 1, 1, h).strftime("%I %p").lstrip("0") for h in all_hours]
    counts = [int(hour_counts.get(h, 0)) for h in all_hours]
    peak_h = int(hour_counts.idxmax()) if not hour_counts.empty else 0
    bar_colors = ["#e74c3c" if h == peak_h else "#f39c12" for h in all_hours]

    fig_tod = go.Figure(go.Bar(
        x=hour_labels, y=counts,
        marker_color=bar_colors,
        text=counts, textposition="outside",
    ))
    fig_tod.update_layout(
        title="Blockage by Hour of Day" + title_suffix,
        height=420,
        xaxis_title="Hour of Day",
        yaxis_title="Blockage Events",
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_tod, use_container_width=True)

    # ── Overflow Heatmap: Day x Hour ─────────────────────────────
    st.markdown("---")
    st.subheader("Blockage Intensity Heatmap" + title_suffix)
    st.caption("Overflow count (interviews above expert capacity) by date and hour")

    blockages_hm = blockages.copy()
    blockages_hm["hour"] = (blockages_hm["bracket_start_min"] // 60).astype(int)
    blockages_hm["date_str"] = blockages_hm["date"].dt.strftime("%Y-%m-%d")

    pivot_hm = blockages_hm.groupby(["date_str", "hour"])["overflow"].max().reset_index()
    pivot_wide = pivot_hm.pivot(index="hour", columns="date_str", values="overflow").fillna(0).astype(int)
    pivot_wide = pivot_wide.reindex(range(24), fill_value=0)
    pivot_wide.index = [datetime(2000, 1, 1, h).strftime("%I %p").lstrip("0") for h in range(24)]

    pivot_wide = pivot_wide.loc[(pivot_wide != 0).any(axis=1)]

    if not pivot_wide.empty:
        fig_hm = px.imshow(
            pivot_wide, text_auto=True, aspect="auto",
            color_continuous_scale="Reds",
            title="Blockage Overflow: Date x Hour" + title_suffix,
            labels=dict(x="Date", y="Hour", color="Overflow"),
        )
        fig_hm.update_layout(height=max(400, len(pivot_wide) * 35))
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Detailed Blockage Table ──────────────────────────────────
    with st.expander("Detailed Blockage Events" + title_suffix):
        display = blockages[["date", "day_name", "bracket_label",
                              "active_experts", "interviews_in_bracket",
                              "overflow", "involved_experts",
                              "mean_start_label", "month"]].copy()
        display.columns = ["Date", "Day", "Hour Window", "Active Experts (7-day)",
                           "Interviews", "Overflow", "Experts Involved",
                           "Mean Start", "Month"]
        display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  WORD CLOUD & FEEDBACK TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════

def render_wordcloud_section(texts, section_title="Word Cloud"):
    if not texts:
        st.info("No text available for word cloud.")
        return

    st.subheader(section_title)

    stop_words = set(stopwords.words("english")) | DOMAIN_STOPWORDS
    lemmatizer = WordNetLemmatizer()

    all_tokens = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        for word, tag in tagged:
            if word in stop_words or len(word) < 3:
                continue
            if tag.startswith("NN"):
                pos_wn = "n"
            elif tag.startswith("VB"):
                pos_wn = "v"
            elif tag.startswith("JJ"):
                pos_wn = "a"
            elif tag.startswith("RB"):
                pos_wn = "r"
            else:
                pos_wn = "n"
            lemma = lemmatizer.lemmatize(word, pos=pos_wn)
            if lemma not in stop_words and len(lemma) >= 3:
                all_tokens.append(lemma)

    if not all_tokens:
        st.info("No meaningful words extracted from the feedback.")
        return

    freq = Counter(all_tokens)
    top_n = 30
    top_words = freq.most_common(top_n)

    fig_top = go.Figure(go.Bar(
        x=[c for _, c in top_words],
        y=[w for w, _ in top_words],
        orientation="h",
        marker_color="#3498db",
        text=[c for _, c in top_words],
        textposition="outside",
    ))
    fig_top.update_layout(
        title=f"Top {top_n} Words",
        height=max(400, top_n * 28),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Frequency",
    )

    wc = WordCloud(width=1200, height=600, background_color="white",
                   max_words=200, colormap="viridis",
                   prefer_horizontal=0.7)
    wc.generate_from_frequencies(freq)

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
    """Extract feedback texts for word cloud / NLP analysis.
    Only includes rows where:
      - task_status == 'completed'
      - expert_name is NOT 'Self' (case-insensitive)
    """
    candidates = [col, "feedback", "Feedback", "feedback_text",
                  "case_feedback", "expert_feedback", "comments", "remark", "remarks"]
    found_col = None
    for c in candidates:
        if c in df.columns:
            found_col = c
            break
    if found_col is None:
        return []

    # Apply the same filter: completed + not Self
    filtered = df.copy()

    if "task_status" in filtered.columns:
        filtered = filtered[filtered["task_status"].astype(str).str.strip().str.lower() == "completed"]

    if "expert_name" in filtered.columns:
        filtered = filtered[filtered["expert_name"].astype(str).str.strip().str.lower() != "self"]

    if filtered.empty:
        return []

    series = filtered[found_col].dropna().astype(str).str.strip()
    series = series[series != ""]
    series = series[series.str.lower() != "nan"]
    series = series[series.str.lower() != "none"]
    return series.tolist()

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


# ── CHANGED: round_charts now uses ONLY completed for pie ────────
def round_charts(df, title_suffix=""):
    if "round_name" not in df.columns or df.empty:
        return

    # ── Pie chart: ONLY completed interviews ─────────────────────
    completed_df = df[df["task_status"] == "completed"] if "task_status" in df.columns else df
    rc = completed_df["round_name"].value_counts()

    c1, c2 = st.columns(2)
    with c1:
        if rc.empty:
            st.info("No completed interviews for round distribution" + title_suffix)
        else:
            fig = go.Figure(go.Pie(labels=rc.index, values=rc.values, hole=.4,
                                   textinfo="label+value+percent"))
            fig.update_layout(title="Round Distribution (Completed Only)" + title_suffix, height=420)
            st.plotly_chart(fig, use_container_width=True)

    # ── Stacked bar: all statuses for reference ──────────────────
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
#  SCHEDULE VIEW — Gantt-style daily timeline per expert
# ═══════════════════════════════════════════════════════════════════

def _parse_time_to_minutes(val):
    """Convert a time string to minutes since midnight. 'noon' → 720."""
    if not isinstance(val, str) or not val.strip():
        return None
    v = val.strip()
    if v.lower() == "noon":
        return 720
    for fmt in ("%I:%M %p", "%H:%M", "%I:%M%p", "%I %p"):
        try:
            t = datetime.strptime(v, fmt)
            return t.hour * 60 + t.minute
        except ValueError:
            continue
    try:
        t = pd.to_datetime(v, errors="coerce")
        if pd.notna(t):
            return t.hour * 60 + t.minute
    except Exception:
        pass
    return None


def _minutes_to_label(m):
    """Convert minutes since midnight to 'HH:MM AM/PM' label."""
    h = int(m // 60) % 24
    mi = int(m % 60)
    return datetime(2000, 1, 1, h, mi).strftime("%I:%M %p")


def build_schedule_data(all_data, selected_date):
    """Build a DataFrame of interview slots for the Gantt chart.

    Returns a DataFrame with columns:
      expert_name, candidate_name, company_name, round_name, support_name,
      task_status, start_min, end_min, start_label, end_label, duration,
      has_clash (bool)
    """
    if all_data.empty or "date" not in all_data.columns:
        return pd.DataFrame()

    day_df = all_data[
        (all_data["date"].dt.date == selected_date) &
        (all_data["support_name"].str.lower() == "interview support")
    ].copy()

    if day_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in day_df.iterrows():
        expert = r.get("expert_name", "Unknown")
        candidate = r.get("candidate_name", "")
        company = r.get("company_name", "")
        round_name = r.get("round_name", "")
        support = r.get("support_name", "")
        status = r.get("task_status", "pending")

        # Parse start time
        start_raw = r.get("start_time", None)
        start_min = _parse_time_to_minutes(str(start_raw)) if pd.notna(start_raw) else None

        # Parse end time
        end_raw = r.get("end_time", None)
        end_min = _parse_time_to_minutes(str(end_raw)) if pd.notna(end_raw) else None

        # If no start time at all, skip this row
        if start_min is None:
            continue

        # Default duration: 30 minutes if no end time
        if end_min is None or end_min <= start_min:
            end_min = start_min + 30

        duration = end_min - start_min

        rows.append({
            "expert_name": expert,
            "candidate_name": candidate,
            "company_name": company,
            "round_name": round_name,
            "support_name": support,
            "task_status": status,
            "start_min": start_min,
            "end_min": end_min,
            "start_label": _minutes_to_label(start_min),
            "end_label": _minutes_to_label(end_min),
            "duration": duration,
        })

    if not rows:
        return pd.DataFrame()

    sched = pd.DataFrame(rows)

    # Detect clashes per expert (overlapping intervals)
    sched["has_clash"] = False
    for expert, grp in sched.groupby("expert_name"):
        if len(grp) < 2:
            continue
        idxs = grp.index.tolist()
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                s1, e1 = sched.loc[idxs[i], "start_min"], sched.loc[idxs[i], "end_min"]
                s2, e2 = sched.loc[idxs[j], "start_min"], sched.loc[idxs[j], "end_min"]
                if s1 < e2 and s2 < e1:  # overlap
                    sched.loc[idxs[i], "has_clash"] = True
                    sched.loc[idxs[j], "has_clash"] = True

    return sched


def render_schedule_gantt(sched, selected_date, all_expert_names=None):
    """Render the Gantt-style timeline chart with merged hover for overlapping slots."""

    # Experts with interviews — sorted by earliest start
    if not sched.empty:
        sched = sched.copy()
        ref = datetime(2000, 1, 1)
        sched["start_dt"] = sched["start_min"].apply(lambda m: ref + timedelta(minutes=int(m)))
        sched["end_dt"] = sched["end_min"].apply(lambda m: ref + timedelta(minutes=int(m)))
        experts_with_interviews = sched.groupby("expert_name")["start_min"].min().sort_values().index.tolist()
    else:
        ref = datetime(2000, 1, 1)
        experts_with_interviews = []

    # ALL active experts (including those with zero interviews)
    if all_expert_names:
        experts_without = [e for e in all_expert_names if e not in experts_with_interviews]
        expert_order = experts_with_interviews + sorted(experts_without)
    else:
        expert_order = experts_with_interviews

    if not expert_order:
        st.info("No experts found for " + str(selected_date))
        return

    status_colors = {
        "completed": "#2ecc71",
        "rescheduled": "#f39c12",
        "cancelled": "#e74c3c",
        "pending": "#3498db",
    }

    fig = go.Figure()

    if not sched.empty:
        for expert in experts_with_interviews:
            expert_df = sched[sched["expert_name"] == expert].sort_values("start_min")

            # Build clusters of overlapping intervals
            clusters = []
            for _, row in expert_df.iterrows():
                row_dict = row.to_dict()
                placed = False
                for cluster in clusters:
                    for c_row in cluster:
                        if row_dict["start_min"] < c_row["end_min"] and c_row["start_min"] < row_dict["end_min"]:
                            cluster.append(row_dict)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    clusters.append([row_dict])

            for cluster in clusters:
                if len(cluster) == 1:
                    row = cluster[0]
                    base_color = status_colors.get(row["task_status"], "#95a5a6")
                    line_dict = dict(color="#ff0000", width=3) if row["has_clash"] else dict(color="white", width=1)

                    hover = (
                        "<b>" + str(row.get("candidate_name", "") or "") + "</b><br>"
                        + "Company: " + str(row.get("company_name", "") or "") + "<br>"
                        + "Round: " + str(row.get("round_name", "") or "") + "<br>"
                        + "Status: " + str(row.get("task_status", "") or "") + "<br>"
                        + "Time: " + str(row["start_label"]) + " – " + str(row["end_label"]) + "<br>"
                        + "Duration: " + str(row["duration"]) + " min"
                    )

                    fig.add_trace(go.Bar(
                        y=[expert],
                        x=[(row["end_dt"] - row["start_dt"]).total_seconds() * 1000],
                        base=[row["start_dt"]],
                        orientation="h",
                        marker=dict(color=base_color, line=line_dict),
                        hovertext=hover,
                        hoverinfo="text",
                        showlegend=False,
                        width=0.6,
                    ))
                else:
                    min_start = min(r["start_min"] for r in cluster)
                    max_end = max(r["end_min"] for r in cluster)
                    start_dt = ref + timedelta(minutes=int(min_start))
                    end_dt = ref + timedelta(minutes=int(max_end))

                    hover_parts = ["<b>⚠️ " + str(len(cluster)) + " OVERLAPPING INTERVIEWS</b><br><br>"]
                    for idx, row in enumerate(cluster):
                        hover_parts.append(
                            "<b>" + str(idx + 1) + ". " + str(row.get("candidate_name", "") or "") + "</b><br>"
                            + "   Company: " + str(row.get("company_name", "") or "") + "<br>"
                            + "   Round: " + str(row.get("round_name", "") or "") + "<br>"
                            + "   Status: " + str(row.get("task_status", "") or "") + "<br>"
                            + "   Time: " + str(row["start_label"]) + " – " + str(row["end_label"]) + "<br>"
                            + "   Duration: " + str(row["duration"]) + " min<br><br>"
                        )

                    base_color = status_colors.get(cluster[0]["task_status"], "#95a5a6")
                    merged_hover = "".join(hover_parts)

                    fig.add_trace(go.Bar(
                        y=[expert],
                        x=[(end_dt - start_dt).total_seconds() * 1000],
                        base=[start_dt],
                        orientation="h",
                        marker=dict(color=base_color, line=dict(color="#ff0000", width=3)),
                        hovertext=merged_hover,
                        hoverinfo="text",
                        showlegend=False,
                        width=0.6,
                    ))

                    for row in cluster:
                        bar_color = status_colors.get(row["task_status"], "#95a5a6")
                        fig.add_trace(go.Bar(
                            y=[expert],
                            x=[(row["end_dt"] - row["start_dt"]).total_seconds() * 1000],
                            base=[row["start_dt"]],
                            orientation="h",
                            marker=dict(color=bar_color, line=dict(color="#ff0000", width=2),
                                        opacity=0.7),
                            hovertext=merged_hover,
                            hoverinfo="text",
                            showlegend=False,
                            width=0.3,
                        ))

    # Add invisible traces for experts with NO interviews (so they appear on y-axis)
    experts_on_chart = set(experts_with_interviews)
    for expert in expert_order:
        if expert not in experts_on_chart:
            fig.add_trace(go.Bar(
                y=[expert],
                x=[0],
                base=[ref + timedelta(hours=8)],
                orientation="h",
                marker=dict(color="rgba(0,0,0,0)"),
                hovertext="No interviews scheduled",
                hoverinfo="text",
                showlegend=False,
                width=0.6,
            ))

    # Legend traces
    for status, color in status_colors.items():
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker=dict(color=color),
            name=TASK_LABEL.get(status, status.title()),
            showlegend=True,
        ))
    fig.add_trace(go.Bar(
        y=[None], x=[None],
        marker=dict(color="#95a5a6", line=dict(color="#ff0000", width=3)),
        name="⚠️ Clash (red border)",
        showlegend=True,
    ))

    # Time range
    if not sched.empty:
        min_start_val = sched["start_min"].min()
        max_end_val = sched["end_min"].max()
    else:
        min_start_val = 8 * 60
        max_end_val = 18 * 60

    range_start = ref + timedelta(minutes=max(0, int(min_start_val) - 30))
    range_end = ref + timedelta(minutes=min(1440, int(max_end_val) + 30))

    fig.update_layout(
        title="Expert Schedule — " + str(selected_date) + " (EDT)",
        height=max(500, len(expert_order) * 45),
        xaxis=dict(
            type="date",
            tickformat="%I:%M %p",
            range=[range_start, range_end],
            title="Time (EDT)",
            dtick=30 * 60 * 1000,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=expert_order[::-1],
            title="Expert",
        ),
        barmode="overlay",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

def render_schedule_view(all_data, active_expert_df):
    """Main renderer for the Schedule View page."""
    st.header("Schedule View")
    st.caption("Select any date to see Interview Support interviews organized by expert timeline (EDT)")

    if "date" not in all_data.columns:
        st.warning("No date column available in data.")
        return

    valid_dates = all_data["date"].dropna()
    if valid_dates.empty:
        st.warning("No valid dates found.")
        return

    min_d = valid_dates.min().date()
    max_d = valid_dates.max().date()

    selected_date = st.date_input("Select Date", value=date.today(),
                                   min_value=min_d,
                                   max_value=max_d + timedelta(days=90),
                                   key="schedule_date")

    sched = build_schedule_data(all_data, selected_date)

    # Get ALL active expert names (excluding HCR, Self)
    EXCLUDE_EXPERTS = {"hcr", "self"}
    all_expert_names = sorted([
        e for e in active_expert_df["expert_name"].dropna().unique()
        if e.strip().lower() not in EXCLUDE_EXPERTS
    ]) if "expert_name" in active_expert_df.columns else []

    if sched.empty:
        st.info("No Interview Support interviews with valid time data on " + str(selected_date))
        day_all = all_data[all_data["date"].dt.date == selected_date]
        if not day_all.empty:
            st.caption(str(len(day_all)) + " record(s) found but none have valid start_time.")
        if all_expert_names:
            st.markdown("---")
            render_schedule_gantt(sched, selected_date, all_expert_names)
            st.markdown("---")
            render_availability_summary(sched, selected_date, all_expert_names)
        return

    # ── KPI row ──────────────────────────────────────────────────
    k = st.columns(6)
    k[0].metric("Total Interviews", len(sched))
    k[1].metric("Experts", sched["expert_name"].nunique())
    k[2].metric("Completed", int((sched["task_status"] == "completed").sum()))
    k[3].metric("Pending", int((sched["task_status"] == "pending").sum()))
    k[4].metric("Rescheduled", int((sched["task_status"] == "rescheduled").sum()))
    k[5].metric("Cancelled", int((sched["task_status"] == "cancelled").sum()))

    clash_count = int(sched["has_clash"].sum())
    experts_with_clash = sched[sched["has_clash"]]["expert_name"].nunique()
    if clash_count > 0:
        st.warning("⚠️ **" + str(clash_count) + " interview(s) have clashes** across **" +
                   str(experts_with_clash) + " expert(s)**. Look for red borders in the timeline.")

    st.caption("Showing Interview Support only")

    # ── GANTT CHART ──────────────────────────────────────────────
    st.markdown("---")
    render_schedule_gantt(sched, selected_date, all_expert_names)

    # ── AVAILABILITY SUMMARY ─────────────────────────────────────
    st.markdown("---")
    render_availability_summary(sched, selected_date, all_expert_names)

    # ── CLASH DETAILS ────────────────────────────────────────────
    clash_df = sched[sched["has_clash"]].copy()
    if not clash_df.empty:
        st.markdown("---")
        st.subheader("Clash Details — " + str(selected_date))
        clash_display = clash_df[["expert_name", "candidate_name", "company_name",
                                   "round_name", "support_name", "task_status",
                                   "start_label", "end_label", "duration"]].copy()
        clash_display.columns = ["Expert", "Candidate", "Company", "Round", "Support",
                                  "Status", "Start", "End", "Duration (min)"]
        st.dataframe(clash_display.sort_values(["Expert", "Start"]),
                     use_container_width=True, hide_index=True)

    # ── FULL SCHEDULE TABLE ──────────────────────────────────────
    with st.expander("Full Schedule Table — " + str(selected_date)):
        table_cols = [c for c in ["expert_name", "candidate_name", "company_name",
                                   "round_name", "support_name", "task_status",
                                   "start_label", "end_label", "duration", "has_clash"]
                      if c in sched.columns]
        display = sched[table_cols].copy()
        display.columns = [c.replace("_", " ").title() for c in table_cols]
        st.dataframe(display.sort_values(["Expert Name", "Start Label"]),
                     use_container_width=True, hide_index=True)
def render_top_candidates_analysis(df, title_suffix="", min_interviews=5):
    """Render Top 10 Strong and Top 10 Weak candidates based on avg sentiment.
    
    Only considers rows where:
      - task_status == 'completed'
      - expert_name is NOT 'Self' (case-insensitive)
      - candidate has at least `min_interviews` completed interviews
    """
    if df.empty or "candidate_name" not in df.columns or "sentiment_score" not in df.columns:
        st.info("No data available for candidate strength analysis" + title_suffix + ".")
        return

    # Filter: completed only, expert != Self
    filtered = df.copy()
    if "task_status" in filtered.columns:
        filtered = filtered[filtered["task_status"].astype(str).str.strip().str.lower() == "completed"]
    if "expert_name" in filtered.columns:
        filtered = filtered[filtered["expert_name"].astype(str).str.strip().str.lower() != "self"]

    # Drop rows without sentiment
    filtered = filtered.dropna(subset=["sentiment_score"])

    if filtered.empty:
        st.info("No completed interviews with sentiment data available" + title_suffix + ".")
        return

    # Aggregate per candidate
    cand_agg = filtered.groupby("candidate_name").agg(
        completed_interviews=("sentiment_score", "size"),
        avg_sentiment=("sentiment_score", "mean"),
        min_sentiment=("sentiment_score", "min"),
        max_sentiment=("sentiment_score", "max"),
        positive_count=("sentiment_score", lambda x: int((x >= 20).sum())),
        neutral_count=("sentiment_score", lambda x: int(((x > -20) & (x < 20)).sum())),
        negative_count=("sentiment_score", lambda x: int((x <= -20).sum())),
        companies=("company_name", "nunique") if "company_name" in filtered.columns else ("sentiment_score", "size"),
        experts=("expert_name", "nunique") if "expert_name" in filtered.columns else ("sentiment_score", "size"),
    ).reset_index()

    cand_agg["avg_sentiment"] = cand_agg["avg_sentiment"].round(1)
    cand_agg["min_sentiment"] = cand_agg["min_sentiment"].round(1)
    cand_agg["max_sentiment"] = cand_agg["max_sentiment"].round(1)

    # Filter: at least min_interviews completed
    qualified = cand_agg[cand_agg["completed_interviews"] >= min_interviews].copy()

    if qualified.empty:
        st.info(
            "No candidates with at least " + str(min_interviews) +
            " completed interviews found" + title_suffix + "."
        )
        return

    # Sort for strong (descending) and weak (ascending)
    strong = qualified.sort_values("avg_sentiment", ascending=False).head(10).reset_index(drop=True)
    weak = qualified.sort_values("avg_sentiment", ascending=True).head(10).reset_index(drop=True)

    # ── Section Header ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Candidate Strength Analysis" + title_suffix)
    st.caption(
        "Based on average feedback sentiment score. Only candidates with "
        + str(min_interviews) + "+ completed interviews (excluding Self-expert) are considered."
    )

    # ── KPI Row ──────────────────────────────────────────────────
    k = st.columns(5)
    k[0].metric("Qualified Candidates", len(qualified))
    k[1].metric("Min Interviews Required", min_interviews)
    k[2].metric("Team Avg Sentiment", f"{qualified['avg_sentiment'].mean():+.1f}%")
    if not strong.empty:
        k[3].metric("Strongest Candidate", str(strong.iloc[0]["candidate_name"]),
                     delta=f"{strong.iloc[0]['avg_sentiment']:+.1f}%", delta_color="normal")
    if not weak.empty:
        k[4].metric("Weakest Candidate", str(weak.iloc[0]["candidate_name"]),
                     delta=f"{weak.iloc[0]['avg_sentiment']:+.1f}%", delta_color="inverse")

    # ── Strong and Weak Side by Side ─────────────────────────────
    col_strong, col_weak = st.columns(2)

    with col_strong:
        st.markdown("### ✅ Top 10 Strong Candidates")
        if not strong.empty:
            strong_sorted = strong.sort_values("avg_sentiment", ascending=True)
            fig_strong = go.Figure(go.Bar(
                y=strong_sorted["candidate_name"],
                x=strong_sorted["avg_sentiment"],
                orientation="h",
                marker_color=["#27ae60" if v >= 40 else "#2ecc71" if v >= 20 else "#f39c12"
                              for v in strong_sorted["avg_sentiment"]],
                text=strong_sorted["avg_sentiment"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
            ))
            fig_strong.update_layout(
                title="Top 10 Strong Candidates (Avg Sentiment)",
                height=max(400, len(strong_sorted) * 40),
                xaxis_title="Avg Sentiment Score (%)",
                xaxis=dict(range=[
                    min(0, strong_sorted["avg_sentiment"].min() - 10),
                    strong_sorted["avg_sentiment"].max() + 15
                ]),
            )
            st.plotly_chart(fig_strong, use_container_width=True)

    with col_weak:
        st.markdown("### ⚠️ Top 10 Weak Candidates")
        if not weak.empty:
            weak_sorted = weak.sort_values("avg_sentiment", ascending=True)
            fig_weak = go.Figure(go.Bar(
                y=weak_sorted["candidate_name"],
                x=weak_sorted["avg_sentiment"],
                orientation="h",
                marker_color=["#c0392b" if v <= -40 else "#e74c3c" if v <= -20 else "#f39c12"
                              for v in weak_sorted["avg_sentiment"]],
                text=weak_sorted["avg_sentiment"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
            ))
            fig_weak.update_layout(
                title="Top 10 Weak Candidates (Avg Sentiment)",
                height=max(400, len(weak_sorted) * 40),
                xaxis_title="Avg Sentiment Score (%)",
                xaxis=dict(range=[
                    weak_sorted["avg_sentiment"].min() - 15,
                    max(0, weak_sorted["avg_sentiment"].max() + 10)
                ]),
            )
            st.plotly_chart(fig_weak, use_container_width=True)

    # ── Sentiment Breakdown Comparison ───────────────────────────
    st.markdown("---")
    st.subheader("Sentiment Breakdown Comparison" + title_suffix)

    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        if not strong.empty:
            fig_sb = go.Figure()
            strong_display = strong.sort_values("avg_sentiment", ascending=False)
            fig_sb.add_trace(go.Bar(
                y=strong_display["candidate_name"], x=strong_display["positive_count"],
                name="Positive", orientation="h", marker_color="#2ecc71",
                text=strong_display["positive_count"], textposition="inside",
            ))
            fig_sb.add_trace(go.Bar(
                y=strong_display["candidate_name"], x=strong_display["neutral_count"],
                name="Neutral", orientation="h", marker_color="#f39c12",
                text=strong_display["neutral_count"], textposition="inside",
            ))
            fig_sb.add_trace(go.Bar(
                y=strong_display["candidate_name"], x=strong_display["negative_count"],
                name="Negative", orientation="h", marker_color="#e74c3c",
                text=strong_display["negative_count"], textposition="inside",
            ))
            fig_sb.update_layout(
                barmode="stack", title="Strong Candidates — Sentiment Breakdown",
                height=max(400, len(strong_display) * 40),
                xaxis_title="Feedback Count",
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_sb, use_container_width=True)

    with comp_c2:
        if not weak.empty:
            fig_wb = go.Figure()
            weak_display = weak.sort_values("avg_sentiment", ascending=True)
            fig_wb.add_trace(go.Bar(
                y=weak_display["candidate_name"], x=weak_display["positive_count"],
                name="Positive", orientation="h", marker_color="#2ecc71",
                text=weak_display["positive_count"], textposition="inside",
            ))
            fig_wb.add_trace(go.Bar(
                y=weak_display["candidate_name"], x=weak_display["neutral_count"],
                name="Neutral", orientation="h", marker_color="#f39c12",
                text=weak_display["neutral_count"], textposition="inside",
            ))
            fig_wb.add_trace(go.Bar(
                y=weak_display["candidate_name"], x=weak_display["negative_count"],
                name="Negative", orientation="h", marker_color="#e74c3c",
                text=weak_display["negative_count"], textposition="inside",
            ))
            fig_wb.update_layout(
                barmode="stack", title="Weak Candidates — Sentiment Breakdown",
                height=max(400, len(weak_display) * 40),
                xaxis_title="Feedback Count",
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_wb, use_container_width=True)

    # ── Candidate Landscape Scatter — ALL candidates with 1+ interview ──
    st.markdown("---")
    st.subheader("Candidate Performance Matrix" + title_suffix)
    st.caption(
        "All candidates with at least 1 completed interview (excluding Self-expert). "
        "Color-coded by quadrant: 🟢 High Volume + High Sentiment, 🔵 Low Volume + High Sentiment, "
        "🟠 High Volume + Low Sentiment, 🔴 Low Volume + Low Sentiment"
    )

    # Build landscape data with min 1 interview
    landscape = cand_agg[cand_agg["completed_interviews"] >= 1].copy()

    if landscape.empty:
        st.info("No candidates with completed interviews for landscape view.")
    else:
        avg_sent = landscape["avg_sentiment"].median()
        avg_vol = landscape["completed_interviews"].median()

        # Assign quadrant
        def _assign_quadrant(row):
            high_vol = row["completed_interviews"] >= avg_vol
            high_sent = row["avg_sentiment"] >= avg_sent
            if high_vol and high_sent:
                return "High Volume + High Sentiment"
            elif not high_vol and high_sent:
                return "Low Volume + High Sentiment"
            elif high_vol and not high_sent:
                return "High Volume + Low Sentiment"
            else:
                return "Low Volume + Low Sentiment"

        landscape["quadrant"] = landscape.apply(_assign_quadrant, axis=1)

        quadrant_colors = {
            "High Volume + High Sentiment": "#2ecc71",
            "Low Volume + High Sentiment": "#3498db",
            "High Volume + Low Sentiment": "#f39c12",
            "Low Volume + Low Sentiment": "#e74c3c",
        }

        quadrant_symbols = {
            "High Volume + High Sentiment": "star",
            "Low Volume + High Sentiment": "diamond",
            "High Volume + Low Sentiment": "triangle-up",
            "Low Volume + Low Sentiment": "circle",
        }

        fig_landscape = go.Figure()

        for quadrant in ["High Volume + High Sentiment", "Low Volume + High Sentiment",
                         "High Volume + Low Sentiment", "Low Volume + Low Sentiment"]:
            q_data = landscape[landscape["quadrant"] == quadrant]
            if q_data.empty:
                continue

            hover_texts = []
            for _, row in q_data.iterrows():
                hover_texts.append(
                    "<b>" + str(row["candidate_name"]) + "</b><br>"
                    + "Completed Interviews: " + str(int(row["completed_interviews"])) + "<br>"
                    + "Avg Sentiment: " + f"{row['avg_sentiment']:+.1f}%" + "<br>"
                    + "Min: " + f"{row['min_sentiment']:+.1f}%" + " | Max: " + f"{row['max_sentiment']:+.1f}%" + "<br>"
                    + "Positive: " + str(int(row["positive_count"]))
                    + " | Neutral: " + str(int(row["neutral_count"]))
                    + " | Negative: " + str(int(row["negative_count"])) + "<br>"
                    + "Companies: " + str(int(row["companies"]))
                    + " | Experts: " + str(int(row["experts"])) + "<br>"
                    + "Quadrant: " + quadrant
                )

            fig_landscape.add_trace(go.Scatter(
                x=q_data["completed_interviews"],
                y=q_data["avg_sentiment"],
                mode="markers+text",
                text=q_data["candidate_name"],
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    size=10,
                    color=quadrant_colors[quadrant],
                    symbol=quadrant_symbols[quadrant],
                    line=dict(width=1, color="white"),
                ),
                hovertext=hover_texts,
                hoverinfo="text",
                name=quadrant,
            ))

        # Quadrant divider lines
        fig_landscape.add_hline(y=avg_sent, line_dash="dash", line_color="#8e44ad", opacity=0.5,
                                annotation_text=f"Median Sentiment: {avg_sent:+.1f}%",
                                annotation_position="top right")
        fig_landscape.add_vline(x=avg_vol, line_dash="dash", line_color="#8e44ad", opacity=0.5,
                                annotation_text=f"Median Interviews: {avg_vol:.0f}",
                                annotation_position="top right")

        # Quadrant background labels
        x_min = max(0, landscape["completed_interviews"].min() - 1)
        x_max = landscape["completed_interviews"].max() + 2
        y_min = landscape["avg_sentiment"].min() - 10
        y_max = landscape["avg_sentiment"].max() + 10

        fig_landscape.add_annotation(x=x_max * 0.85, y=y_max * 0.9, text="⭐ Stars",
                                     showarrow=False, font=dict(size=14, color="#2ecc71"), opacity=0.4)
        fig_landscape.add_annotation(x=x_min + 1, y=y_max * 0.9, text="💎 Potential",
                                     showarrow=False, font=dict(size=14, color="#3498db"), opacity=0.4)
        fig_landscape.add_annotation(x=x_max * 0.85, y=y_min + 5, text="⚠️ At Risk",
                                     showarrow=False, font=dict(size=14, color="#f39c12"), opacity=0.4)
        fig_landscape.add_annotation(x=x_min + 1, y=y_min + 5, text="🔴 Concern",
                                     showarrow=False, font=dict(size=14, color="#e74c3c"), opacity=0.4)

        # Quadrant count KPIs
        q_counts = landscape["quadrant"].value_counts()
        qk = st.columns(4)
        qk[0].metric("⭐ Stars (High Vol + High Sent)",
                      int(q_counts.get("High Volume + High Sentiment", 0)))
        qk[1].metric("💎 Potential (Low Vol + High Sent)",
                      int(q_counts.get("Low Volume + High Sentiment", 0)))
        qk[2].metric("⚠️ At Risk (High Vol + Low Sent)",
                      int(q_counts.get("High Volume + Low Sentiment", 0)))
        qk[3].metric("🔴 Concern (Low Vol + Low Sent)",
                      int(q_counts.get("Low Volume + Low Sentiment", 0)))

        fig_landscape.update_layout(
            title="Candidate Performance Matrix — All Candidates (1+ Completed Interview)" + title_suffix,
            height=650,
            xaxis_title="Completed Interviews",
            yaxis_title="Avg Sentiment Score (%)",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            hovermode="closest",
        )
        st.plotly_chart(fig_landscape, use_container_width=True)

        # Quadrant detail tables
        with st.expander("⭐ Stars — High Volume + High Sentiment" + title_suffix):
            stars = landscape[landscape["quadrant"] == "High Volume + High Sentiment"].sort_values("avg_sentiment", ascending=False)
            if not stars.empty:
                st.dataframe(stars[["candidate_name", "completed_interviews", "avg_sentiment",
                                    "positive_count", "neutral_count", "negative_count", "companies", "experts"]].rename(
                    columns={"candidate_name": "Candidate", "completed_interviews": "Interviews",
                             "avg_sentiment": "Avg Sentiment", "positive_count": "Positive",
                             "neutral_count": "Neutral", "negative_count": "Negative",
                             "companies": "Companies", "experts": "Experts"}),
                    use_container_width=True, hide_index=True)
            else:
                st.info("No candidates in this quadrant.")

        with st.expander("💎 Potential — Low Volume + High Sentiment" + title_suffix):
            potential = landscape[landscape["quadrant"] == "Low Volume + High Sentiment"].sort_values("avg_sentiment", ascending=False)
            if not potential.empty:
                st.dataframe(potential[["candidate_name", "completed_interviews", "avg_sentiment",
                                        "positive_count", "neutral_count", "negative_count", "companies", "experts"]].rename(
                    columns={"candidate_name": "Candidate", "completed_interviews": "Interviews",
                             "avg_sentiment": "Avg Sentiment", "positive_count": "Positive",
                             "neutral_count": "Neutral", "negative_count": "Negative",
                             "companies": "Companies", "experts": "Experts"}),
                    use_container_width=True, hide_index=True)
            else:
                st.info("No candidates in this quadrant.")

        with st.expander("⚠️ At Risk — High Volume + Low Sentiment" + title_suffix):
            at_risk = landscape[landscape["quadrant"] == "High Volume + Low Sentiment"].sort_values("avg_sentiment", ascending=True)
            if not at_risk.empty:
                st.dataframe(at_risk[["candidate_name", "completed_interviews", "avg_sentiment",
                                      "positive_count", "neutral_count", "negative_count", "companies", "experts"]].rename(
                    columns={"candidate_name": "Candidate", "completed_interviews": "Interviews",
                             "avg_sentiment": "Avg Sentiment", "positive_count": "Positive",
                             "neutral_count": "Neutral", "negative_count": "Negative",
                             "companies": "Companies", "experts": "Experts"}),
                    use_container_width=True, hide_index=True)
            else:
                st.info("No candidates in this quadrant.")

        with st.expander("🔴 Concern — Low Volume + Low Sentiment" + title_suffix):
            concern = landscape[landscape["quadrant"] == "Low Volume + Low Sentiment"].sort_values("avg_sentiment", ascending=True)
            if not concern.empty:
                st.dataframe(concern[["candidate_name", "completed_interviews", "avg_sentiment",
                                      "positive_count", "neutral_count", "negative_count", "companies", "experts"]].rename(
                    columns={"candidate_name": "Candidate", "completed_interviews": "Interviews",
                             "avg_sentiment": "Avg Sentiment", "positive_count": "Positive",
                             "neutral_count": "Neutral", "negative_count": "Negative",
                             "companies": "Companies", "experts": "Experts"}),
                    use_container_width=True, hide_index=True)
            else:
                st.info("No candidates in this quadrant.")

    # ── Detailed Tables ──────────────────────────────────────────
    display_cols = ["candidate_name", "completed_interviews", "avg_sentiment",
                    "min_sentiment", "max_sentiment", "positive_count",
                    "neutral_count", "negative_count"]
    display_headers = ["Candidate", "Completed Interviews", "Avg Sentiment",
                       "Min Sentiment", "Max Sentiment", "Positive",
                       "Neutral", "Negative"]

    if "company_name" in filtered.columns:
        display_cols.append("companies")
        display_headers.append("Companies")
    if "expert_name" in filtered.columns:
        display_cols.append("experts")
        display_headers.append("Experts")

    with st.expander("✅ Strong Candidates — Full Data" + title_suffix):
        if not strong.empty:
            strong_disp = strong[display_cols].copy()
            strong_disp.columns = display_headers
            strong_disp.index = range(1, len(strong_disp) + 1)
            strong_disp.index.name = "Rank"
            st.dataframe(strong_disp, use_container_width=True)
        else:
            st.info("No strong candidates found.")

    with st.expander("⚠️ Weak Candidates — Full Data" + title_suffix):
        if not weak.empty:
            weak_disp = weak[display_cols].copy()
            weak_disp.columns = display_headers
            weak_disp.index = range(1, len(weak_disp) + 1)
            weak_disp.index.name = "Rank"
            st.dataframe(weak_disp, use_container_width=True)
        else:
            st.info("No weak candidates found.")

    with st.expander("All Qualified Candidates (" + str(min_interviews) + "+ interviews)" + title_suffix):
        all_disp = qualified.sort_values("avg_sentiment", ascending=False)[display_cols].copy()
        all_disp.columns = display_headers
        all_disp.index = range(1, len(all_disp) + 1)
        all_disp.index.name = "Rank"
        st.dataframe(all_disp, use_container_width=True)

# ── AVAILABILITY SUMMARY ─────────────────────────────────────
def render_availability_summary(sched, selected_date, all_expert_names=None):
    """Show expert availability — free slots between interviews."""
    st.subheader("Expert Availability — " + str(selected_date) + " (EDT)")
    st.caption("Free gaps between scheduled interviews (working hours: 8 AM – 10 PM EDT)")

    work_start = 8 * 60
    work_end = 22 * 60

    avail_rows = []

    if not sched.empty:
        for expert, grp in sched.groupby("expert_name"):
            intervals = sorted(zip(grp["start_min"], grp["end_min"]))
            merged = [intervals[0]]
            for s, e in intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))

            free_slots = []
            prev_end = work_start
            for s, e in merged:
                gap_start = max(prev_end, work_start)
                gap_end = min(s, work_end)
                if gap_end > gap_start and (gap_end - gap_start) >= 15:
                    free_slots.append(_minutes_to_label(gap_start) + " – " + _minutes_to_label(gap_end) +
                                      " (" + str(int(gap_end - gap_start)) + " min)")
                prev_end = max(prev_end, e)

            if prev_end < work_end and (work_end - prev_end) >= 15:
                free_slots.append(_minutes_to_label(prev_end) + " – " + _minutes_to_label(work_end) +
                                  " (" + str(int(work_end - prev_end)) + " min)")

            total_busy = sum(e - s for s, e in merged)
            total_interviews = len(grp)
            has_clash = grp["has_clash"].any()

            avail_rows.append({
                "Expert": expert,
                "Interviews": total_interviews,
                "Busy Time": str(int(total_busy)) + " min",
                "Clashes": "⚠️ Yes" if has_clash else "No",
                "Free Slots": " | ".join(free_slots) if free_slots else "No free slots",
            })

    if all_expert_names:
        experts_in_sched = set(sched["expert_name"].unique()) if not sched.empty else set()
        for expert in sorted(all_expert_names):
            if expert not in experts_in_sched:
                avail_rows.append({
                    "Expert": expert,
                    "Interviews": 0,
                    "Busy Time": "0 min",
                    "Clashes": "No",
                    "Free Slots": "Fully available (8:00 AM – 10:00 PM EDT)",
                })

    if not avail_rows:
        st.info("No expert data available.")
        return

    avail_df = pd.DataFrame(avail_rows).sort_values("Interviews", ascending=False)
    st.dataframe(avail_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
#  ASSESSMENT CONVERSION ANALYTICS
#  Tracks which assessments led to interviews within 60 days
#  using fuzzy company name matching (cosine similarity on char n-grams)
# ═══════════════════════════════════════════════════════════════════

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

@st.cache_data(ttl=600)
def _build_company_similarity_cache(company_list_a, company_list_b, threshold=0.55):
    """Build a mapping of company_a -> best matching company_b using TF-IDF cosine similarity."""
    if not company_list_a or not company_list_b:
        return {}

    def _clean(name):
        name = str(name).lower().strip()
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\b(inc|llc|ltd|corp|corporation|pvt|private|limited|the|group|services|technologies|technology|consulting|solutions)\b', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    cleaned_a = [_clean(c) for c in company_list_a]
    cleaned_b = [_clean(c) for c in company_list_b]

    all_names = cleaned_a + cleaned_b
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(all_names)

    tfidf_a = tfidf_matrix[:len(cleaned_a)]
    tfidf_b = tfidf_matrix[len(cleaned_a):]

    sim_matrix = sklearn_cosine(tfidf_a, tfidf_b)

    mapping = {}
    for i, comp_a in enumerate(company_list_a):
        best_idx = sim_matrix[i].argmax()
        best_score = sim_matrix[i][best_idx]
        if best_score >= threshold:
            mapping[comp_a] = (company_list_b[best_idx], round(float(best_score), 3))
        else:
            mapping[comp_a] = (None, round(float(best_score), 3))

    return mapping


def compute_assessment_conversions(all_data, window_days=60, similarity_threshold=0.55):
    """Compute assessment-to-interview conversions.

    For each assessment record, check if there's an interview with the same
    candidate + similar company within `window_days` of the assessment date.

    Returns a DataFrame with one row per assessment, columns:
      All original assessment columns plus:
      converted (0/1), matched_interview_date, matched_company,
      match_score, days_to_interview, interview_round, interview_expert,
      interview_status, assessment_month
    """
    if all_data.empty or "support_name" not in all_data.columns:
        return pd.DataFrame()

    assessments = all_data[all_data["support_name"].str.lower() == "assessment support"].copy()
    interviews = all_data[all_data["support_name"].str.lower() == "interview support"].copy()

    # Only consider completed assessments
    if "task_status" in assessments.columns:
        assessments = assessments[assessments["task_status"].astype(str).str.strip().str.lower() == "completed"]

    if assessments.empty:
        return pd.DataFrame()


    if "date" not in assessments.columns or "candidate_name" not in assessments.columns:
        return pd.DataFrame()

    assessments = assessments.dropna(subset=["date", "candidate_name"]).copy()
    interviews = interviews.dropna(subset=["date", "candidate_name"]).copy() if not interviews.empty else pd.DataFrame()

    assessments["assessment_month"] = assessments["date"].dt.to_period("M").astype(str)

    # Get unique company names from both sides
    assess_companies = assessments["company_name"].dropna().unique().tolist() if "company_name" in assessments.columns else []
    interview_companies = interviews["company_name"].dropna().unique().tolist() if not interviews.empty and "company_name" in interviews.columns else []

    # Build company similarity mapping
    company_map = _build_company_similarity_cache(
        tuple(assess_companies), tuple(interview_companies), similarity_threshold
    ) if assess_companies and interview_companies else {}

    # Process each assessment
    result_rows = []
    for _, assess_row in assessments.iterrows():
        candidate = assess_row.get("candidate_name", "")
        company = assess_row.get("company_name", "")
        assess_date = assess_row["date"]
        expert = assess_row.get("expert_name", "")

        converted = 0
        matched_interview_date = None
        matched_company = None
        match_score = None
        days_to_interview = None
        interview_round = None
        interview_expert = None
        interview_status = None

        if not interviews.empty and candidate:
            # Filter interviews for same candidate
            cand_interviews = interviews[
                interviews["candidate_name"].str.lower().str.strip() == str(candidate).lower().strip()
            ]

            if not cand_interviews.empty and company:
                # Get matching company names
                mapped = company_map.get(company, (None, 0))
                matched_comp_name = mapped[0]
                comp_score = mapped[1]

                if matched_comp_name:
                    # Filter by matched company
                    comp_interviews = cand_interviews[
                        cand_interviews["company_name"].str.lower().str.strip() == matched_comp_name.lower().strip()
                    ]
                else:
                    # Try exact match as fallback
                    comp_interviews = cand_interviews[
                        cand_interviews["company_name"].str.lower().str.strip() == str(company).lower().strip()
                    ]
                    comp_score = 1.0 if not comp_interviews.empty else comp_score

                if not comp_interviews.empty:
                    # Check within window
                    window_end = assess_date + timedelta(days=window_days)
                    within_window = comp_interviews[
                        (comp_interviews["date"] >= assess_date) &
                        (comp_interviews["date"] <= window_end)
                    ]

                    if not within_window.empty:
                        converted = 1
                        first_interview = within_window.sort_values("date").iloc[0]
                        matched_interview_date = first_interview["date"]
                        matched_company = first_interview.get("company_name", "")
                        match_score = comp_score
                        days_to_interview = (matched_interview_date - assess_date).days
                        interview_round = first_interview.get("round_name", "")
                        interview_expert = first_interview.get("expert_name", "")
                        interview_status = first_interview.get("task_status", "")

        row = assess_row.to_dict()
        row.update({
            "converted": converted,
            "matched_interview_date": matched_interview_date,
            "matched_company": matched_company,
            "match_score": match_score,
            "days_to_interview": days_to_interview,
            "interview_round": interview_round,
            "interview_expert": interview_expert,
            "interview_status": interview_status,
            "assessment_month": assess_row["date"].to_period("M"),
        })
        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    result["assessment_month"] = result["date"].dt.to_period("M").astype(str)
    return result


def render_assessment_conversion_kpi(conv_df, title_suffix=""):
    """Render the main conversion KPI row."""
    if conv_df.empty:
        st.info("No assessment data available for conversion analysis" + title_suffix + ".")
        return

    total = len(conv_df)
    converted = int(conv_df["converted"].sum())
    not_converted = total - converted
    pct = round(converted / total * 100, 1) if total > 0 else 0
    avg_days = conv_df.loc[conv_df["converted"] == 1, "days_to_interview"].mean()
    avg_days_str = f"{avg_days:.1f} days" if pd.notna(avg_days) else "N/A"
    unique_candidates = conv_df["candidate_name"].nunique() if "candidate_name" in conv_df.columns else 0
    unique_companies = conv_df["company_name"].nunique() if "company_name" in conv_df.columns else 0
    unique_experts = conv_df["expert_name"].nunique() if "expert_name" in conv_df.columns else 0

    k = st.columns(7)
    k[0].metric("Total Assessments", total)
    k[1].metric("Converted to Interview", converted)
    k[2].metric("Not Converted", not_converted)
    k[3].metric("Conversion Rate", f"{pct}%",
                delta="Higher is better", delta_color="normal")
    k[4].metric("Avg Days to Interview", avg_days_str)
    k[5].metric("Candidates", unique_candidates)
    k[6].metric("Companies", unique_companies)


def render_assessment_conversion_charts(conv_df, title_suffix=""):
    """Render the full set of assessment conversion charts."""
    if conv_df.empty:
        return

    total = len(conv_df)
    converted = int(conv_df["converted"].sum())
    not_converted = total - converted
    pct = round(converted / total * 100, 1) if total > 0 else 0

    # ── Conversion Donut ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Assessment Conversion Overview" + title_suffix)

    ov_c1, ov_c2 = st.columns(2)
    with ov_c1:
        fig_donut = go.Figure(go.Pie(
            labels=["Converted", "Not Converted"],
            values=[converted, not_converted],
            hole=0.5,
            marker=dict(colors=["#2ecc71", "#e74c3c"]),
            textinfo="label+value+percent",
        ))
        fig_donut.update_layout(title="Conversion Split" + title_suffix, height=400, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with ov_c2:
        # Days to interview distribution
        converted_df = conv_df[conv_df["converted"] == 1]
        if not converted_df.empty and "days_to_interview" in converted_df.columns:
            valid_days = converted_df["days_to_interview"].dropna()
            if not valid_days.empty:
                fig_hist = go.Figure(go.Histogram(
                    x=valid_days, nbinsx=15,
                    marker_color="#3498db",
                    marker_line=dict(color="white", width=1),
                ))
                avg_d = valid_days.mean()
                fig_hist.add_vline(x=avg_d, line_dash="dash", line_color="#e74c3c",
                                   annotation_text=f"Avg: {avg_d:.1f} days")
                fig_hist.update_layout(title="Days to Interview Distribution" + title_suffix,
                                       height=400, xaxis_title="Days", yaxis_title="Count")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No days-to-interview data available.")
        else:
            st.info("No converted assessments to show distribution.")

    # ── Monthly Conversion Trend ─────────────────────────────────
    st.markdown("---")
    st.subheader("Monthly Assessment Conversion Trend" + title_suffix)

    monthly_conv = conv_df.groupby("assessment_month").agg(
        total=("converted", "size"),
        converted=("converted", "sum"),
    ).reset_index()
    monthly_conv["pct"] = (monthly_conv["converted"] / monthly_conv["total"] * 100).round(1)
    monthly_conv["not_converted"] = monthly_conv["total"] - monthly_conv["converted"]

    mc1, mc2 = st.columns(2)
    with mc1:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Bar(x=monthly_conv["assessment_month"], y=monthly_conv["converted"],
                                name="Converted", marker_color="#2ecc71",
                                text=monthly_conv["converted"], textposition="inside"))
        fig_mc.add_trace(go.Bar(x=monthly_conv["assessment_month"], y=monthly_conv["not_converted"],
                                name="Not Converted", marker_color="#e74c3c",
                                text=monthly_conv["not_converted"], textposition="inside"))
        fig_mc.update_layout(barmode="stack", title="Monthly Assessments: Converted vs Not",
                             height=420, yaxis_title="Assessments",
                             legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
        st.plotly_chart(fig_mc, use_container_width=True)

    with mc2:
        fig_pct = go.Figure()
        colors = ["#2ecc71" if p >= 50 else "#f39c12" if p >= 25 else "#e74c3c"
                  for p in monthly_conv["pct"]]
        fig_pct.add_trace(go.Scatter(
            x=monthly_conv["assessment_month"], y=monthly_conv["pct"],
            mode="lines+markers+text",
            text=monthly_conv["pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="top center",
            line=dict(color="#8e44ad", width=3),
            marker=dict(size=10, color=colors),
        ))
        fig_pct.update_layout(title="Monthly Conversion Rate %",
                              height=420, yaxis_title="Conversion %",
                              yaxis=dict(range=[0, max(100, monthly_conv["pct"].max() + 10)]))
        st.plotly_chart(fig_pct, use_container_width=True)

    # ── Expert-wise Conversion ───────────────────────────────────
    if "expert_name" in conv_df.columns:
        st.markdown("---")
        st.subheader("Expert-wise Assessment Conversion" + title_suffix)

        expert_conv = conv_df.groupby("expert_name").agg(
            total=("converted", "size"),
            converted=("converted", "sum"),
            avg_days=("days_to_interview", lambda x: round(x.dropna().mean(), 1) if x.dropna().any() else None),
            candidates=("candidate_name", "nunique"),
            companies=("company_name", "nunique"),
        ).reset_index()
        expert_conv["pct"] = (expert_conv["converted"] / expert_conv["total"] * 100).round(1)
        expert_conv["not_converted"] = expert_conv["total"] - expert_conv["converted"]
        expert_conv = expert_conv.sort_values("pct", ascending=False)

        # Top/Bottom performers
        if len(expert_conv) >= 2:
            top_performer = expert_conv.iloc[0]
            low_performer = expert_conv.iloc[-1]
            avg_pct = round(expert_conv["pct"].mean(), 1)

            perf_cols = st.columns(3)
            perf_cols[0].metric("Top Converter",
                                str(top_performer["expert_name"]),
                                delta=f"{top_performer['pct']:.1f}%", delta_color="normal")
            perf_cols[1].metric("Needs Improvement",
                                str(low_performer["expert_name"]),
                                delta=f"{low_performer['pct']:.1f}%", delta_color="inverse")
            perf_cols[2].metric("Team Average", f"{avg_pct:.1f}%")

        ec1, ec2 = st.columns(2)
        with ec1:
            expert_sorted = expert_conv.sort_values("pct", ascending=True)
            avg_pct_val = expert_conv["pct"].mean()
            bar_colors = ["#2ecc71" if p >= avg_pct_val else "#e74c3c" for p in expert_sorted["pct"]]
            fig_ec = go.Figure(go.Bar(
                y=expert_sorted["expert_name"], x=expert_sorted["pct"],
                orientation="h", marker_color=bar_colors,
                text=expert_sorted["pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_ec.add_vline(x=avg_pct_val, line_dash="dash", line_color="#f39c12",
                             annotation_text=f"Avg: {avg_pct_val:.1f}%")
            fig_ec.update_layout(title="Expert Conversion Rate %",
                                 height=max(420, len(expert_sorted) * 35),
                                 xaxis_title="Conversion %",
                                 xaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_ec, use_container_width=True)

        with ec2:
            fig_ev = go.Figure()
            expert_by_vol = expert_conv.sort_values("total", ascending=True)
            fig_ev.add_trace(go.Bar(y=expert_by_vol["expert_name"], x=expert_by_vol["converted"],
                                    name="Converted", orientation="h", marker_color="#2ecc71",
                                    text=expert_by_vol["converted"], textposition="inside"))
            fig_ev.add_trace(go.Bar(y=expert_by_vol["expert_name"], x=expert_by_vol["not_converted"],
                                    name="Not Converted", orientation="h", marker_color="#e74c3c",
                                    text=expert_by_vol["not_converted"], textposition="inside"))
            fig_ev.update_layout(barmode="stack", title="Expert Assessment Volume",
                                 height=max(420, len(expert_by_vol) * 35),
                                 xaxis_title="Assessments",
                                 legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
            st.plotly_chart(fig_ev, use_container_width=True)

        with st.expander("Expert Conversion Data" + title_suffix):
            display_ec = expert_conv[["expert_name", "total", "converted", "not_converted",
                                      "pct", "avg_days", "candidates", "companies"]].copy()
            display_ec.columns = ["Expert", "Total Assessments", "Converted", "Not Converted",
                                  "Conversion %", "Avg Days to Interview", "Candidates", "Companies"]
            st.dataframe(display_ec, use_container_width=True, hide_index=True)

    # ── Candidate-wise Conversion ────────────────────────────────
    if "candidate_name" in conv_df.columns:
        st.markdown("---")
        st.subheader("Candidate-wise Assessment Conversion" + title_suffix)

        cand_conv = conv_df.groupby("candidate_name").agg(
            total=("converted", "size"),
            converted=("converted", "sum"),
            avg_days=("days_to_interview", lambda x: round(x.dropna().mean(), 1) if x.dropna().any() else None),
            experts=("expert_name", "nunique"),
            companies=("company_name", "nunique"),
        ).reset_index()
        cand_conv["pct"] = (cand_conv["converted"] / cand_conv["total"] * 100).round(1)
        cand_conv = cand_conv.sort_values("total", ascending=False)

        cc1, cc2 = st.columns(2)
        with cc1:
            top_cands = cand_conv.sort_values("pct", ascending=False).head(20).sort_values("pct", ascending=True)
            avg_cand_pct = cand_conv["pct"].mean()
            cand_colors = ["#2ecc71" if p >= avg_cand_pct else "#e74c3c" for p in top_cands["pct"]]
            fig_cc = go.Figure(go.Bar(
                y=top_cands["candidate_name"], x=top_cands["pct"],
                orientation="h", marker_color=cand_colors,
                text=top_cands["pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_cc.update_layout(title="Top 20 Candidates: Conversion Rate",
                                 height=max(420, len(top_cands) * 35),
                                 xaxis_title="Conversion %",
                                 xaxis=dict(range=[0, 110]))
            st.plotly_chart(fig_cc, use_container_width=True)

        with cc2:
            top_cands_vol = cand_conv.head(20).sort_values("total", ascending=True)
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(y=top_cands_vol["candidate_name"], x=top_cands_vol["converted"],
                                    name="Converted", orientation="h", marker_color="#2ecc71",
                                    text=top_cands_vol["converted"], textposition="inside"))
            not_conv = top_cands_vol["total"] - top_cands_vol["converted"]
            fig_cv.add_trace(go.Bar(y=top_cands_vol["candidate_name"], x=not_conv,
                                    name="Not Converted", orientation="h", marker_color="#e74c3c",
                                    text=not_conv, textposition="inside"))
            fig_cv.update_layout(barmode="stack", title="Top 20 Candidates: Assessment Volume",
                                 height=max(420, len(top_cands_vol) * 35),
                                 xaxis_title="Assessments",
                                 legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
            st.plotly_chart(fig_cv, use_container_width=True)

        with st.expander("Candidate Conversion Data" + title_suffix):
            display_cc = cand_conv[["candidate_name", "total", "converted", "pct",
                                    "avg_days", "experts", "companies"]].copy()
            display_cc.columns = ["Candidate", "Total Assessments", "Converted", "Conversion %",
                                  "Avg Days to Interview", "Experts", "Companies"]
            st.dataframe(display_cc, use_container_width=True, hide_index=True)

    # ── Company-wise Conversion ──────────────────────────────────
    if "company_name" in conv_df.columns:
        st.markdown("---")
        st.subheader("Company-wise Assessment Conversion" + title_suffix)

        comp_conv = conv_df.groupby("company_name").agg(
            total=("converted", "size"),
            converted=("converted", "sum"),
            avg_days=("days_to_interview", lambda x: round(x.dropna().mean(), 1) if x.dropna().any() else None),
            candidates=("candidate_name", "nunique"),
            experts=("expert_name", "nunique"),
        ).reset_index()
        comp_conv["pct"] = (comp_conv["converted"] / comp_conv["total"] * 100).round(1)
        comp_conv = comp_conv.sort_values("total", ascending=False)

        co1, co2 = st.columns(2)
        with co1:
            top_comps = comp_conv.head(20).sort_values("pct", ascending=True)
            avg_comp_pct = comp_conv["pct"].mean()
            comp_colors = ["#2ecc71" if p >= avg_comp_pct else "#e74c3c" for p in top_comps["pct"]]
            fig_co = go.Figure(go.Bar(
                y=top_comps["company_name"], x=top_comps["pct"],
                orientation="h", marker_color=comp_colors,
                text=top_comps["pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_co.add_vline(x=avg_comp_pct, line_dash="dash", line_color="#f39c12",
                             annotation_text=f"Avg: {avg_comp_pct:.1f}%")
            fig_co.update_layout(title="Top 20 Companies: Conversion Rate",
                                 height=max(420, len(top_comps) * 35),
                                 xaxis_title="Conversion %",
                                 xaxis=dict(range=[0, 110]))
            st.plotly_chart(fig_co, use_container_width=True)

        with co2:
            top_comps_vol = comp_conv.head(20).sort_values("total", ascending=True)
            fig_cov = go.Figure()
            fig_cov.add_trace(go.Bar(y=top_comps_vol["company_name"], x=top_comps_vol["converted"],
                                     name="Converted", orientation="h", marker_color="#2ecc71",
                                     text=top_comps_vol["converted"], textposition="inside"))
            not_conv_c = top_comps_vol["total"] - top_comps_vol["converted"]
            fig_cov.add_trace(go.Bar(y=top_comps_vol["company_name"], x=not_conv_c,
                                     name="Not Converted", orientation="h", marker_color="#e74c3c",
                                     text=not_conv_c, textposition="inside"))
            fig_cov.update_layout(barmode="stack", title="Top 20 Companies: Assessment Volume",
                                  height=max(420, len(top_comps_vol) * 35),
                                  xaxis_title="Assessments",
                                  legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
            st.plotly_chart(fig_cov, use_container_width=True)

        with st.expander("Company Conversion Data" + title_suffix):
            display_co = comp_conv[["company_name", "total", "converted", "pct",
                                    "avg_days", "candidates", "experts"]].copy()
            display_co.columns = ["Company", "Total Assessments", "Converted", "Conversion %",
                                  "Avg Days to Interview", "Candidates", "Experts"]
            st.dataframe(display_co, use_container_width=True, hide_index=True)

    # ── Interview Round Distribution for Converted ───────────────
    converted_only = conv_df[conv_df["converted"] == 1]
    if not converted_only.empty and "interview_round" in converted_only.columns:
        st.markdown("---")
        st.subheader("Which Rounds Did Converted Assessments Reach?" + title_suffix)

        round_dist = converted_only["interview_round"].value_counts()
        if not round_dist.empty:
            rd1, rd2 = st.columns(2)
            with rd1:
                fig_rd = go.Figure(go.Pie(
                    labels=round_dist.index, values=round_dist.values,
                    hole=0.4, textinfo="label+value+percent",
                ))
                fig_rd.update_layout(title="Interview Round Distribution (Converted)", height=420)
                st.plotly_chart(fig_rd, use_container_width=True)
            with rd2:
                fig_rb = go.Figure(go.Bar(
                    y=round_dist.index, x=round_dist.values,
                    orientation="h", marker_color="#1abc9c",
                    text=round_dist.values, textposition="outside",
                ))
                fig_rb.update_layout(title="Converted Assessments by Interview Round",
                                     height=max(400, len(round_dist) * 35),
                                     yaxis=dict(autorange="reversed"),
                                     xaxis_title="Count")
                st.plotly_chart(fig_rb, use_container_width=True)

    # ── Speed Heatmap: Expert x Month (Avg Days to Interview) ────
    if "expert_name" in conv_df.columns:
        converted_with_days = conv_df[(conv_df["converted"] == 1) & (conv_df["days_to_interview"].notna())]
        if not converted_with_days.empty:
            st.markdown("---")
            st.subheader("Expert x Month: Avg Days to Interview" + title_suffix)
            st.caption("Lower values (lighter colors) indicate faster conversion")

            speed_pivot = converted_with_days.groupby(["expert_name", "assessment_month"])["days_to_interview"].mean().round(1)
            speed_wide = speed_pivot.unstack(fill_value=None)

            if not speed_wide.empty and speed_wide.shape[0] > 0:
                fig_speed = px.imshow(
                    speed_wide, text_auto=True, aspect="auto",
                    color_continuous_scale="RdYlGn_r",
                    title="Expert x Month: Avg Days to Interview (Lower = Faster)" + title_suffix,
                    labels=dict(x="Month", y="Expert", color="Avg Days"),
                )
                fig_speed.update_layout(height=max(450, len(speed_wide) * 30))
                st.plotly_chart(fig_speed, use_container_width=True)

    # ── Fuzzy Match Quality Report ───────────────────────────────
    matched = conv_df[conv_df["converted"] == 1].copy()
    if not matched.empty and "match_score" in matched.columns and "company_name" in matched.columns:
        fuzzy_matches = matched[matched["match_score"].notna() & (matched["match_score"] < 1.0)]
        if not fuzzy_matches.empty:
            with st.expander("Fuzzy Company Match Report" + title_suffix):
                st.caption("These conversions were matched using fuzzy company name matching. Review for accuracy.")
                display_fm = fuzzy_matches[["candidate_name", "company_name", "matched_company",
                                            "match_score", "days_to_interview"]].drop_duplicates()
                display_fm.columns = ["Candidate", "Assessment Company", "Matched Interview Company",
                                      "Match Score", "Days to Interview"]
                st.dataframe(display_fm.sort_values("Match Score"),
                             use_container_width=True, hide_index=True)

    # ── Full Conversion Data ─────────────────────────────────────
    with st.expander("Full Assessment Conversion Data" + title_suffix):
        display_cols = [c for c in ["date", "candidate_name", "expert_name", "company_name",
                                    "task_status", "converted", "matched_interview_date",
                                    "matched_company", "match_score", "days_to_interview",
                                    "interview_round", "interview_expert", "interview_status",
                                    "assessment_month"]
                        if c in conv_df.columns]
        st.dataframe(conv_df[display_cols].sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    title_col, dl_col = st.columns([4, 1])
    with title_col:
        st.title("Vizva Interview Dashboard")

    raw = fetch_all_data()
    if raw.empty:
        st.error("No data returned from API.")
        st.stop()

    raw = normalize(raw)
    raw = filter_current_year(raw)
    if raw.empty:
        st.error("No data found for current year.")
        st.stop()
    # ── AUTO-UPDATE: Self-expert rows ─────────────────────────────
    if "expert_name" in raw.columns:
        self_mask = raw["expert_name"].str.strip().str.lower() == "self"
        if "task_status" in raw.columns:
            raw.loc[self_mask, "task_status"] = "completed"
        # Update whichever feedback column(s) exist
        feedback_text = "This Interview is given by Candidate himself and so no support was required."
        for col in ["feedback", "expert_feedback", "client_feedback"]:
            if col in raw.columns:
                raw.loc[self_mask, col] = feedback_text

    all_case_df = raw.copy()
    active_expert_df = filter_active_experts(raw)
    # All Interview Support data (no active expert filter) for blockage lookback
    all_interview_support_df = all_case_df[all_case_df["support_name"].str.lower() == "interview support"].copy()
    if "start_time" in all_interview_support_df.columns:
        all_interview_support_df = add_start_time_columns(all_interview_support_df)


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

    # ── Sidebar: Overall sentiment gauge for selected support type
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

    # ── Sidebar: Clash indicator for Interview Support ───────────
    if selected_support == "Interview Support" and "_parsed_start" in support_df.columns:
        clash_groups_check, _ = detect_expert_clashes(support_df)
        if not clash_groups_check.empty:
            st.sidebar.markdown("---")
            st.sidebar.metric("⚠️ Total Clash Groups", len(clash_groups_check))
            st.sidebar.metric("Experts with Clashes", clash_groups_check["expert_name"].nunique())

        # Blockage sidebar indicator
        blockage_check = detect_blockages(support_df, all_experts_df=all_interview_support_df)

        if not blockage_check.empty:
            st.sidebar.metric("🚨 Total Blockages", len(blockage_check))
            st.sidebar.metric("Days with Blockage", blockage_check["date"].dt.date.nunique())

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
                                            "Schedule View"], label_visibility="collapsed")

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

        # ── TODAY'S SENTIMENT KPI ────────────────────────────────
        if not today_df.empty:
            today_stats = get_sentiment_stats(today_df)
            if today_stats:
                st.markdown("---")
                render_sentiment_kpi(today_stats, title="Today's Feedback Sentiment")

        # ── TODAY'S START TIME INSIGHTS ──────────────────────────
        if selected_support == "Interview Support" and not today_df.empty and "start_hour" in today_df.columns:
            st.markdown("---")
            render_start_time_insights(today_df, title_suffix=" - " + str(today))

        # ── TODAY'S CLASH DETECTION ──────────────────────────────
        if selected_support == "Interview Support" and not today_df.empty and "_parsed_start" in today_df.columns:
            st.markdown("---")
            st.subheader("Scheduling Clashes - " + str(today))
            render_today_clash_summary(today_df)

        # ── TODAY'S BLOCKAGE DETECTION ───────────────────────────
        if selected_support == "Interview Support" and not today_df.empty and "_parsed_start" in today_df.columns:
            st.markdown("---")
            st.subheader("Blockage Indicator - " + str(today))
            render_today_blockage(today_df, all_experts_df=all_interview_support_df)

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

            # ── SENTIMENT DONUT + HISTOGRAM — TODAY ──────────────
            st.markdown("---")
            render_sentiment_section(today_df, section_title="Sentiment Analysis - " + str(today))

            # ── FEEDBACK WORD CLOUD — TODAY ──────────────────────
            st.markdown("---")
            feedback_texts = extract_feedback_texts(today_df)
            render_wordcloud_section(
                feedback_texts,
                section_title="Feedback Word Cloud - " + str(today),
            )

            with st.expander("Raw Data (includes Sentiment Score)"):
                st.dataframe(today_df, use_container_width=True, height=400)

        # ── ABOUT TO MOVE TO MARKET ──────────────────────────────
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
                st.columns(6)[0].empty()
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

        # ── START TIME INSIGHTS — OVERALL ────────────────────────
        if selected_support == "Interview Support" and "start_hour" in support_df.columns:
            st.markdown("---")
            render_start_time_insights(support_df, title_suffix=" - All Months")
            st.markdown("---")
            render_monthly_start_time_trend(support_df, title_suffix=" - " + support_label)

        # ── CLASH DETECTION — OVERALL ────────────────────────────
        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns:
            st.markdown("---")
            render_clash_summary(support_df, title_suffix=" - All Months")

        # ── BLOCKAGE ANALYSIS — OVERALL ──────────────────────────
        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns:
            st.markdown("---")
            render_blockage_summary(support_df, title_suffix=" - All Months", all_experts_df=all_interview_support_df)

        # ── MONTHLY SENTIMENT TRENDS ─────────────────────────────
        st.markdown("---")
        st.subheader("Monthly Sentiment Trends")
        sent_monthly = monthly_sentiment_trend(support_df)
        if not sent_monthly.empty:
            col_t, col_s = st.columns(2)
            with col_t:
                fig = render_sentiment_trend_chart(sent_monthly, "Monthly Avg Sentiment")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col_s:
                fig = render_sentiment_stacked_bar(sent_monthly, "Monthly Sentiment Breakdown")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with st.expander("Sentiment Trend Data"):
                st.dataframe(sent_monthly, use_container_width=True)
        else:
            st.info("No sentiment data available for trend analysis.")

        # ── MONTHLY (Interview Support Only) ─────────────────────
        if selected_support == "Interview Support":
            st.markdown("---")
            st.subheader("Round-wise Monthly Breakdown")
            if "round_name" in support_df.columns:
                support_df_copy = support_df.copy()
                support_df_copy["month"] = support_df_copy["date"].dt.to_period("M").astype(str)
                months_available = sorted(support_df_copy["month"].unique(), reverse=True)
                current_m = date.today().strftime("%Y-%m")
                default_idx = months_available.index(current_m) if current_m in months_available else 0
                sel_month = st.selectbox("Select Month", months_available, index=default_idx, key="round_month")

                month_data = support_df_copy[support_df_copy["month"] == sel_month]
                if not month_data.empty:
                    round_charts(month_data, " - " + sel_month)
                    if "round_name" in month_data.columns:
                        month_round_df = month_data.groupby(["round_name", "task_status"]).size().unstack(fill_value=0)
                        st.dataframe(month_round_df, use_container_width=True)

            st.markdown("---")
            st.subheader("Expert-wise Monthly Breakdown")
            month_exp = expert_monthly(support_df)
            if not month_exp.empty:
                exp_months = sorted(month_exp["month"].unique(), reverse=True)
                current_m = date.today().strftime("%Y-%m")
                default_idx2 = exp_months.index(current_m) if current_m in exp_months else 0
                sel_m2 = st.selectbox("Select Month", exp_months, index=default_idx2, key="exp_month")

                me = month_exp[month_exp["month"] == sel_m2].sort_values("total", ascending=False)
                st.dataframe(me, use_container_width=True)

        # ── COMPLETED ROUNDS ANALYSIS & FINAL BY WHOM ────────────
        if selected_support == "Interview Support":
            st.markdown("---")
            st.subheader("Completed Rounds Analysis & Final Given By Whom")
            st.caption("Shows only completed interviews — which rounds were completed, who gave the Final round, and their sentiment scores.")

            if "round_name" in support_df.columns and "date" in support_df.columns:
                completed_iv = support_df[support_df["task_status"] == "completed"].copy()
                if completed_iv.empty:
                    st.info("No completed interviews found.")
                else:
                    completed_iv["month"] = completed_iv["date"].dt.to_period("M").astype(str)
                    months_avail_cr = sorted(completed_iv["month"].unique(), reverse=True)
                    sel_cr_month = st.selectbox("Select Month", months_avail_cr, index=0, key="completed_rounds_month")
                    cr_month_data = completed_iv[completed_iv["month"] == sel_cr_month]

                    if cr_month_data.empty:
                        st.info("No completed interviews in " + sel_cr_month)
                    else:
                        cr_cols = st.columns(3)
                        cr_cols[0].metric("Completed Interviews", len(cr_month_data))
                        cr_cols[1].metric("Unique Rounds", cr_month_data["round_name"].nunique())
                        cr_cols[2].metric("Unique Experts", cr_month_data["expert_name"].nunique() if "expert_name" in cr_month_data.columns else 0)

                        cr_round_counts = cr_month_data["round_name"].value_counts()
                        cr_c1, cr_c2 = st.columns(2)
                        with cr_c1:
                            fig_cr = go.Figure(go.Pie(labels=cr_round_counts.index, values=cr_round_counts.values,
                                                      hole=.4, textinfo="label+value+percent"))
                            fig_cr.update_layout(title="Completed Rounds - " + sel_cr_month, height=420)
                            st.plotly_chart(fig_cr, use_container_width=True)
                        with cr_c2:
                            if "expert_name" in cr_month_data.columns:
                                re_pivot = cr_month_data.groupby(["round_name", "expert_name"]).size().reset_index(name="count")
                                top_experts_cr = re_pivot.groupby("expert_name")["count"].sum().nlargest(10).index
                                re_pivot_top = re_pivot[re_pivot["expert_name"].isin(top_experts_cr)]
                                fig_re = go.Figure()
                                for exp in sorted(top_experts_cr):
                                    exp_data = re_pivot_top[re_pivot_top["expert_name"] == exp]
                                    fig_re.add_trace(go.Bar(x=exp_data["round_name"], y=exp_data["count"],
                                                            name=exp, text=exp_data["count"], textposition="inside"))
                                fig_re.update_layout(barmode="stack", title="Rounds by Expert (Top 10) - " + sel_cr_month,
                                                     height=420, legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
                                st.plotly_chart(fig_re, use_container_width=True)

                        st.markdown("---")
                        st.subheader("Final Round — Given By Whom (" + sel_cr_month + ")")
                        final_mask = cr_month_data["round_name"].str.lower().str.contains("final", na=False)
                        final_data = cr_month_data[final_mask]

                        if final_data.empty:
                            st.info("No 'Final' rounds found in completed interviews for " + sel_cr_month)
                        else:
                            fc = st.columns(3)
                            fc[0].metric("Final Rounds Completed", len(final_data))
                            fc[1].metric("Experts Who Gave Final", final_data["expert_name"].nunique() if "expert_name" in final_data.columns else 0)
                            fc[2].metric("Candidates in Final", final_data["candidate_name"].nunique() if "candidate_name" in final_data.columns else 0)

                            if "expert_name" in final_data.columns:
                                final_by_expert = final_data["expert_name"].value_counts()
                                fig_final = go.Figure(go.Bar(
                                    y=final_by_expert.index, x=final_by_expert.values,
                                    orientation="h", marker_color="#8e44ad",
                                    text=final_by_expert.values, textposition="outside"
                                ))
                                fig_final.update_layout(title="Final Round — Expert Breakdown",
                                                        height=max(400, len(final_by_expert) * 35),
                                                        yaxis=dict(autorange="reversed"),
                                                        xaxis_title="Final Rounds Given")
                                st.plotly_chart(fig_final, use_container_width=True)

                            final_stats = get_sentiment_stats(final_data)
                            if final_stats:
                                st.markdown("---")
                                render_sentiment_kpi(final_stats, title="Sentiment — Final Round (" + sel_cr_month + ")")
                                fcol1, fcol2 = st.columns(2)
                                with fcol1:
                                    fig = render_sentiment_donut(final_stats, "Final Round Sentiment Split")
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                with fcol2:
                                    fig = render_sentiment_histogram(final_data, "Final Round Score Distribution")
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)

                            with st.expander("Final Round Details - " + sel_cr_month):
                                display_cols = [c for c in ["date", "candidate_name", "expert_name", "round_name",
                                                            "company_name", "sentiment_score", "sentiment_label", "feedback"]
                                                if c in final_data.columns]
                                st.dataframe(final_data[display_cols] if display_cols else final_data,
                                             use_container_width=True, hide_index=True)
                                                # ── TECHNICAL / CODING ROUND — GIVEN BY WHOM ─────
                        st.markdown("---")
                        st.subheader("Technical / Coding Round — Given By Whom (" + sel_cr_month + ")")
                        tech_mask = cr_month_data["round_name"].str.lower().str.contains("technical|coding", na=False)
                        tech_data = cr_month_data[tech_mask]

                        if tech_data.empty:
                            st.info("No 'Technical / Coding' rounds found in completed interviews for " + sel_cr_month)
                        else:
                            tc_cols = st.columns(3)
                            tc_cols[0].metric("Technical/Coding Rounds Completed", len(tech_data))
                            tc_cols[1].metric("Experts Who Gave Technical/Coding",
                                              tech_data["expert_name"].nunique() if "expert_name" in tech_data.columns else 0)
                            tc_cols[2].metric("Candidates in Technical/Coding",
                                              tech_data["candidate_name"].nunique() if "candidate_name" in tech_data.columns else 0)

                            if "expert_name" in tech_data.columns:
                                tech_by_expert = tech_data["expert_name"].value_counts()
                                fig_tech = go.Figure(go.Bar(
                                    y=tech_by_expert.index, x=tech_by_expert.values,
                                    orientation="h", marker_color="#2980b9",
                                    text=tech_by_expert.values, textposition="outside"
                                ))
                                fig_tech.update_layout(title="Technical/Coding Round — Expert Breakdown",
                                                       height=max(400, len(tech_by_expert) * 35),
                                                       yaxis=dict(autorange="reversed"),
                                                       xaxis_title="Technical/Coding Rounds Given")
                                st.plotly_chart(fig_tech, use_container_width=True)

                            tech_stats = get_sentiment_stats(tech_data)
                            if tech_stats:
                                st.markdown("---")
                                render_sentiment_kpi(tech_stats,
                                                     title="Sentiment — Technical/Coding Round (" + sel_cr_month + ")")
                                tcol1, tcol2 = st.columns(2)
                                with tcol1:
                                    fig = render_sentiment_donut(tech_stats, "Technical/Coding Sentiment Split")
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                with tcol2:
                                    fig = render_sentiment_histogram(tech_data, "Technical/Coding Score Distribution")
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)

                            with st.expander("Technical/Coding Round Details - " + sel_cr_month):
                                display_cols = [c for c in ["date", "candidate_name", "expert_name", "round_name",
                                                            "company_name", "sentiment_score", "sentiment_label", "feedback"]
                                                if c in tech_data.columns]
                                st.dataframe(tech_data[display_cols] if display_cols else tech_data,
                                             use_container_width=True, hide_index=True)

                        # ── FINALS + TECHNICAL/CODING MONTHLY TOTALS ─────
                        st.markdown("---")
                        st.subheader("Finals + Technical/Coding — Monthly Totals")
                        st.caption("Month-wise count of Final and Technical/Coding rounds from completed interviews only")

                        ft_mask = completed_iv["round_name"].str.lower().str.contains("final|technical|coding", na=False)
                        ft_all = completed_iv[ft_mask].copy()

                        if ft_all.empty:
                            st.info("No Final or Technical/Coding rounds found in completed interviews.")
                        else:
                            ft_all["round_type"] = ft_all["round_name"].apply(
                                lambda r: "Final" if "final" in str(r).lower()
                                else "Technical/Coding" if any(k in str(r).lower() for k in ["technical", "coding"])
                                else "Other"
                            )
                            ft_monthly = ft_all.groupby(["month", "round_type"]).size().unstack(fill_value=0).reset_index()

                            fig_ft = go.Figure()
                            if "Final" in ft_monthly.columns:
                                fig_ft.add_trace(go.Bar(x=ft_monthly["month"], y=ft_monthly["Final"],
                                                        name="Final", marker_color="#8e44ad",
                                                        text=ft_monthly["Final"], textposition="inside"))
                            if "Technical/Coding" in ft_monthly.columns:
                                fig_ft.add_trace(go.Bar(x=ft_monthly["month"], y=ft_monthly["Technical/Coding"],
                                                        name="Technical/Coding", marker_color="#2980b9",
                                                        text=ft_monthly["Technical/Coding"], textposition="inside"))
                            fig_ft.update_layout(barmode="stack",
                                                 title="Monthly Finals + Technical/Coding Rounds (Completed Only)",
                                                 height=420, yaxis_title="Rounds",
                                                 legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
                            st.plotly_chart(fig_ft, use_container_width=True)

                            with st.expander("Monthly Finals + Technical/Coding Data"):
                                st.dataframe(ft_monthly, use_container_width=True, hide_index=True)

                        # ── CONVERSION KPI: INTERVIEWS PER FINAL+TECHNICAL ──
                        st.markdown("---")
                        st.subheader("Interview Conversion Rate — " + sel_cr_month)
                        st.caption(
                            "Total Completed Interviews ÷ (Finals + Technical/Coding) = "
                            "how many interviews it takes on average to reach a Final or Technical/Coding round. "
                            "Lower is better."
                        )

                        total_completed_month = len(cr_month_data)
                        final_count_month = len(cr_month_data[cr_month_data["round_name"].str.lower().str.contains("final", na=False)])
                        tech_count_month = len(cr_month_data[cr_month_data["round_name"].str.lower().str.contains("technical|coding", na=False)])
                        ft_total_month = final_count_month + tech_count_month

                        conv_cols = st.columns(5)
                        conv_cols[0].metric("Total Completed", total_completed_month)
                        conv_cols[1].metric("Final Rounds", final_count_month)
                        conv_cols[2].metric("Technical/Coding Rounds", tech_count_month)
                        conv_cols[3].metric("Finals + Technical/Coding", ft_total_month)
                        if ft_total_month > 0:
                            ratio = round(total_completed_month / ft_total_month, 2)
                            conv_cols[4].metric("Interviews per Final+Tech", f"{ratio:.2f}",
                                                delta="Lower is better", delta_color="inverse")
                        else:
                            conv_cols[4].metric("Interviews per Final+Tech", "N/A")

                        # ── MONTHLY CONVERSION TREND ─────────────────
                        st.markdown("---")
                        st.subheader("Monthly Conversion Trend")
                        st.caption("Completed Interviews ÷ (Finals + Technical/Coding) per month — lower means faster conversion")

                        conv_rows = []
                        for m in sorted(completed_iv["month"].unique()):
                            m_data = completed_iv[completed_iv["month"] == m]
                            m_total = len(m_data)
                            m_final = len(m_data[m_data["round_name"].str.lower().str.contains("final", na=False)])
                            m_tech = len(m_data[m_data["round_name"].str.lower().str.contains("technical|coding", na=False)])
                            m_ft = m_final + m_tech
                            m_ratio = round(m_total / m_ft, 2) if m_ft > 0 else None
                            conv_rows.append({
                                "Month": m,
                                "Completed": m_total,
                                "Finals": m_final,
                                "Technical/Coding": m_tech,
                                "Finals+Tech": m_ft,
                                "Ratio": m_ratio,
                            })
                        conv_df = pd.DataFrame(conv_rows)

                        valid_conv = conv_df.dropna(subset=["Ratio"])
                        if not valid_conv.empty:
                            fig_conv = go.Figure()
                            fig_conv.add_trace(go.Scatter(
                                x=valid_conv["Month"], y=valid_conv["Ratio"],
                                mode="lines+markers+text",
                                text=valid_conv["Ratio"].apply(lambda v: f"{v:.2f}"),
                                textposition="top center",
                                line=dict(color="#e67e22", width=3),
                                marker=dict(size=10),
                                name="Interviews per Final+Tech",
                            ))
                            fig_conv.add_trace(go.Bar(
                                x=valid_conv["Month"], y=valid_conv["Finals+Tech"],
                                name="Finals+Tech Count", marker_color="#8e44ad", opacity=0.3,
                                text=valid_conv["Finals+Tech"], textposition="outside",
                                yaxis="y2",
                            ))
                            fig_conv.update_layout(
                                title="Monthly Conversion: Interviews per Final+Technical Round",
                                height=450,
                                yaxis=dict(title="Ratio (lower = better)", side="left"),
                                yaxis2=dict(title="Finals+Tech Count", side="right", overlaying="y"),
                                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                            )
                            st.plotly_chart(fig_conv, use_container_width=True)

                        with st.expander("Monthly Conversion Data"):
                            st.dataframe(conv_df, use_container_width=True, hide_index=True)

                        # ── EXPERT-WISE CONVERSION KPI ───────────────
                        st.markdown("---")
                        st.subheader("Expert-wise Conversion Rate — " + sel_cr_month)
                        st.caption(
                            "Each expert's Completed Interviews ÷ (Finals + Technical/Coding). "
                            "Lower ratio = expert reaches advanced rounds faster."
                        )

                        if "expert_name" in cr_month_data.columns:
                            expert_conv_rows = []
                            for expert in sorted(cr_month_data["expert_name"].unique()):
                                e_data = cr_month_data[cr_month_data["expert_name"] == expert]
                                e_total = len(e_data)
                                e_final = len(e_data[e_data["round_name"].str.lower().str.contains("final", na=False)])
                                e_tech = len(e_data[e_data["round_name"].str.lower().str.contains("technical|coding", na=False)])
                                e_ft = e_final + e_tech
                                e_ratio = round(e_total / e_ft, 2) if e_ft > 0 else None
                                expert_conv_rows.append({
                                    "Expert": expert,
                                    "Completed": e_total,
                                    "Finals": e_final,
                                    "Technical/Coding": e_tech,
                                    "Finals+Tech": e_ft,
                                    "Ratio": e_ratio,
                                })
                            expert_conv_df = pd.DataFrame(expert_conv_rows)

                            # Show KPI cards for top/bottom performers
                            valid_expert_conv = expert_conv_df.dropna(subset=["Ratio"])
                            if not valid_expert_conv.empty:
                                best = valid_expert_conv.loc[valid_expert_conv["Ratio"].idxmin()]
                                worst = valid_expert_conv.loc[valid_expert_conv["Ratio"].idxmax()]
                                avg_ratio = round(valid_expert_conv["Ratio"].mean(), 2)

                                perf_cols = st.columns(3)
                                perf_cols[0].metric("Best Converter", str(best["Expert"]),
                                                    delta=f"{best['Ratio']:.2f} ratio", delta_color="inverse")
                                perf_cols[1].metric("Needs Improvement", str(worst["Expert"]),
                                                    delta=f"{worst['Ratio']:.2f} ratio", delta_color="normal")
                                perf_cols[2].metric("Team Average", f"{avg_ratio:.2f}")

                            # Bar chart of expert conversion ratios
                            valid_sorted = valid_expert_conv.sort_values("Ratio", ascending=True) if not valid_expert_conv.empty else pd.DataFrame()
                            if not valid_sorted.empty:
                                bar_colors = ["#2ecc71" if r <= avg_ratio else "#e74c3c"
                                              for r in valid_sorted["Ratio"]] if not valid_expert_conv.empty else []
                                fig_ec = go.Figure(go.Bar(
                                    y=valid_sorted["Expert"], x=valid_sorted["Ratio"],
                                    orientation="h", marker_color=bar_colors,
                                    text=valid_sorted["Ratio"].apply(lambda v: f"{v:.2f}"),
                                    textposition="outside",
                                ))
                                fig_ec.add_vline(x=avg_ratio, line_dash="dash", line_color="#f39c12",
                                                 annotation_text=f"Avg: {avg_ratio:.2f}")
                                fig_ec.update_layout(
                                    title="Expert Conversion Rate — Interviews per Final+Technical (" + sel_cr_month + ")",
                                    height=max(450, len(valid_sorted) * 35),
                                    xaxis_title="Ratio (lower = better)",
                                    yaxis=dict(autorange="reversed"),
                                )
                                st.plotly_chart(fig_ec, use_container_width=True)

                            with st.expander("Expert Conversion Data — " + sel_cr_month):
                                st.dataframe(expert_conv_df.sort_values("Ratio", ascending=True, na_position="last"),
                                             use_container_width=True, hide_index=True)


                        st.markdown("---")
                        st.subheader("Sentiment — All Completed Interviews (" + sel_cr_month + ")")
                        render_sentiment_section(cr_month_data,
                                                 section_title="Completed Interview Sentiment - " + sel_cr_month)
        # ── ASSESSMENT CONVERSION ANALYTICS ──────────────────────
        if selected_support == "Assessment Support":
            st.markdown("---")
            st.subheader("Assessment to Interview Conversion Analytics")
            st.caption("Tracks which assessments led to interviews within 60 days using fuzzy company name matching")

            conv_df = compute_assessment_conversions(all_case_df)
            if not conv_df.empty:
                current_m = date.today().strftime("%Y-%m")
                conv_months = sorted(conv_df["assessment_month"].unique(), reverse=True)
                default_conv_idx = conv_months.index(current_m) if current_m in conv_months else 0
                sel_conv_month = st.selectbox("Select Month", conv_months, index=default_conv_idx,
                                              key="conv_month")
                conv_month_data = conv_df[conv_df["assessment_month"] == sel_conv_month]

                if conv_month_data.empty:
                    st.info("No assessments in " + sel_conv_month)
                else:
                    render_assessment_conversion_kpi(conv_month_data, " - " + sel_conv_month)
                    render_assessment_conversion_charts(conv_month_data, " - " + sel_conv_month)
            else:
                st.info("No assessment data available for conversion analysis.")
        
        # ── CANDIDATE-WISE MONTHLY COUNTS ────────────────────────
        st.markdown("---")
        st.subheader("Candidate-wise Monthly Counts (All Support Types)")
        cand_monthly = candidate_monthly_support(active_expert_df)
        if not cand_monthly.empty:
            cand_months = sorted(cand_monthly["month"].unique(), reverse=True)
            current_m = date.today().strftime("%Y-%m")
            default_idx3 = cand_months.index(current_m) if current_m in cand_months else 0
            sel_m3 = st.selectbox("Select Month", cand_months, index=default_idx3, key="cand_month")

            cm = cand_monthly[cand_monthly["month"] == sel_m3].sort_values("total", ascending=False)
            st.dataframe(cm, use_container_width=True)

    # ====== DAILY DRILL-DOWN ======
    elif view == "Daily Drill-Down":
        st.header("Daily Drill-Down - " + support_label)
        if "date" not in support_df.columns:
            st.warning("No date column available.")
            st.stop()
        min_d = support_df["date"].min().date()
        max_d = support_df["date"].max().date()
        ca, cb = st.columns(2)
        start = ca.date_input("From", value=max_d - timedelta(days=30),
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

            # ── DAILY STACKED BAR (kept) ─────────────────────────
            dd_plot = dd.copy()
            dd_plot["day"] = pd.to_datetime(dd_plot["day"])
            st.plotly_chart(stacked_bar(dd_plot, x="day", title="Daily Stacked View"), use_container_width=True)

            # ── PERIOD SENTIMENT DONUT + HISTOGRAM ───────────────
            if period_stats:
                st.markdown("---")
                st.subheader("Sentiment Analysis (" + str(start) + " to " + str(end) + ")")
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    fig = render_sentiment_donut(period_stats,
                                                "Sentiment Split (" + str(start) + " to " + str(end) + ")")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with s_c2:
                    fig = render_sentiment_histogram(period,
                                                    "Score Distribution (" + str(start) + " to " + str(end) + ")")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

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
                    marker=dict(size=8, color=d_colors),
                ))
                fig_ds.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_ds.update_layout(title="Daily Avg Sentiment", height=400,
                                     yaxis_title="Avg Sentiment (%)")
                st.plotly_chart(fig_ds, use_container_width=True)

            # ── EXPERT / COMPANY / CANDIDATE BARS (period) ───────
            st.markdown("---")
            st.subheader("Period Breakdown (" + str(start) + " to " + str(end) + ")")
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                fig = h_bar_by_task(period, "company_name", 10,
                                    "Top Companies (" + str(start) + " to " + str(end) + ")")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with p_c2:
                fig = h_bar_by_task(period, "expert_name", 10,
                                    "Top Experts (" + str(start) + " to " + str(end) + ")")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            p_c3, p_c4 = st.columns(2)
            with p_c3:
                fig = h_bar_by_task(period, "candidate_name", 10,
                                    "Top Candidates (" + str(start) + " to " + str(end) + ")")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with p_c4:
                st.plotly_chart(donut(kd, "Task Split (" + str(start) + " to " + str(end) + ")"),
                                use_container_width=True)

            # ── ROUND INSIGHTS (Interview Support, completed only) ─
            if selected_support == "Interview Support" and "round_name" in period.columns:
                st.markdown("---")
                st.subheader("Round Distribution — Completed Only (" + str(start) + " to " + str(end) + ")")
                completed_period = period[period["task_status"] == "completed"] if "task_status" in period.columns else period

                if completed_period.empty:
                    st.info("No completed interviews in this period for round analysis.")
                else:
                    cr_counts = completed_period["round_name"].value_counts()
                    rnd_c1, rnd_c2 = st.columns(2)
                    with rnd_c1:
                        fig_rnd = go.Figure(go.Pie(labels=cr_counts.index, values=cr_counts.values,
                                                   hole=.4, textinfo="label+value+percent"))
                        fig_rnd.update_layout(title="Round Split (Completed Only)", height=420)
                        st.plotly_chart(fig_rnd, use_container_width=True)
                    with rnd_c2:
                        if "expert_name" in completed_period.columns:
                            re_pivot = completed_period.groupby(["round_name", "expert_name"]).size().reset_index(name="count")
                            top_exp = re_pivot.groupby("expert_name")["count"].sum().nlargest(10).index
                            re_pivot_top = re_pivot[re_pivot["expert_name"].isin(top_exp)]
                            fig_re = go.Figure()
                            for exp in sorted(top_exp):
                                exp_data = re_pivot_top[re_pivot_top["expert_name"] == exp]
                                fig_re.add_trace(go.Bar(x=exp_data["round_name"], y=exp_data["count"],
                                                        name=exp, text=exp_data["count"], textposition="inside"))
                            fig_re.update_layout(barmode="stack", title="Rounds by Expert (Top 10)",
                                                 height=420, legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
                            st.plotly_chart(fig_re, use_container_width=True)

                    if "task_status" in period.columns:
                        rt = period.groupby(["round_name", "task_status"]).size().unstack(fill_value=0)
                        fig_rt = px.imshow(rt, text_auto=True, aspect="auto",
                                           color_continuous_scale="Oranges", title="Round × Task Heatmap (All Statuses)")
                        fig_rt.update_layout(height=400)
                        st.plotly_chart(fig_rt, use_container_width=True)

            # ── COMPLETED ROUNDS & FINAL GIVEN BY WHOM (period) ──
            if selected_support == "Interview Support" and "round_name" in period.columns and "date" in period.columns:
                completed_p = period[period["task_status"] == "completed"].copy() if "task_status" in period.columns else period.copy()
                if not completed_p.empty:
                    st.markdown("---")
                    st.subheader("Final Round — Given By Whom (" + str(start) + " to " + str(end) + ")")
                    final_mask = completed_p["round_name"].str.lower().str.contains("final", na=False)
                    final_data = completed_p[final_mask]

                    if final_data.empty:
                        st.info("No 'Final' rounds found in completed interviews for this period.")
                    else:
                        fc = st.columns(3)
                        fc[0].metric("Final Rounds Completed", len(final_data))
                        fc[1].metric("Experts Who Gave Final",
                                     final_data["expert_name"].nunique() if "expert_name" in final_data.columns else 0)
                        fc[2].metric("Candidates in Final",
                                     final_data["candidate_name"].nunique() if "candidate_name" in final_data.columns else 0)

                        if "expert_name" in final_data.columns:
                            final_by_expert = final_data["expert_name"].value_counts()
                            fig_final = go.Figure(go.Bar(
                                y=final_by_expert.index, x=final_by_expert.values,
                                orientation="h", marker_color="#8e44ad",
                                text=final_by_expert.values, textposition="outside"
                            ))
                            fig_final.update_layout(title="Final Round — Expert Breakdown",
                                                    height=max(400, len(final_by_expert) * 35),
                                                    yaxis=dict(autorange="reversed"),
                                                    xaxis_title="Final Rounds Given")
                            st.plotly_chart(fig_final, use_container_width=True)

                        final_stats = get_sentiment_stats(final_data)
                        if final_stats:
                            st.markdown("---")
                            render_sentiment_kpi(final_stats,
                                                 title="Sentiment — Final Round (" + str(start) + " to " + str(end) + ")")
                            fcol1, fcol2 = st.columns(2)
                            with fcol1:
                                fig = render_sentiment_donut(final_stats, "Final Round Sentiment Split")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            with fcol2:
                                fig = render_sentiment_histogram(final_data, "Final Round Score Distribution")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Final Round Details - " + str(start) + " to " + str(end)):
                            display_cols = [c for c in ["date", "candidate_name", "expert_name", "round_name",
                                                        "company_name", "sentiment_score", "sentiment_label", "feedback"]
                                            if c in final_data.columns]
                            st.dataframe(final_data[display_cols] if display_cols else final_data,
                                         use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.subheader("Sentiment — All Completed Interviews (" + str(start) + " to " + str(end) + ")")
                    render_sentiment_section(completed_p,
                                            section_title="Completed Interview Sentiment - " + str(start) + " to " + str(end))

            # ── START TIME INSIGHTS (period) ─────────────────────
            if selected_support == "Interview Support" and "start_hour" in period.columns:
                st.markdown("---")
                render_start_time_insights(period, title_suffix=" - " + str(start) + " to " + str(end))

            # ── CLASH DETECTION (period) ─────────────────────────
            if selected_support == "Interview Support" and "_parsed_start" in period.columns:
                st.markdown("---")
                st.subheader("Scheduling Clashes (" + str(start) + " to " + str(end) + ")")
                render_today_clash_summary(period)
            # ── ASSESSMENT CONVERSION (period) ───────────────────
            if selected_support == "Assessment Support":
                st.markdown("---")
                st.subheader("Assessment Conversion (" + str(start) + " to " + str(end) + ")")
                conv_df_all = compute_assessment_conversions(all_case_df)
                if not conv_df_all.empty:
                    conv_period = conv_df_all[
                        (conv_df_all["date"].dt.date >= start) &
                        (conv_df_all["date"].dt.date <= end)
                    ]
                    if not conv_period.empty:
                        render_assessment_conversion_kpi(conv_period,
                                                         " - " + str(start) + " to " + str(end))
                        render_assessment_conversion_charts(conv_period,
                                                            " - " + str(start) + " to " + str(end))
                    else:
                        st.info("No assessments in this period.")

        

            # ── TECHNOLOGY BREAKDOWN (period) ────────────────────
            if "candidate_technology" in period.columns:
                st.markdown("---")
                fig = h_bar_by_task(period, "candidate_technology", 15,
                                    "Top Technologies (" + str(start) + " to " + str(end) + ")")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # ── SINGLE DAY DRILL-DOWN ────────────────────────────────
        st.markdown("---")
        st.subheader("Single Day Details")
        single = st.date_input("Select a day", value=max_d, min_value=min_d, max_value=max_d,
                                key="single_day")
        sdf = support_df[support_df["date"].dt.date == single]
        if not sdf.empty:
            tc = sdf["task_status"].value_counts()
            skd = {s: int(tc.get(s, 0)) for s in TASK_ORDER}
            skd["candidates"] = sdf["candidate_name"].nunique()
            kpi_row(skd)

            single_stats = get_sentiment_stats(sdf)
            if single_stats:
                st.markdown("---")
                render_sentiment_kpi(single_stats, title="Sentiment - " + str(single))

            if selected_support == "Interview Support" and "start_hour" in sdf.columns:
                st.markdown("---")
                render_start_time_insights(sdf, title_suffix=" - " + str(single))

            if selected_support == "Interview Support" and "_parsed_start" in sdf.columns:
                st.markdown("---")
                st.subheader("Scheduling Clashes - " + str(single))
                render_today_clash_summary(sdf)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(donut(skd, "Task Split - " + str(single)), use_container_width=True)
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

        tab_names = ["Experts", "Companies", "Rounds", "Day of Week", "Cross-Support", "Candidates", "Technology"]
        if selected_support == "Assessment Support":
            tab_names.append("Conversion Analytics")

        if selected_support == "Interview Support" and "start_hour" in support_df.columns:
            tab_names.append("Start Time")
        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns:
            tab_names.append("Clash Detection")
        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns:
            tab_names.append("Blockage")
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
                completed_only = support_df[support_df["task_status"] == "completed"] if "task_status" in support_df.columns else support_df
                rc = completed_only["round_name"].value_counts()
                if rc.empty:
                    st.info("No completed interviews for round distribution.")
                else:
                    fig = go.Figure(go.Pie(labels=rc.index, values=rc.values, hole=.4,
                                           textinfo="label+value+percent"))
                    fig.update_layout(title="Round Distribution (Completed Only)", height=420)
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
                
            # ── CANDIDATE STRENGTH ANALYSIS (All Data) ──────────
            if selected_support == "Interview Support":
                render_top_candidates_analysis(support_df,
                                                title_suffix=" — All Data",
                                                min_interviews=5)
    

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
        

        if selected_support == "Interview Support" and "start_hour" in support_df.columns and "Start Time" in tab_names:
            with tabs[tab_names.index("Start Time")]:
                render_start_time_insights(support_df, title_suffix=" - All Data")
                st.markdown("---")
                render_monthly_start_time_trend(support_df, title_suffix="")

        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns and "Clash Detection" in tab_names:
            with tabs[tab_names.index("Clash Detection")]:
                render_clash_summary(support_df, title_suffix=" - All Data")

        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns and "Blockage" in tab_names:
            with tabs[tab_names.index("Blockage")]:
                render_blockage_summary(support_df, title_suffix=" - All Data", all_experts_df=all_interview_support_df)
 
        if selected_support == "Assessment Support" and "Conversion Analytics" in tab_names:
            with tabs[tab_names.index("Conversion Analytics")]:
                st.subheader("Assessment to Interview Conversion (All Data)")
                st.caption("Full-year conversion analysis with fuzzy company matching")
                conv_df_deep = compute_assessment_conversions(all_case_df)
                if not conv_df_deep.empty:
                    render_assessment_conversion_kpi(conv_df_deep, " - All Data")
                    render_assessment_conversion_charts(conv_df_deep, " - All Data")
                else:
                    st.info("No assessment data available for conversion analysis.")
    # ====== SCHEDULE VIEW ======
    elif view == "Schedule View":
        render_schedule_view(all_case_df, active_expert_df)

    st.sidebar.markdown("---")
    st.sidebar.caption("Vizva Dashboard v20.0 | API-powered | Active Experts Only | Start Time Analytics | Clash Detection | Blockage")

# ═══════════════════════════════════════════════════════════════════
# AUTHENTICATION LAYER
# ═══════════════════════════════════════════════════════════════════

def login():
    st.title("Vizva Secure Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            valid_user = st.secrets["VIZVA_USERNAME"]
            valid_pass = st.secrets["VIZVA_PASSWORD"]
        except KeyError:
            st.error("Login secrets (VIZVA_USERNAME / VIZVA_PASSWORD) not configured. "
         "Please add them to your Streamlit secrets.")

            return

        if username.strip() == valid_user.strip() and password == valid_pass:
            st.session_state["authenticated"] = True
            st.session_state["login_time"] = datetime.now()
            st.rerun()
        else:
            st.error("Invalid credentials")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    if "login_time" in st.session_state:
        delta = datetime.now() - st.session_state["login_time"]
        if delta.total_seconds() > 3600:
            st.session_state["authenticated"] = False
            st.rerun()
    main()
else:
    login()
