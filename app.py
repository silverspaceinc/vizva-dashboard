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
            clash_groups_count=("group_size", "size"),
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

    # ── Clash Time-of-Day Distribution ───────────────────────────
    st.markdown("---")
    st.subheader("Clash Time-of-Day Distribution" + title_suffix)
    st.caption("Which hours of the day do clashes most frequently occur?")

    clash_hours = (clash_groups["mean_start_minutes"] // 60).astype(int)
    hour_clash_counts = clash_hours.value_counts().sort_index()

    all_hours = list(range(0, 24))
    hour_labels_clash = [datetime(2000, 1, 1, h).strftime("%I %p").lstrip("0") for h in all_hours]
    counts_clash = [int(hour_clash_counts.get(h, 0)) for h in all_hours]
    peak_clash_h = int(hour_clash_counts.idxmax()) if not hour_clash_counts.empty else 0
    bar_colors_clash = ["#e74c3c" if h == peak_clash_h else "#f39c12" for h in all_hours]

    fig_hour = go.Figure(go.Bar(
        x=hour_labels_clash, y=counts_clash,
        marker_color=bar_colors_clash,
        text=counts_clash, textposition="outside",
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

    # ── Monthly summary table ────────────────────────────────────
    with st.expander("Monthly Clash Summary" + title_suffix):
        monthly_table = clash_groups.groupby("month").agg(
            clash_groups_count=("group_size", "size"),
            total_interviews=("group_size", "sum"),
            experts=("expert_name", "nunique"),
            mean_start=("mean_start_minutes", "mean"),
        ).reset_index()
        monthly_table["mean_start_label"] = monthly_table["mean_start"].apply(
            _mean_start_label_from_minutes
        )
        monthly_table.columns = ["Month", "Clash Groups", "Interviews in Clashes",
                                 "Experts Affected", "Mean Start (min)", "Mean Clash Start Time"]
        st.dataframe(monthly_table[["Month", "Clash Groups", "Interviews in Clashes",
                                     "Experts Affected", "Mean Clash Start Time"]],
                     use_container_width=True, hide_index=True)

    # ── Detailed clash groups ────────────────────────────────────
    with st.expander("Detailed Clash Groups" + title_suffix):
        display_groups = clash_groups.copy()
        display_groups["date"] = display_groups["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            display_groups[["expert_name", "date", "group_size", "interviews_str",
                            "mean_start_label", "month"]].rename(columns={
                "expert_name": "Expert",
                "date": "Date",
                "group_size": "Interviews",
                "interviews_str": "Start Times",
                "mean_start_label": "Mean Start Time",
                "month": "Month",
            }),
            use_container_width=True, hide_index=True,
        )

    # ── Detailed clash pairs ─────────────────────────────────────
    if not clash_pairs.empty:
        with st.expander("All Clash Pairs" + title_suffix):
            display_pairs = clash_pairs.copy()
            display_pairs["date"] = display_pairs["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                display_pairs[["expert_name", "date", "start_time_1",
                                "start_time_2", "time_diff_min", "month"]],
                use_container_width=True, hide_index=True,
            )


def render_today_clash_summary(df):
    """Render a compact clash summary for today's snapshot."""
    clash_groups, clash_pairs = detect_expert_clashes(df)

    if clash_groups.empty:
        st.success("No scheduling clashes today.")
        return

    total_groups = len(clash_groups)
    total_interviews = int(clash_groups["group_size"].sum())
    experts_affected = clash_groups["expert_name"].nunique()
    mean_min = clash_groups["mean_start_minutes"].mean()
    mean_label = _mean_start_label_from_minutes(mean_min)

    k = st.columns(4)
    k[0].metric("Clash Groups Today", total_groups)
    k[1].metric("Interviews in Clashes", total_interviews)
    k[2].metric("Experts Affected", experts_affected)
    k[3].metric("Mean Clash Start Time", mean_label)

    size_counts = clash_groups["group_size"].value_counts().sort_index()
    size_parts = [str(int(v)) + "x " + str(int(s)) + "-interview" for s, v in size_counts.items()]
    st.caption("Clash breakdown: " + ", ".join(size_parts))

    with st.expander("Today's Clash Details"):
        display_groups = clash_groups.copy()
        display_groups["date"] = display_groups["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            display_groups[["expert_name", "date", "group_size", "interviews_str",
                            "mean_start_label"]].rename(columns={
                "expert_name": "Expert",
                "date": "Date",
                "group_size": "Interviews",
                "interviews_str": "Start Times",
                "mean_start_label": "Mean Start Time",
            }),
            use_container_width=True, hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════
#  BLOCKAGE DETECTION
#  Rule: In any 30-minute bracket, ALL experts have at least one
#  interview AND at least one expert has a clash in that bracket.
#  This means zero capacity — no one is free and there's a conflict.
# ═══════════════════════════════════════════════════════════════════

def detect_blockages(df):
    """Detect Blockage events.

    A Blockage occurs in a 30-minute time bracket on a given day when:
      1) Every unique expert (for that day) has >=1 interview starting
         within that bracket.
      2) At least one expert has a clash (2+ interviews) in that bracket.

    Brackets slide in 30-min steps: 00:00-00:30, 00:30-01:00, ... 23:30-00:00

    Returns a DataFrame with columns:
        date, day_name, bracket_start_min, bracket_label,
        total_experts_today, experts_in_bracket, experts_with_clash,
        involved_experts (list), clash_experts (list),
        mean_start_minutes, mean_start_label, month
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

    for day, day_grp in valid.groupby("_day"):
        all_experts_today = set(day_grp["expert_name"].unique())
        n_experts = len(all_experts_today)
        if n_experts < 2:
            continue

        for bracket_start in range(0, 1440, 30):
            bracket_end = bracket_start + 30

            in_bracket = day_grp[
                (day_grp["_start_minutes"] >= bracket_start) &
                (day_grp["_start_minutes"] < bracket_end)
            ]

            if in_bracket.empty:
                continue

            experts_in_bracket = set(in_bracket["expert_name"].unique())

            if experts_in_bracket != all_experts_today:
                continue

            expert_counts = in_bracket.groupby("expert_name").size()
            clash_experts = set(expert_counts[expert_counts >= 2].index)

            if not clash_experts:
                continue

            bracket_h = bracket_start // 60
            bracket_m = bracket_start % 60
            bracket_label = datetime(2000, 1, 1, bracket_h, bracket_m).strftime("%I:%M %p")

            mean_min = in_bracket["_start_minutes"].mean()
            mean_h = int(mean_min // 60) % 24
            mean_m = int(mean_min % 60)
            mean_label = datetime(2000, 1, 1, mean_h, mean_m).strftime("%I:%M %p")

            blockage_rows.append({
                "date": pd.Timestamp(day),
                "day_name": pd.Timestamp(day).day_name(),
                "bracket_start_min": bracket_start,
                "bracket_label": bracket_label,
                "total_experts_today": n_experts,
                "experts_in_bracket": len(experts_in_bracket),
                "experts_with_clash": len(clash_experts),
                "involved_experts": sorted(experts_in_bracket),
                "clash_experts": sorted(clash_experts),
                "mean_start_minutes": round(mean_min, 1),
                "mean_start_label": mean_label,
            })

    if not blockage_rows:
        return empty

    blockage_df = pd.DataFrame(blockage_rows)
    blockage_df["month"] = blockage_df["date"].dt.to_period("M").astype(str)
    return blockage_df


def render_today_blockage(df):
    """Render Blockage indicator for Today's Snapshot."""
    blockage_df = detect_blockages(df)

    if blockage_df.empty:
        st.success("No Blockage detected today — capacity is available.")
        return

    n = len(blockage_df)
    st.error(
        f"**BLOCKAGE DETECTED** — {n} time bracket{'s' if n > 1 else ''} "
        f"where ALL experts are busy AND a clash exists. Zero available capacity!"
    )

    k = st.columns(4)
    k[0].metric("Blockage Brackets", n)
    k[1].metric("Experts (All Busy)", int(blockage_df["total_experts_today"].iloc[0]))
    k[2].metric("Total Clash Experts", int(blockage_df["experts_with_clash"].sum()))
    mean_min = blockage_df["mean_start_minutes"].mean()
    k[3].metric("Mean Blockage Time", _mean_start_label_from_minutes(mean_min))

    with st.expander("Blockage Details — Today"):
        display = blockage_df[["bracket_label", "total_experts_today",
                               "experts_with_clash", "mean_start_label"]].copy()
        display.columns = ["Time Bracket", "Total Experts", "Experts with Clash",
                           "Mean Start Time"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        for _, row in blockage_df.iterrows():
            st.caption(
                f"**{row['bracket_label']}** — Clash experts: "
                + ", ".join(row["clash_experts"])
                + f" | All experts: " + ", ".join(row["involved_experts"])
            )


def render_blockage_summary(df, title_suffix=""):
    """Full Blockage analytics: month-wise, expert-wise, time-wise."""
    blockage_df = detect_blockages(df)

    st.subheader("Blockage Analysis" + title_suffix)
    st.caption(
        "A **Blockage** = a 30-min bracket where ALL experts have interviews "
        "AND at least one expert has a clash (2+ interviews). "
        "Result: zero available capacity in that window."
    )

    if blockage_df.empty:
        st.success("No Blockages detected" + title_suffix + ".")
        return

    # ── KPI row ──────────────────────────────────────────────────
    total_blockages = len(blockage_df)
    days_with_blockage = blockage_df["date"].dt.date.nunique()
    months_with_blockage = blockage_df["month"].nunique()
    total_clash_experts = int(blockage_df["experts_with_clash"].sum())
    mean_min = blockage_df["mean_start_minutes"].mean()
    mean_label = _mean_start_label_from_minutes(mean_min)

    k = st.columns(5)
    k[0].metric("Total Blockage Events", total_blockages)
    k[1].metric("Days with Blockage", days_with_blockage)
    k[2].metric("Months with Blockage", months_with_blockage)
    k[3].metric("Total Clash Experts", total_clash_experts)
    k[4].metric("Mean Blockage Time", mean_label)

    # ── Month-wise Blockage Trend ────────────────────────────────
    st.markdown("---")
    st.subheader("Monthly Blockage Trend" + title_suffix)

    monthly_b = blockage_df.groupby("month").agg(
        blockage_count=("bracket_label", "size"),
        days=("date", lambda x: x.dt.date.nunique()),
        clash_experts=("experts_with_clash", "sum"),
        mean_start=("mean_start_minutes", "mean"),
    ).reset_index()
    monthly_b["mean_start_label"] = monthly_b["mean_start"].apply(
        _mean_start_label_from_minutes
    )
    monthly_b["mean_start_hour"] = (monthly_b["mean_start"] / 60).round(2)

    col_mb1, col_mb2 = st.columns(2)
    with col_mb1:
        fig_mb = go.Figure(go.Bar(
            x=monthly_b["month"], y=monthly_b["blockage_count"],
            marker_color="#e74c3c",
            text=monthly_b["blockage_count"], textposition="outside",
        ))
        fig_mb.update_layout(
            title="Blockage Events by Month",
            height=420,
            xaxis_title="Month",
            yaxis_title="Blockage Events",
        )
        st.plotly_chart(fig_mb, use_container_width=True)

    with col_mb2:
        fig_mm = go.Figure()
        fig_mm.add_trace(go.Scatter(
            x=monthly_b["month"],
            y=monthly_b["mean_start_hour"],
            mode="lines+markers+text",
            text=monthly_b["mean_start_label"],
            textposition="top center",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=10),
        ))
        fig_mm.update_layout(
            title="Mean Blockage Time by Month",
            height=420,
            xaxis_title="Month",
            yaxis_title="Hour of Day (24h)",
            yaxis=dict(range=[
                max(0, monthly_b["mean_start_hour"].min() - 2),
                min(24, monthly_b["mean_start_hour"].max() + 2),
            ]),
        )
        st.plotly_chart(fig_mm, use_container_width=True)

    # ── Expert-wise Blockage Frequency ───────────────────────────
    st.markdown("---")
    st.subheader("Expert-wise Blockage Involvement" + title_suffix)
    st.caption("How often each expert appears in a Blockage bracket (as busy or clashing)")

    exploded = blockage_df.explode("involved_experts")
    expert_blockage_counts = exploded.groupby("involved_experts").agg(
        blockage_events=("bracket_label", "size"),
        months=("month", "nunique"),
    ).reset_index().rename(columns={"involved_experts": "expert_name"})
    expert_blockage_counts = expert_blockage_counts.sort_values(
        "blockage_events", ascending=False
    )

    exploded_clash = blockage_df.explode("clash_experts")
    clash_expert_counts = exploded_clash.groupby("clash_experts").size().rename(
        "as_clash_expert"
    ).reset_index().rename(columns={"clash_experts": "expert_name"})

    expert_blockage = expert_blockage_counts.merge(
        clash_expert_counts, on="expert_name", how="left"
    ).fillna(0)
    expert_blockage["as_clash_expert"] = expert_blockage["as_clash_expert"].astype(int)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        top = expert_blockage.head(15).sort_values("blockage_events")
        fig_eb = go.Figure()
        fig_eb.add_trace(go.Bar(
            y=top["expert_name"], x=top["blockage_events"],
            orientation="h", marker_color="#e74c3c",
            text=top["blockage_events"], textposition="outside",
            name="Blockage Brackets",
        ))
        fig_eb.update_layout(
            title="Top 15 Experts in Blockage Events",
            height=max(420, len(top) * 35),
            xaxis_title="Blockage Events",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_eb, use_container_width=True)

    with col_e2:
        top2 = expert_blockage.head(15).sort_values("as_clash_expert")
        fig_ec = go.Figure()
        fig_ec.add_trace(go.Bar(
            y=top2["expert_name"], x=top2["as_clash_expert"],
            orientation="h", marker_color="#8e44ad",
            text=top2["as_clash_expert"], textposition="outside",
            name="As Clash Expert",
        ))
        fig_ec.update_layout(
            title="Top 15 — Times as Clash Expert in Blockage",
            height=max(420, len(top2) * 35),
            xaxis_title="Times as Clash Expert",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_ec, use_container_width=True)

    # ── Time-of-Day Blockage Distribution ────────────────────────
    st.markdown("---")
    st.subheader("Blockage Time-of-Day Distribution" + title_suffix)

    blockage_hours = (blockage_df["bracket_start_min"] // 60).astype(int)
    hour_counts = blockage_hours.value_counts().sort_index()

    all_hours = list(range(0, 24))
    hour_labels = [datetime(2000, 1, 1, h).strftime("%I %p").lstrip("0") for h in all_hours]
    counts = [int(hour_counts.get(h, 0)) for h in all_hours]
    peak_h = int(hour_counts.idxmax()) if not hour_counts.empty else 0
    bar_colors = ["#e74c3c" if h == peak_h else "#f39c12" for h in all_hours]

    fig_bh = go.Figure(go.Bar(
        x=hour_labels, y=counts,
        marker_color=bar_colors,
        text=counts, textposition="outside",
    ))
    fig_bh.update_layout(
        title="Blockage Events by Hour of Day" + title_suffix,
        height=420,
        xaxis_title="Hour of Day",
        yaxis_title="Blockage Events",
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_bh, use_container_width=True)

    # ── Expert x Month Heatmap ───────────────────────────────────
    st.markdown("---")
    st.subheader("Expert x Month Blockage Heatmap" + title_suffix)

    exploded_monthly = exploded.groupby(["involved_experts", "month"]).size().reset_index(
        name="count"
    ).rename(columns={"involved_experts": "expert_name"})
    pivot_em = exploded_monthly.pivot(
        index="expert_name", columns="month", values="count"
    ).fillna(0).astype(int)

    if not pivot_em.empty:
        fig_hm = px.imshow(
            pivot_em, text_auto=True, aspect="auto",
            color_continuous_scale="Reds",
            title="Expert x Month Blockage Frequency",
            labels=dict(x="Month", y="Expert", color="Blockages"),
        )
        fig_hm.update_layout(height=max(450, len(pivot_em) * 25))
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Detailed Blockage Table ──────────────────────────────────
    with st.expander("Detailed Blockage Events" + title_suffix):
        display_b = blockage_df.copy()
        display_b["date"] = display_b["date"].dt.strftime("%Y-%m-%d")
        display_b["involved_experts_str"] = display_b["involved_experts"].apply(
            lambda x: ", ".join(x)
        )
        display_b["clash_experts_str"] = display_b["clash_experts"].apply(
            lambda x: ", ".join(x)
        )
        st.dataframe(
            display_b[["date", "day_name", "bracket_label", "total_experts_today",
                        "experts_with_clash", "mean_start_label",
                        "involved_experts_str", "clash_experts_str", "month"]].rename(columns={
                "date": "Date",
                "day_name": "Day",
                "bracket_label": "Time Bracket",
                "total_experts_today": "Total Experts",
                "experts_with_clash": "Clash Experts",
                "mean_start_label": "Mean Start Time",
                "involved_experts_str": "All Experts",
                "clash_experts_str": "Clashing Experts",
                "month": "Month",
            }),
            use_container_width=True, hide_index=True,
        )


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
            k2.metric("P(Negative)", f"{overall['p_negative']:.1f}%")
            k3.metric("P(Neutral)", f"{overall['p_neutral']:.1f}%")
            k4.metric("P(Positive)", f"{overall['p_positive']:.1f}%")

            fig_donut = go.Figure(go.Pie(
                labels=["Negative", "Neutral", "Positive"],
                values=[overall["p_negative"], overall["p_neutral"], overall["p_positive"]],
                hole=0.5,
                marker=dict(colors=["#e74c3c", "#f39c12", "#2ecc71"]),
                textinfo="label+value+percent",
            ))
            fig_donut.update_layout(
                title="Probability Distribution", height=380, showlegend=False,
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
                        c_score = chunk["score"]
                        c_label = chunk["label"]
                        c_color = sentiment_color(c_score) if c_score is not None else "#999"
                        with st.expander(unit_name + " " + str(i + 1) + " — " + c_label +
                                          (" (" + f"{c_score:+.1f}%" + ")" if c_score is not None else "")):
                            st.markdown(chunk["full_text"])
                            if c_score is not None:
                                st.caption(
                                    f"Score: {c_score:+.1f}% | "
                                    f"P(neg): {chunk['p_negative']:.1f}% | "
                                    f"P(neu): {chunk['p_neutral']:.1f}% | "
                                    f"P(pos): {chunk['p_positive']:.1f}%"
                                )
            else:
                st.plotly_chart(fig_donut, use_container_width=True)


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

    # ── Sidebar: Clash indicator for Interview Support ────────────
    if selected_support == "Interview Support" and "_parsed_start" in support_df.columns:
        clash_groups_check, _ = detect_expert_clashes(support_df)
        if not clash_groups_check.empty:
            st.sidebar.markdown("---")
            st.sidebar.metric("⚠️ Total Clash Groups", len(clash_groups_check))
            st.sidebar.metric("Experts with Clashes", clash_groups_check["expert_name"].nunique())

        # Blockage sidebar indicator
        blockage_check = detect_blockages(support_df)
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
                                            "Transcript Analyzer"],
                            label_visibility="collapsed")

    # ======= TRANSCRIPT ANALYZER =======
    if view == "Transcript Analyzer":
        transcript_analyzer_view()
        st.sidebar.markdown("---")
        st.sidebar.caption("Vizva Dashboard v19.0 | API-powered | Active Experts Only | RoBERTa ONNX | Start Time Analytics | Clash Detection | Blockage")
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

        # ── TODAY'S SENTIMENT KPI ────────────────────────────────────
        if not today_df.empty:
            today_stats = get_sentiment_stats(today_df)
            if today_stats:
                st.markdown("---")
                render_sentiment_kpi(today_stats, title="Today's Feedback Sentiment")

        # ── TODAY'S START TIME INSIGHTS ──────────────────────────────
        if selected_support == "Interview Support" and not today_df.empty and "start_hour" in today_df.columns:
            st.markdown("---")
            render_start_time_insights(today_df, title_suffix=" - " + str(today))

        # ── TODAY'S CLASH DETECTION ──────────────────────────────────
        if selected_support == "Interview Support" and not today_df.empty and "_parsed_start" in today_df.columns:
            st.markdown("---")
            st.subheader("Scheduling Clashes - " + str(today))
            render_today_clash_summary(today_df)

        # ── TODAY'S BLOCKAGE DETECTION ───────────────────────────────
        if selected_support == "Interview Support" and not today_df.empty and "_parsed_start" in today_df.columns:
            st.markdown("---")
            st.subheader("Blockage Indicator - " + str(today))
            render_today_blockage(today_df)

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

            # ── SENTIMENT DONUT + HISTOGRAM — TODAY ──────────────────
            st.markdown("---")
            render_sentiment_section(today_df, section_title="Sentiment Analysis - " + str(today))

            # ── FEEDBACK WORD CLOUD — TODAY ──────────────────────────
            st.markdown("---")
            feedback_texts = extract_feedback_texts(today_df)
            render_wordcloud_section(
                feedback_texts,
                section_title="Feedback Word Cloud - " + str(today),
            )

            with st.expander("Raw Data (includes Sentiment Score)"):
                st.dataframe(today_df, use_container_width=True, height=400)

        # ── ABOUT TO MOVE TO MARKET ──────────────────────────────────
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

        # ── CURRENT MONTH SENTIMENT KPI ──────────────────────────────
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

        # ── START TIME INSIGHTS — OVERALL ────────────────────────────
        if selected_support == "Interview Support" and "start_hour" in support_df.columns:
            st.markdown("---")
            render_start_time_insights(support_df, title_suffix=" - All Months")

            st.markdown("---")
            render_monthly_start_time_trend(support_df, title_suffix=" - " + support_label)

        # ── CLASH DETECTION — OVERALL ────────────────────────────────
        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns:
            st.markdown("---")
            render_clash_summary(support_df, title_suffix=" - All Months")

        # ── BLOCKAGE ANALYSIS — OVERALL ──────────────────────────────
        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns:
            st.markdown("---")
            render_blockage_summary(support_df, title_suffix=" - All Months")

        # ── MONTHLY SENTIMENT TRENDS ─────────────────────────────────
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

        # ── MONTHLY (Interview Support Only) ─────────────────────────
        if selected_support == "Interview Support":
            st.markdown("---")
            st.subheader("Round-wise Monthly Breakdown")
            if "round_name" in support_df.columns:
                support_df_copy = support_df.copy()
                support_df_copy["month"] = support_df_copy["date"].dt.to_period("M").astype(str)
                months_available = sorted(support_df_copy["month"].unique(), reverse=True)
                sel_month = st.selectbox("Select Month", months_available, index=0, key="round_month")
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
                sel_m2 = st.selectbox("Select Month", sorted(month_exp["month"].unique(), reverse=True),
                                      index=0, key="exp_month")
                me = month_exp[month_exp["month"] == sel_m2].sort_values("total", ascending=False)
                st.dataframe(me, use_container_width=True)

        # ── CANDIDATE-WISE MONTHLY COUNTS ────────────────────────────
        st.markdown("---")
        st.subheader("Candidate-wise Monthly Counts (All Support Types)")
        cand_monthly = candidate_monthly_support(active_expert_df)
        if not cand_monthly.empty:
            sel_m3 = st.selectbox("Select Month", sorted(cand_monthly["month"].unique(), reverse=True),
                                  index=0, key="cand_month")
            cm = cand_monthly[cand_monthly["month"] == sel_m3].sort_values("total", ascending=False)
            st.dataframe(cm, use_container_width=True)

    # ======= DAILY DRILL-DOWN =======
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

            # ── PERIOD SENTIMENT KPI ─────────────────────────────────
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

            # ── DAILY SENTIMENT TREND LINE ───────────────────────────
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

            # ── SINGLE DAY SENTIMENT KPI ─────────────────────────────
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

            # ── SINGLE DAY SENTIMENT DONUT + HISTOGRAM ───────────────
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

        # Build tab list dynamically
        tab_names = ["Experts", "Companies", "Rounds", "Day of Week", "Cross-Support", "Candidates", "Technology"]
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
                    Rescheduled=("task_status", lambda x: (x== "rescheduled").sum()),
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

        # ── START TIME ANALYSIS TAB ──────────────────────────────────
        if selected_support == "Interview Support" and "start_hour" in support_df.columns and "Start Time" in tab_names:
            with tabs[tab_names.index("Start Time")]:
                render_start_time_insights(support_df, title_suffix=" - All Data")
                st.markdown("---")
                render_monthly_start_time_trend(support_df, title_suffix="")

        # ── CLASH DETECTION TAB ──────────────────────────────────────
        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns and "Clash Detection" in tab_names:
            with tabs[tab_names.index("Clash Detection")]:
                render_clash_summary(support_df, title_suffix=" - All Data")

        # ── BLOCKAGE TAB ─────────────────────────────────────────────
        if selected_support == "Interview Support" and "_parsed_start" in support_df.columns and "Blockage" in tab_names:
            with tabs[tab_names.index("Blockage")]:
                render_blockage_summary(support_df, title_suffix=" - All Data")

    st.sidebar.markdown("---")
    st.sidebar.caption("Vizva Dashboard v19.0 | API-powered | Active Experts Only | RoBERTa ONNX | Start Time Analytics | Clash Detection | Blockage")


# ═══════════════════════════════════════════════════════════════════
# AUTHENTICATION LAYER
# ═══════════════════════════════════════════════════════════════════

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
