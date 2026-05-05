import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import requests

st.set_page_config(page_title="Vizva Interview Dashboard", page_icon="chart_with_upwards_trend",
                   layout="wide", initial_sidebar_state="expanded")

API_KEY = "9c7f3b1d9a8e4c6f1b2a0e7d5f9c3b8a9e1d2f4c6b8a0e3d7f9a1b2c4d6e8f0a"
BASE_URL = "http://69.62.76.34"

TASK_ORDER = ["completed", "rescheduled", "cancelled"]
TASK_LABEL = {"completed": "Completed", "rescheduled": "Rescheduled",
              "cancelled": "Cancelled"}
CLR = {"completed": "#2ecc71", "rescheduled": "#f39c12",
       "cancelled": "#e74c3c"}

HIST = {
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
}


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
    # --- ADD THIS LINE TO REMOVE PHONE AND EMAIL ---
    cols_to_drop = ["case_candidate_phone", "case_candidate_email","candidate_phone","candidate_email"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "task_status" in df.columns:
        df["task_status"] = df["task_status"].astype(str).str.strip().str.lower()
        df["task_status"] = df["task_status"].replace("pending", "cancelled")
        df["task_status"] = df["task_status"].replace("not done", "cancelled")
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
    return df[df["expert_status_flag"] == True].copy()


def get_by_support(df, support_type):
    if df.empty or "support_name" not in df.columns:
        return df
    return df[df["support_name"].str.lower() == support_type.lower()].copy()


def hist_monthly_df():
    rows = []
    for m, d in HIST.items():
        total = d["completed"] + d["rescheduled"] + d["cancelled"]
        rows.append({"month": m, "completed": d["completed"], "rescheduled": d["rescheduled"],
                      "cancelled": d["cancelled"],
                      "total": total, "candidates": d["candidates"]})
    return pd.DataFrame(rows)


def live_monthly(idf, from_date="2026-05-01"):
    if idf.empty or "date" not in idf.columns:
        return pd.DataFrame()
    f = idf[idf["date"] >= pd.Timestamp(from_date)].copy()
    if f.empty:
        return pd.DataFrame()
    f["month"] = f["date"].dt.to_period("M").astype(str)
    rows = []
    for m, g in f.groupby("month"):
        tc = g["task_status"].value_counts()
        rows.append({"month": m, "completed": int(tc.get("completed", 0)),
                      "rescheduled": int(tc.get("rescheduled", 0)),
                      "cancelled": int(tc.get("cancelled", 0)),
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
        interview_count = len(g[g["support_name"].str.lower() == "interview support"])
        assessment_count = len(g[g["support_name"].str.lower() == "assessment support"])
        rows.append({"month": month, "candidate_name": cand,
                      "interview_count": interview_count,
                      "assessment_count": assessment_count,
                      "total": interview_count + assessment_count})
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
                      "total": len(g),
                      "candidates": g["candidate_name"].nunique() if "candidate_name" in g.columns else 0})
    return pd.DataFrame(rows)


def kpi_row(data):
    c = st.columns(4)
    c[0].metric("Completed", int(data.get("completed", 0)))
    c[1].metric("Rescheduled", int(data.get("rescheduled", 0)))
    c[2].metric("Cancelled", int(data.get("cancelled", 0)))
    c[3].metric("Candidates", int(data.get("candidates", 0)))


def stacked_bar(df, x="month", title="Monthly Interview Counts"):
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
    fig.update_layout(title="Interviews by Day of Week", height=400)
    return fig


def main():
    st.title("Vizva Interview Support Dashboard")

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
    interview_df = get_by_support(active_expert_df, "interview support")
    assessment_df = get_by_support(active_expert_df, "assessment support")

    if interview_df.empty:
        st.error("No Interview Support rows found for active experts.")
        st.stop()

    st.sidebar.success("Loaded " + str(len(interview_df)) + " interview rows")
    st.sidebar.metric("Total Cases (This Year)", len(all_case_df))
    st.sidebar.metric("Interview Support", len(interview_df))
    st.sidebar.metric("Assessment Support", len(assessment_df))
    st.sidebar.caption("Only active experts shown")
    st.sidebar.caption("Data refreshes every 10 minutes")

    if st.sidebar.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    hist = hist_monthly_df()
    live = live_monthly(interview_df)
    monthly = pd.concat([hist, live], ignore_index=True).drop_duplicates("month", keep="last")

    st.sidebar.markdown("---")
    st.sidebar.header("View")
    view = st.sidebar.radio("", ["Todays Snapshot", "Monthly Overview",
                                  "Daily Drill-Down", "Deep-Dive Analytics"])

    # ═══════ TODAY ═══════
    if view == "Todays Snapshot":
        st.header("Todays Snapshot")
        today = date.today()
        today_df = interview_df[interview_df["date"].dt.date == today]

        st.caption(today.strftime("%A, %B %d, %Y"))

        if today_df.empty:
            st.info("No interviews scheduled for today.")
            kd = {"completed": 0, "rescheduled": 0, "cancelled": 0, "candidates": 0}
        else:
            tc = today_df["task_status"].value_counts()
            kd = {s: int(tc.get(s, 0)) for s in TASK_ORDER}
            kd["candidates"] = today_df["candidate_name"].nunique()

        kpi_row(kd)

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

            with st.expander("Raw Data"):
                st.dataframe(today_df, use_container_width=True, height=400)

    # ═══════ MONTHLY ═══════
    elif view == "Monthly Overview":
        st.header("Monthly Overview (Jul 2025 - Present)")
        latest = monthly.iloc[-1]

        c = st.columns(5)
        c[0].metric("Month", latest["month"])
        c[1].metric("Completed", int(latest["completed"]))
        c[2].metric("Rescheduled", int(latest["rescheduled"]))
        c[3].metric("Cancelled", int(latest["cancelled"]))
        c[4].metric("Candidates", int(latest["candidates"]))

        st.plotly_chart(stacked_bar(monthly), use_container_width=True)

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

        st.plotly_chart(trend_line(monthly, "total", "Total Interviews per Month", "#8e44ad"), use_container_width=True)

        with st.expander("Monthly Data Table"):
            st.dataframe(monthly, use_container_width=True)

        # ── EXPERT WISE MONTHLY ──
        st.markdown("---")
        st.subheader("Expert-wise Monthly Breakdown")

        exp_monthly = expert_monthly(interview_df)
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
                                        "total", "candidates"]],
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

        # ── CANDIDATE WISE MONTHLY ──
        st.markdown("---")
        st.subheader("Candidate-wise Monthly Counts (Interview + Assessment)")

        cand_monthly = candidate_monthly_support(active_expert_df)
        if not cand_monthly.empty:
            cand_months = sorted(cand_monthly["month"].unique())
            sel_cand_month = st.selectbox("Select Month ", cand_months, index=len(cand_months) - 1, key="cand_month")

            cand_month_data = cand_monthly[cand_monthly["month"] == sel_cand_month].sort_values("total", ascending=False)

            if not cand_month_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=cand_month_data["candidate_name"],
                                     y=cand_month_data["interview_count"],
                                     name="Interview Support", marker_color="#3498db",
                                     text=cand_month_data["interview_count"], textposition="inside"))
                fig.add_trace(go.Bar(x=cand_month_data["candidate_name"],
                                     y=cand_month_data["assessment_count"],
                                     name="Assessment Support", marker_color="#e67e22",
                                     text=cand_month_data["assessment_count"], textposition="inside"))
                fig.update_layout(barmode="stack",
                                  title="Candidate Counts - " + sel_cand_month,
                                  height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(cand_month_data, use_container_width=True)

    # ═══════ DAILY ═══════
    elif view == "Daily Drill-Down":
        st.header("Daily Drill-Down")
        min_d = interview_df["date"].dt.date.min()
        max_d = interview_df["date"].dt.date.max()

        ca, cb = st.sidebar.columns(2)
        start = ca.date_input("From", value=max(min_d, date(2026, 5, 1)),
                              min_value=min_d, max_value=max_d)
        end = cb.date_input("To", value=max_d, min_value=min_d, max_value=max_d)

        mask = (interview_df["date"].dt.date >= start) & (interview_df["date"].dt.date <= end)
        period = interview_df[mask]
        dd = daily_agg(period)

        if dd.empty:
            st.info("No data in selected range.")
        else:
            kd = {s: dd[s].sum() for s in TASK_ORDER}
            kd["candidates"] = period["candidate_name"].nunique()
            kpi_row(kd)

            dd_plot = dd.copy()
            dd_plot["day"] = pd.to_datetime(dd_plot["day"])
            fig = go.Figure()
            for s in TASK_ORDER:
                fig.add_trace(go.Scatter(x=dd_plot["day"], y=dd_plot[s], mode="lines+markers",
                                         name=TASK_LABEL[s], line=dict(color=CLR[s])))
            fig.update_layout(title="Daily Trend (" + str(start) + " to " + str(end) + ")", height=450)
            st.plotly_chart(fig, use_container_width=True)

            st.plotly_chart(stacked_bar(dd_plot, x="day", title="Daily Stacked View"), use_container_width=True)

            with st.expander("Daily Table"):
                disp = dd.copy()
                disp["day"] = pd.to_datetime(disp["day"]).dt.strftime("%Y-%m-%d")
                st.dataframe(disp, use_container_width=True)

        st.sidebar.markdown("---")
        single = st.sidebar.date_input("Inspect single day", value=max_d,
                                       min_value=min_d, max_value=max_d, key="single")
        sdf = interview_df[interview_df["date"].dt.date == single]
        if not sdf.empty:
            st.subheader("Details - " + str(single))
            tc = sdf["task_status"].value_counts()
            kd2 = {s: int(tc.get(s, 0)) for s in TASK_ORDER}
            kd2["candidates"] = sdf["candidate_name"].nunique()
            kpi_row(kd2)

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

            st.dataframe(sdf, use_container_width=True)
        elif single:
            st.info("No interviews on " + str(single))

    # ═══════ DEEP DIVE ═══════
    elif view == "Deep-Dive Analytics":
        st.header("Deep-Dive Analytics")

        tabs = st.tabs(["Experts", "Companies", "Rounds",
                         "Day of Week", "Support Types",
                         "Candidates", "Technology", "Status"])

        with tabs[0]:
            fig = expert_stack(interview_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if "expert_name" in interview_df.columns:
                ea = interview_df.groupby("expert_name").agg(
                    Total=("task_status", "size"),
                    Completed=("task_status", lambda x: (x == "completed").sum()),
                    Rescheduled=("task_status", lambda x: (x == "rescheduled").sum()),
                    Cancelled=("task_status", lambda x: (x == "cancelled").sum()),
                    Candidates=("candidate_name", "nunique"),
                    Companies=("company_name", "nunique"),
                ).reset_index()
                ea["Completion_Pct"] = (ea["Completed"] / ea["Total"] * 100).round(1)
                st.dataframe(ea.sort_values("Total", ascending=False), use_container_width=True)

        with tabs[1]:
            fig = h_bar_by_task(interview_df, "company_name", 20, "Top 20 Companies by Volume")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if "company_name" in interview_df.columns:
                top = interview_df["company_name"].value_counts().head(15).index
                ct = interview_df[interview_df["company_name"].isin(top)]
                pv = ct.groupby(["company_name", "task_status"]).size().unstack(fill_value=0)
                fig2 = px.imshow(pv, text_auto=True, aspect="auto",
                                color_continuous_scale="Blues", title="Company x Task Heatmap")
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[2]:
            if "round_name" in interview_df.columns:
                rc = interview_df["round_name"].value_counts()
                fig = go.Figure(go.Pie(labels=rc.index, values=rc.values, hole=.4,
                                      textinfo="label+value+percent"))
                fig.update_layout(title="Round Distribution", height=420)
                st.plotly_chart(fig, use_container_width=True)

                rt = interview_df.groupby(["round_name", "task_status"]).size().unstack(fill_value=0)
                fig2 = px.imshow(rt, text_auto=True, aspect="auto",
                                color_continuous_scale="Oranges", title="Round x Task Heatmap")
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[3]:
            fig = day_of_week_chart(interview_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if "date" in interview_df.columns:
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                tmp = interview_df.copy()
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
            if "candidate_name" in interview_df.columns:
                ca_agg = interview_df.groupby("candidate_name").agg(
                    Interviews=("task_status", "size"),
                    Completed=("task_status", lambda x: (x == "completed").sum()),
                    Cancelled=("task_status", lambda x: (x == "cancelled").sum()),
                    Companies=("company_name", "nunique"),
                    Experts=("expert_name", "nunique"),
                ).reset_index()
                ca_agg["Completion_Pct"] = (ca_agg["Completed"] / ca_agg["Interviews"] * 100).round(1)
                ca_agg = ca_agg.sort_values("Interviews", ascending=False)
                st.subheader("Candidate Performance Table")
                st.dataframe(ca_agg, use_container_width=True)

                top30 = ca_agg.head(20).sort_values("Interviews")
                fig = go.Figure()
                fig.add_trace(go.Bar(y=top30["candidate_name"], x=top30["Completed"],
                                     name="Completed", orientation="h", marker_color="#2ecc71"))
                fig.add_trace(go.Bar(y=top30["candidate_name"], x=top30["Cancelled"],
                                     name="Cancelled", orientation="h", marker_color="#e74c3c"))
                fig.update_layout(barmode="stack", title="Top 20 Candidates", height=600,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

        with tabs[6]:
            if "candidate_technology" in interview_df.columns:
                fig = h_bar_by_task(interview_df, "candidate_technology", 20, "Top 20 Technologies")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                top_tech = interview_df["candidate_technology"].value_counts().head(15).index
                tt = interview_df[interview_df["candidate_technology"].isin(top_tech)]
                pv = tt.groupby(["candidate_technology", "task_status"]).size().unstack(fill_value=0)
                fig2 = px.imshow(pv, text_auto=True, aspect="auto",
                                color_continuous_scale="Greens", title="Technology x Task Heatmap")
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No technology column found.")

        with tabs[7]:
            if "status" in interview_df.columns:
                sc = interview_df["status"].value_counts()
                fig = go.Figure(go.Bar(x=sc.index, y=sc.values, marker_color="#e67e22",
                                       text=sc.values, textposition="outside"))
                fig.update_layout(title="Case Status Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Vizva Dashboard v7.0 | API-powered | Active Experts Only")

# ════════════════════════════════════════════════════════════
# AUTHENTICATION LAYER
# ════════════════════════════════════════════════════════════

def login():
    st.title("Vizva Secure Access")
    with st.container():
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if user == "ukteamwork" and pw == "5ilv3rSpac3!":
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
