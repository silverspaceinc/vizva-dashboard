import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import requests

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
    cols_to_drop = ["case_candidate_phone", "case_candidate_email", "candidate_phone", "candidate_email", "candidate_status_flag",
                    "expert_is_team_lead", "expert_date_of_joining", "filled_by_first_name", "filled_by_last_name",
                    "filled_by_email"]
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


def main():
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

    # --- SUPPORT TYPE SELECTOR ---
    st.sidebar.header("Support Type")
    selected_support = st.sidebar.selectbox("Select Support Type", SUPPORT_TYPES, index=0)
    support_label = selected_support

    support_df = get_by_support(active_expert_df, selected_support)

    # Sidebar counts for all types
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
    view = st.sidebar.radio("", ["Todays Snapshot", "Monthly Overview",
                                  "Daily Drill-Down", "Deep-Dive Analytics"])

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

            # -- ROUND WISE TODAY (Interview Support Only) --
            if selected_support == "Interview Support":
                st.markdown("---")
                st.subheader("Round Breakdown - " + str(today))
                round_charts(today_df, " - " + str(today))

            with st.expander("Raw Data"):
                st.dataframe(today_df, use_container_width=True, height=400)

    # ======= MONTHLY =======
    elif view == "Monthly Overview":
        if monthly.empty:
            st.warning("No monthly data available for " + support_label)
            st.stop()

        has_hist = selected_support in HIST
        title_range = "(Jul 2025 - Present)" if has_hist else "(2026 - Present)"
        st.header("Monthly Overview - " + support_label + " " + title_range)
        latest = monthly.iloc[-1]

        c = st.columns(6)
        c[0].metric("Month", latest["month"])
        c[1].metric("Completed", int(latest["completed"]))
        c[2].metric("Rescheduled", int(latest["rescheduled"]))
        c[3].metric("Cancelled", int(latest["cancelled"]))
        c[4].metric("Pending", int(latest["pending"]))
        c[5].metric("Candidates", int(latest["candidates"]))

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

        with st.expander("Monthly Data Table"):
            st.dataframe(monthly, use_container_width=True)

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

            # -- ROUND WISE FOR DATE RANGE (Interview Support Only) --
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

            # -- ROUND WISE FOR SINGLE DAY (Interview Support Only) --
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

        tabs = st.tabs(["Experts", "Companies", "Rounds",
                         "Day of Week", "All Support Types",
                         "Candidates", "Technology", "Status"])

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

        with tabs[7]:
            if "status" in support_df.columns:
                sc = support_df["status"].value_counts()
                fig = go.Figure(go.Bar(x=sc.index, y=sc.values, marker_color="#e67e22",
                                       text=sc.values, textposition="outside"))
                fig.update_layout(title="Case Status Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Vizva Dashboard v10.0 | API-powered | Active Experts Only")


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
