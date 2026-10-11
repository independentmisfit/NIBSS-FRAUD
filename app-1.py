"""
NIBSS Settlement Risk Monitor — Streamlit version
Mirrors the 3-page Power BI build: Overview / Temporal & Behavioral / Amount & Settlement Risk
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------- Page config & theme colors (matches the Power BI blue theme) ----------
st.set_page_config(page_title="Settlement Risk Monitor", layout="wide", page_icon="🏦")

NAVY = "#0A3D91"
BLUE = "#1857B8"
LIGHT_BLUE = "#7FA6E0"
AMBER = "#C88A1A"
ALERT = "#C33A2E"
TEAL = "#1B8A8A"
BG = "#EEF1F6"

DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {NAVY}; }}

    /* KPI metric cards */
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #DDE3EC;
        border-radius: 10px;
        padding: 14px 16px;
        border-top: 3px solid {BLUE};
    }}
    div[data-testid="stMetricLabel"] {{ color: #5B6B82 !important; }}
    div[data-testid="stMetricValue"] {{ color: {NAVY} !important; }}
    div[data-testid="stMetricDelta"] {{ color: {ALERT} !important; }}

    /* Chart section headers (st.subheader) */
    h3 {{ color: {NAVY} !important; }}

    /* Tab labels */
    button[data-baseweb="tab"] {{
        color: #5B6B82 !important;
    }}
    button[data-baseweb="tab"] p {{
        color: inherit !important;
        font-weight: 600;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {BLUE} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] p {{
        color: {BLUE} !important;
    }}

    /* Sidebar text */
    section[data-testid="stSidebar"] {{ background-color: #FFFFFF; }}
    section[data-testid="stSidebar"] * {{ color: {NAVY} !important; }}

    /* Captions and general body text */
    .stCaption, p {{ color: {NAVY}; }}

    .block-container {{ padding-top: 1.5rem; }}
</style>
""", unsafe_allow_html=True)


# ---------- Data loading ----------
@st.cache_data
def load_data():
    df = pd.read_parquet("nibss_data.parquet")
    df["day_name"] = df["day_of_week"].map(DAY_NAMES)
    df["fraud_status"] = df["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "nibss_data.parquet not found. Run convert_to_parquet.py on your full CSV first, "
        "then place the resulting nibss_data.parquet in this same folder."
    )
    st.stop()


# ---------- Header ----------
st.markdown(f"""
<div style="background:linear-gradient(120deg,{NAVY},{BLUE});padding:20px 28px;border-radius:10px;margin-bottom:10px;">
    <div style="color:white;font-size:22px;font-weight:700;">Settlement Risk Monitor — NIBSS Transaction Fraud</div>
    <div style="color:#C7D7F0;font-size:12.5px;margin-top:4px;">
        {len(df):,} transactions · {df['bank'].nunique()} banks ·
        fraud rate {df['is_fraud'].mean()*100:.2f}% ({df['is_fraud'].sum():,} cases) ·
        engineered risk features, model scoring pending
    </div>
</div>
""", unsafe_allow_html=True)


# ---------- Sidebar: bank filter (with All option) ----------
st.sidebar.header("Filters")
all_banks = sorted(df["bank"].unique().tolist())
bank_selection = st.sidebar.multiselect("Bank", options=["All"] + all_banks, default=["All"])

if "All" in bank_selection or not bank_selection:
    filtered = df
else:
    filtered = df[df["bank"].isin(bank_selection)]


# ---------- Shared measures ----------
total_tx = len(filtered)
fraud_tx = int(filtered["is_fraud"].sum())
fraud_rate = fraud_tx / total_tx if total_tx else 0
fraud_value = filtered.loc[filtered["is_fraud"] == 1, "amount"].sum()
total_value = filtered["amount"].sum()


# ---------- Tabs (mirrors the Power BI page navigator) ----------
tab1, tab2, tab3 = st.tabs(["1 · Overview", "2 · Temporal & Behavioral", "3 · Amount & Settlement Risk"])

# ================= PAGE 1: OVERVIEW =================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total_tx:,}")
    c2.metric("Fraud Transactions", f"{fraud_tx:,}", f"{fraud_rate*100:.2f}% of volume")
    c3.metric("Fraud Value", f"₦{fraud_value/1e6:,.1f}M", f"of ₦{total_value/1e9:,.1f}B total")
    c4.metric("Fraud Rate", f"{fraud_rate*100:.2f}%")

    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.subheader("Transaction Volume vs Fraud Rate by Hour")
        hourly = filtered.groupby("hour").agg(
            total_tx=("is_fraud", "count"),
            fraud_rate=("is_fraud", "mean")
        ).reset_index()
        fig = go.Figure()
        fig.add_bar(x=hourly["hour"], y=hourly["total_tx"], name="Volume",
                    marker_color=LIGHT_BLUE, yaxis="y1")
        fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["fraud_rate"]*100, name="Fraud rate %",
                                  mode="lines+markers", line=dict(color=ALERT), yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Txn count"),
            yaxis2=dict(title="Fraud %", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=10, b=10), height=330
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Fraud Transactions by Bank")
        by_bank = filtered[filtered.is_fraud == 1].groupby("bank").size().reset_index(name="fraud_tx")
        by_bank = by_bank.sort_values("fraud_tx", ascending=True)
        fig = px.bar(by_bank, x="fraud_tx", y="bank", orientation="h", color_discrete_sequence=[NAVY])
        fig.update_layout(margin=dict(t=10, b=10), height=330, xaxis_title="Fraud transactions", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns([1, 2])
    with col_c:
        st.subheader("Fraud by Channel")
        by_channel = filtered[filtered.is_fraud == 1].groupby("channel").size().reset_index(name="fraud_tx")
        fig = px.pie(by_channel, names="channel", values="fraud_tx", hole=0.6,
                     color_discrete_sequence=[ALERT, AMBER, TEAL, BLUE, LIGHT_BLUE, "#D9A566"])
        fig.update_layout(margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Top 5 Locations by Fraud Value")
        by_loc = filtered.groupby("location").agg(
            total_tx=("is_fraud", "count"),
            fraud_tx=("is_fraud", "sum"),
            fraud_value=("amount", lambda x: x[filtered.loc[x.index, "is_fraud"] == 1].sum())
        ).reset_index().sort_values("fraud_value", ascending=False).head(5)
        by_loc["fraud_value"] = by_loc["fraud_value"].apply(lambda v: f"₦{v/1e6:,.0f}M")
        by_loc["total_tx"] = by_loc["total_tx"].apply(lambda v: f"{v:,}")
        by_loc.columns = ["Location", "Total Txns", "Fraud Txns", "Fraud Value"]
        st.dataframe(by_loc, use_container_width=True, hide_index=True)


# ================= PAGE 2: TEMPORAL & BEHAVIORAL =================
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fraud Heatmap — Day × Hour")
        heat = filtered.groupby(["day_name", "hour"])["is_fraud"].sum().reset_index()
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot = heat.pivot(index="day_name", columns="hour", values="is_fraud").reindex(day_order)
        fig = px.imshow(pivot, color_continuous_scale=["white", ALERT], aspect="auto")
        fig.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Velocity Score — Peak Hour vs Off-Peak")
        vel = filtered.groupby(["is_peak_hour", "fraud_status"])["velocity_score"].mean().reset_index()
        vel["is_peak_hour"] = vel["is_peak_hour"].map({True: "Peak Hour", False: "Off-Peak"})
        fig = px.bar(vel, x="is_peak_hour", y="velocity_score", color="fraud_status", barmode="group",
                     color_discrete_map={"Legitimate": TEAL, "Fraud": ALERT})
        fig.update_layout(margin=dict(t=10, b=10), height=320, xaxis_title="", yaxis_title="Avg velocity score")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d, col_e = st.columns(3)
    with col_c:
        st.subheader("Fraud by Age Group")
        by_age = filtered[filtered.is_fraud == 1].groupby("age_group").size().reset_index(name="fraud_tx")
        fig = px.bar(by_age, x="age_group", y="fraud_tx", color_discrete_sequence=[BLUE])
        fig.update_layout(margin=dict(t=10, b=10), height=300, xaxis_title="", yaxis_title="Fraud transactions")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Fraud Technique Breakdown")
        by_tech = filtered[filtered.is_fraud == 1].groupby("fraud_technique").size().reset_index(name="count")
        by_tech = by_tech.sort_values("count", ascending=True)
        fig = px.bar(by_tech, x="count", y="fraud_technique", orientation="h", color_discrete_sequence=[NAVY])
        fig.update_layout(margin=dict(t=10, b=10), height=300, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_e:
        st.subheader("Composite Risk Distribution")
        fig = px.histogram(filtered, x="composite_risk", color="fraud_status", nbins=20, barmode="overlay",
                            color_discrete_map={"Legitimate": LIGHT_BLUE, "Fraud": ALERT}, opacity=0.75)
        fig.update_layout(margin=dict(t=10, b=10), height=300, xaxis_title="composite_risk")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("velocity_score and composite_risk are pre-built engineered risk indicators from the dataset, "
               "not model output — the ANN fraud classifier is not yet deployed.")


# ================= PAGE 3: AMOUNT & SETTLEMENT RISK =================
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Amount Distribution (log-scaled)")
        fig = px.histogram(filtered, x="amount_log", color="fraud_status", nbins=40, barmode="overlay",
                            color_discrete_map={"Legitimate": LIGHT_BLUE, "Fraud": ALERT}, opacity=0.75)
        fig.update_layout(margin=dict(t=10, b=10), height=330, xaxis_title="amount_log")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Amount Spread by Fraud Status")
        fig = px.box(filtered, x="fraud_status", y="amount", color="fraud_status",
                     color_discrete_map={"Legitimate": TEAL, "Fraud": ALERT})
        fig.update_layout(margin=dict(t=10, b=10), height=330, showlegend=False, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Merchant Category Risk")
        by_merch = filtered.groupby("merchant_category").agg(
            fraud_tx=("is_fraud", "sum"),
            fraud_rate=("is_fraud", "mean"),
            avg_risk=("merchant_risk_score", "mean")
        ).reset_index().sort_values("fraud_tx", ascending=False)
        by_merch["fraud_rate"] = (by_merch["fraud_rate"] * 100).round(2).astype(str) + "%"
        by_merch["avg_risk"] = by_merch["avg_risk"].round(2)
        by_merch.columns = ["Category", "Fraud Txns", "Fraud Rate", "Avg Risk Score"]
        st.dataframe(by_merch, use_container_width=True, hide_index=True)

    with col_d:
        st.subheader("Top 50 Risk Transactions")
        top_risk = filtered.sort_values("composite_risk", ascending=False).head(50)[
            ["transaction_id", "bank", "channel", "amount", "composite_risk", "fraud_status"]
        ]
        top_risk["amount"] = top_risk["amount"].apply(lambda v: f"₦{v:,.0f}")
        top_risk["composite_risk"] = top_risk["composite_risk"].round(2)
        top_risk.columns = ["Txn ID", "Bank", "Channel", "Amount", "Comp. Risk", "Status"]
        st.dataframe(top_risk, use_container_width=True, hide_index=True, height=330)

