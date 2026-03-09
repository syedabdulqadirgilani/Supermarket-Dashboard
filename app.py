import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# 1. PAGE SETUP
st.set_page_config(page_title="Supermarket Executive Dashboard", layout="wide", page_icon="🏢")

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #0083B8; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; }
    .stTabs [aria-selected="true"] { background-color: #0083B8; color: white; }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA ENGINE
@st.cache_data
def load_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time_dt"] = pd.to_datetime(df["Time"], errors='coerce')
    df["Hour"] = df["Time_dt"].dt.hour
    df["Day"] = df["Date"].dt.day_name()
    return df

df = load_data()

# 3. SIDEBAR FILTERS
st.sidebar.title("🔍 Global Dashboard Filters")
city_filter = st.sidebar.multiselect("Select City", options=df["City"].unique(), default=df["City"].unique())
ctype_filter = st.sidebar.multiselect("Customer Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())
branch_filter = st.sidebar.multiselect("Branch", options=df["Branch"].unique(), default=df["Branch"].unique())
gender_filter = st.sidebar.multiselect("Gender", options=df["Gender"].unique(), default=df["Gender"].unique())

# Filtered Dataframe
df_selection = df.query(
    "City == @city_filter & `Customer type` == @ctype_filter & Branch == @branch_filter & Gender == @gender_filter"
)

# 4. DASHBOARD TABS
st.title("🛒 Supermarket Business Intelligence System")
tab_overview, tab_sales, tab_tax, tab_time, tab_forecast = st.tabs([
    "📊 Overview & KPIs", "💰 Sales Deep-Dive", "🧾 Tax Analysis", "📈 Time Series", "🔮 Sales Forecasting"
])

# --- TAB 1: OVERVIEW & KPIs ---
with tab_overview:
    total_sales = df_selection["Sales"].sum()
    total_profit = df_selection["gross income"].sum()
    avg_rating = round(df_selection["Rating"].mean(), 1)
    total_qty = df_selection["Quantity"].sum()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Revenue", f"${total_sales:,.2f}")
    kpi2.metric("Gross Income", f"${total_profit:,.2f}")
    kpi3.metric("Units Sold", f"{total_qty:,}")
    kpi4.metric("Avg Rating", f"{avg_rating} / 10")

    col1, col2 = st.columns(2)
    with col1:
        fig_city = px.pie(df_selection, values="Sales", names="City", hole=0.5, title="Revenue by City")
        st.plotly_chart(fig_city, use_container_width=True)
    with col2:
        fig_prod = px.bar(df_selection.groupby("Product line")["Sales"].sum().reset_index(), 
                          x="Sales", y="Product line", orientation='h', title="Revenue by Product Line", color="Sales")
        st.plotly_chart(fig_prod, use_container_width=True)

# --- TAB 2: SALES DEEP-DIVE ---
with tab_sales:
    st.subheader("Comprehensive Sales Breakdown")
    s_col1, s_col2 = st.columns(2)
    
    with s_col1:
        # Day wise
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        sales_day = df_selection.groupby("Day")["Sales"].sum().reindex(day_order).reset_index()
        st.plotly_chart(px.bar(sales_day, x="Day", y="Sales", title="Sales by Day of Week"), use_container_width=True)
        
        # Branch wise
        st.plotly_chart(px.bar(df_selection.groupby("Branch")["Sales"].sum().reset_index(), x="Branch", y="Sales", title="Sales by Branch", color="Branch"), use_container_width=True)

    with s_col2:
        # Payment wise
        st.plotly_chart(px.pie(df_selection, values="Sales", names="Payment", title="Sales by Payment Method"), use_container_width=True)
        
        # Gender wise
        st.plotly_chart(px.bar(df_selection.groupby("Gender")["Sales"].sum().reset_index(), x="Gender", y="Sales", title="Sales by Gender", color="Gender"), use_container_width=True)

# --- TAB 3: TAX ANALYSIS ---
with tab_tax:
    st.subheader("5% Tax Contribution Analysis")
    t_col1, t_col2 = st.columns(2)
    
    with t_col1:
        st.plotly_chart(px.bar(df_selection.groupby("Product line")["Tax 5%"].sum().reset_index(), x="Tax 5%", y="Product line", orientation='h', title="Tax by Product Line"), use_container_width=True)
        st.plotly_chart(px.bar(df_selection.groupby("Quantity")["Tax 5%"].sum().reset_index(), x="Quantity", y="Tax 5%", title="Tax by Unit Quantity"), use_container_width=True)

    with t_col2:
        st.plotly_chart(px.pie(df_selection, values="Tax 5%", names="City", title="Tax contribution by City", hole=0.4), use_container_width=True)
        st.plotly_chart(px.bar(df_selection.groupby("Payment")["Tax 5%"].sum().reset_index(), x="Payment", y="Tax 5%", title="Tax by Payment Method"), use_container_width=True)

# --- TAB 4: TIME SERIES ---
with tab_time:
    st.subheader("Time & Trend Analysis")
    
    # Daily Sales with 7-Day Moving Average
    daily_sales = df_selection.groupby("Date")["Sales"].sum().reset_index()
    daily_sales['Moving_Avg'] = daily_sales['Sales'].rolling(window=7).mean()
    
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['Sales'], name='Daily Sales', opacity=0.4))
    fig_ma.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['Moving_Avg'], name='7-Day Moving Average', line=dict(color='red', width=2)))
    fig_ma.update_layout(title="Sales Trend with Moving Average Smoothing")
    st.plotly_chart(fig_ma, use_container_width=True)

    # Hourly Peak Analysis
    hourly_sales = df_selection.groupby("Hour")["Sales"].sum().reset_index()
    st.plotly_chart(px.area(hourly_sales, x="Hour", y="Sales", title="Peak Shopping Hours (24H Format)"), use_container_width=True)

# --- TAB 5: FORECASTING ---
with tab_forecast:
    st.subheader("🚀 Future Sales Forecasting")
    f_days = st.slider("Select Forecast Horizon (Days):", 7, 60, 30)
    
    # Prepare Regression Data
    fs_data = df_selection.groupby('Date')['Sales'].sum().reset_index()
    fs_data['Date_Ord'] = fs_data['Date'].map(datetime.datetime.toordinal)
    
    X = fs_data[['Date_Ord']]
    y = fs_data['Sales']
    model = LinearRegression().fit(X, y)
    
    # Predict Future
    last_date = fs_data['Date'].max()
    future_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=f_days)
    future_ords = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    preds = model.predict(future_ords)
    
    forecast_df = pd.DataFrame({'Date': future_dates, 'Sales': preds})
    
    # Combined Plot
    fig_fore = go.Figure()
    fig_fore.add_trace(go.Scatter(x=fs_data['Date'], y=fs_data['Sales'], name='Historical Sales'))
    fig_fore.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Sales'], name='Predicted Sales', line=dict(dash='dash', color='green')))
    fig_fore.update_layout(title=f"Sales Forecast for Next {f_days} Days")
    st.plotly_chart(fig_fore, use_container_width=True)
    
    st.info(f"The model predicts a total of **${preds.sum():,.2f}** in revenue over the next {f_days} days.")

# DETAILED DATA LOG
with st.expander("📂 Access Detailed Transaction Records"):
    st.dataframe(df_selection)
