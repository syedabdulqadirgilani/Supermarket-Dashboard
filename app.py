import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# 1. PAGE SETUP
st.set_page_config(page_title="Executive Supermarket Dashboard", layout="wide", page_icon="📊")

# Custom CSS for UI styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    [data-testid="stMetricValue"] { font-size: 30px; color: #0083B8; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
    }
    h2 { color: #1f3b4d; border-bottom: 2px solid #0083B8; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA ENGINE
@st.cache_data
def load_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    # Converting 'Time' (e.g., 1:08:00 PM) to an actual datetime object for extraction
    df["Time_dt"] = pd.to_datetime(df["Time"], format='%I:%M:%S %p', errors='coerce')
    df["Hour"] = df["Time_dt"].dt.hour
    df["Day"] = df["Date"].dt.day_name()
    return df

try:
    df = load_data()

    # 3. SIDEBAR FILTERS
    st.sidebar.title("🔍 Global Filters")
    city = st.sidebar.multiselect("Select City", options=df["City"].unique(), default=df["City"].unique())
    ctype = st.sidebar.multiselect("Customer Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())
    branch = st.sidebar.multiselect("Branch", options=df["Branch"].unique(), default=df["Branch"].unique())
    gender = st.sidebar.multiselect("Gender", options=df["Gender"].unique(), default=df["Gender"].unique())

    # THE FIX: Added backticks around Customer type
    df_selection = df.query(
        "City == @city & Customer type == @ctype & Branch == @branch & Gender == @gender"
    )

    # 4. DASHBOARD TABS
    st.title("🛒 Supermarket Business Intelligence")
    tab1, tab2, tab3 = st.tabs(["📊 Performance Overview", "📈 Trends & Forecasting", "🧾 Tax & Detailed Data"])

    if not df_selection.empty:
        # --- TAB 1: OVERVIEW ---
        with tab1:
            total_sales = df_selection["Sales"].sum()
            total_profit = df_selection["gross income"].sum()
            avg_rating = round(df_selection["Rating"].mean(), 1)
            total_qty = df_selection["Quantity"].sum()

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Revenue", f"${total_sales:,.0f}")
            kpi2.metric("Gross Profit", f"${total_profit:,.2f}")
            kpi3.metric("Total Units Sold", f"{total_qty:,}")
            kpi4.metric("Avg Satisfaction", f"{avg_rating} / 10")

            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                # Sales by Product Line
                product_sales = df_selection.groupby("Product line")[["Sales"]].sum().sort_values("Sales")
                fig_prod = px.bar(product_sales, x="Sales", y=product_sales.index, orientation='h', title="Top Product Categories", color="Sales", color_continuous_scale="Blues")
                st.plotly_chart(fig_prod, use_container_width=True)
            with col2:
                # Revenue by City Pie
                city_rev = df_selection.groupby("City")[["Sales"]].sum().reset_index()
                fig_city = px.pie(city_rev, values="Sales", names="City", hole=0.5, title="Revenue by City")
                st.plotly_chart(fig_city, use_container_width=True)

        # --- TAB 2: TRENDS & FORECASTING ---
        with tab2:
            st.subheader("Time Series & 30-Day Forecast")
            
            # Historical Trend
            daily_sales = df_selection.groupby("Date")["Sales"].sum().reset_index()
            fig_trend = px.line(daily_sales, x="Date", y="Sales", title="Daily Revenue Trend")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Simple Forecast Logic
            daily_sales['Date_Ord'] = daily_sales['Date'].map(datetime.datetime.toordinal)
            X = daily_sales[['Date_Ord']]
            y = daily_sales['Sales']
            model = LinearRegression().fit(X, y)
            
            last_date = daily_sales['Date'].max()
            future_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=30)
            future_ords = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            preds = model.predict(future_ords)
            
            fig_fore = go.Figure()
            fig_fore.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['Sales'], name='Actual'))
            fig_fore.add_trace(go.Scatter(x=future_dates, y=preds, name='Forecast', line=dict(dash='dash', color='red')))
            st.plotly_chart(fig_fore, use_container_width=True)

        # --- TAB 3: TAX & DATA ---
        with tab3:
            st.subheader("Tax Analysis (5%)")
            tax_col1, tax_col2 = st.columns(2)
            with tax_col1:
                tax_by_pay = df_selection.groupby("Payment")[["Tax 5%"]].sum().reset_index()
                st.plotly_chart(px.bar(tax_by_pay, x="Payment", y="Tax 5%", title="Tax by Payment Method"), use_container_width=True)
            with tax_col2:
                tax_by_gen = df_selection.groupby("Gender")[["Tax 5%"]].sum().reset_index()
                st.plotly_chart(px.pie(tax_by_gen, values="Tax 5%", names="Gender", title="Tax by Gender"), use_container_width=True)
            
            with st.expander("📝 View Detailed Transaction Log"):
                st.dataframe(df_selection.drop(columns=["Time_dt", "Hour", "Day"]))

    else:
        st.warning("Please adjust filters to view data.")

except Exception as e:
    st.error(f"An error occurred: {e}")
