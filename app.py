import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime

# --- PAGE CONFIGURATION (Dashboard ka look professional banane ke liye) ---
st.set_page_config(page_title="Apple Sales BI Dashboard", layout="wide", page_icon="🍎")

# Custom CSS for a "Premium BI" feel
st.markdown("""
    <style>
    .main { background-color: #f5f5f7; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #1d1d1f; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("🍎 Apple Global Sales: BI Forecasting Dashboard")
st.markdown("Professional-grade time series analysis using the **Prophet** engine to predict 2025-2026 trends.")

# --- 1. DATA LOADING ENGINE (Auto-detecting separators and columns) ---
@st.cache_data
def load_data():
    file_path = 'SuperMarket Analysis.csv' # Ensure this file is in your GitHub
    try:
        # sep=None detects if file is Tab (\t) or Comma (,) separated automatically
        df = pd.read_csv(file_path, sep=None, engine='python')
        
        # Headers clean up (Lowercase and strip spaces)
        df.columns = df.columns.str.strip().str.lower()
        
        # Column Mapping (Finding Date and Revenue columns)
        date_col = next((c for c in ['sale_date', 'date', 'transaction_date'] if c in df.columns), None)
        rev_col = next((c for c in ['revenue_usd', 'total', 'revenue', 'gross income'] if c in df.columns), None)
        cat_col = next((c for c in ['category', 'product line', 'product_name'] if c in df.columns), None)

        if not date_col or not rev_col:
            st.error(f"Required columns not found! Columns found: {list(df.columns)}")
            st.stop()

        df[date_col] = pd.to_datetime(df[date_col])
        return df, date_col, rev_col, cat_col
    except Exception as e:
        st.error(f"File Load Error: {e}")
        st.stop()

df, date_col, rev_col, cat_col = load_data()

# --- 2. SIDEBAR - CONTROL PANEL ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=50)
st.sidebar.header("🕹️ Control Panel")

# Category Filter
categories = ["Global (All)"] + sorted(df[cat_col].unique().tolist())
selected_cat = st.sidebar.selectbox("Select Product Category", categories)

# Forecasting period (Set to 1095 to cover 2026 and 2027)
st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Horizon")
period = st.sidebar.slider("Days to predict (730 = Year 2026):", 30, 1095, 730)

# Filter Logic
if selected_cat == "Global (All)":
    plot_df = df
else:
    plot_df = df[df[cat_col] == selected_cat]

# --- 3. BI METRICS (TOP ROW KPIs) ---
total_rev = plot_df[rev_col].sum()
avg_sale = plot_df[rev_col].mean()
unique_products = plot_df[cat_col].nunique()

m1, m2, m3 = st.columns(3)
m1.metric("Total Historical Revenue", f"${total_rev:,.0f}")
m2.metric("Average Transaction Value", f"${avg_sale:,.2f}")
m3.metric("Product Categories", unique_products)

# --- 4. HISTORICAL TREND VISUALIZATION ---
st.subheader(f"📈 Historical Sales Performance: {selected_cat}")
# Daily aggregation for smooth plotting
daily_sales = plot_df.groupby(date_col)[rev_col].sum().reset_index()

fig_hist = px.line(daily_sales, x=date_col, y=rev_col, 
              title="Daily Revenue Trend (2022 - 2024)",
              color_discrete_sequence=['#0071e3'], # Apple Blue
              template="plotly_white")
fig_hist.update_layout(hovermode="x unified")
st.plotly_chart(fig_hist, use_container_width=True)

# --- 5. THE FORECASTING ENGINE (Prophet) ---
st.markdown("---")
st.header("🔮 Future Forecast (2025 - 2026)")

# Prophet requires 'ds' (date) and 'y' (value)
prophet_df = daily_sales.rename(columns={date_col: 'ds', rev_col: 'y'})

if st.button("🚀 Generate 2026 Forecast"):
    with st.spinner('Analyzing historical patterns and seasonality...'):
        # Model training
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        
        # Future dataframe creation
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)
        
        # PLOT 1: Interactive Forecast
        st.subheader(f"Projected Revenue Growth for {selected_cat}")
        fig_f = plot_plotly(model, forecast)
        fig_f.update_layout(template="plotly_white", title_font_size=20)
        st.plotly_chart(fig_f, use_container_width=True)
        
        # PLOT 2: Seasonal Components (Trends)
        st.subheader("🗓️ Seasonal Intelligence")
        st.info("The charts below show the underlying Trend (Growth), Weekly peaks, and Yearly seasonality (Holidays).")
        fig_comp = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # DATA TABLE: 2026 Specifics
        st.subheader("📅 2026 Data Preview")
        forecast_2026 = forecast[forecast['ds'] >= '2026-01-01']
        
        col_a, col_b = st.columns([1, 2])
        with col_a:
            avg_2026 = forecast_2026['yhat'].mean()
            st.metric("Predicted Avg Daily Revenue (2026)", f"${avg_2026:,.2f}")
        with col_b:
            st.dataframe(forecast_2026[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10), use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Dashboard generated by Apple Sales Intelligence | Data: SuperMarket Analysis.csv")
