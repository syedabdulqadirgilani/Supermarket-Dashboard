import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from prophet.plot import plot_plotly, plot_components_plotly

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Apple Sales BI Dashboard", layout="wide", page_icon="🍎")

# Custom Apple-Style CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f7; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1d1d1f; font-family: 'SF Pro Display', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🍎 Apple Global Sales: BI Forecasting & Analytics")
st.markdown("Interactive comparison of **Actual vs Predicted** revenue trends for 2025-2026.")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    file_path = 'SuperMarket Analysis.csv' 
    try:
        # Detects Tab vs Comma automatically
        df = pd.read_csv(file_path, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        
        # Identify columns
        date_col = next((c for c in ['sale_date', 'date', 'transaction_date'] if c in df.columns), None)
        rev_col = next((c for c in ['revenue_usd', 'total', 'revenue', 'gross income'] if c in df.columns), None)
        cat_col = next((c for c in ['category', 'product line'] if c in df.columns), df.columns[-1])

        df[date_col] = pd.to_datetime(df[date_col])
        return df, date_col, rev_col, cat_col
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

df, date_col, rev_col, cat_col = load_data()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Dashboard Controls")
categories = ["All Products"] + sorted(df[cat_col].unique().tolist())
selected_cat = st.sidebar.selectbox("Filter Category", categories)

# Forecast Period (Covering 2026)
period = st.sidebar.slider("Forecast Horizon (Days):", 30, 1095, 730)

# Filter Data
plot_df = df if selected_cat == "All Products" else df[df[cat_col] == selected_cat]

# --- 3. TOP KPI METRICS ---
m1, m2, m3, m4 = st.columns(4)
total_rev = plot_df[rev_col].sum()
avg_val = plot_df[rev_col].mean()
last_date = plot_df[date_col].max().strftime('%Y-%m-%d')

m1.metric("Historical Revenue", f"${total_rev/1e6:.2f}M")
m2.metric("Avg Order Value", f"${avg_val:.2f}")
m3.metric("Data Range End", last_date)
m4.metric("Transactions", len(plot_df))

# --- 4. THE ANALYTICS ENGINE (Prophet) ---
# Prepare data
ts_data = plot_df.groupby(date_col)[rev_col].sum().reset_index()
ts_data.columns = ['ds', 'y']

# Train Model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(ts_data)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# --- 5. INTERACTIVE BAR CHART (ACTUAL VS PREDICTED) ---
st.markdown("---")
st.header("📊 Monthly Performance Breakdown (Actual vs Forecast)")
st.info("Niche diya gaya Bar Graph aapko monthly comparison dikhayega. 2024 tak 'Actual' hai aur uske aage 'Predicted'.")

# Process for Monthly Bar Chart
# 1. Monthly Actuals
hist_monthly = ts_data.copy()
hist_monthly['month'] = hist_monthly['ds'].dt.to_period('M').dt.to_timestamp()
hist_monthly = hist_monthly.groupby('month')['y'].sum().reset_index()

# 2. Monthly Forecast
fore_monthly = forecast[['ds', 'yhat']].copy()
fore_monthly['month'] = fore_monthly['ds'].dt.to_period('M').dt.to_timestamp()
fore_monthly = fore_monthly.groupby('month')['yhat'].sum().reset_index()

# 3. Merge for plotting
comparison_df = pd.merge(fore_monthly, hist_monthly, on='month', how='left')
comparison_df.columns = ['Month', 'Predicted Revenue', 'Actual Revenue']

# Create Interactive Bar Chart
fig_bar = go.Figure()

# Actual Bars (Blue)
fig_bar.add_trace(go.Bar(
    x=comparison_df['Month'],
    y=comparison_df['Actual Revenue'],
    name='Actual Sales (Historical)',
    marker_color='#0071e3'
))

# Predicted Bars (Light Silver/Gray)
fig_bar.add_trace(go.Bar(
    x=comparison_df['Month'],
    y=comparison_df['Predicted Revenue'],
    name='Predicted Sales (Future)',
    marker_color='#d2d2d7',
    opacity=0.8
))

fig_bar.update_layout(
    barmode='group',
    template='plotly_white',
    xaxis_title="Timeline (Months)",
    yaxis_title="Revenue (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- 6. TIME SERIES LINE CHART ---
st.markdown("---")
st.header("📈 Growth Trajectory (Line View)")
fig_line = plot_plotly(model, forecast)
fig_line.update_layout(template='plotly_white', title="")
st.plotly_chart(fig_line, use_container_width=True)

# --- 7. 2026 DATA INSIGHTS ---
st.markdown("---")
st.header("📅 2026 Deep Dive Preview")
c1, c2 = st.columns([1, 2])

# Filter for 2026 only
forecast_2026 = forecast[forecast['ds'].dt.year == 2026]

with c1:
    total_2026 = forecast_2026['yhat'].sum()
    st.metric("Total Projected Revenue (2026)", f"${total_2026/1e6:.2f}M")
    st.success("2026 mein sales seasonal holidays ke dauran peak hone ka imkan hai.")

with c2:
    # Components chart (Trends)
    fig_comp = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_comp, use_container_width=True)

st.caption("Developed by Apple Sales Intelligence BI | Engine: Facebook Prophet")
