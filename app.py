import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from prophet.plot import plot_plotly, plot_components_plotly

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Apple Sales BI Dashboard", layout="wide")

# Custom Professional CSS (Sans-serif and clean)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #e1e4e8; }
    h1, h2, h3 { color: #1a1a1a; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("Apple Global Sales: Long-Term BI Forecasting")
st.markdown("Professional time-series analysis and revenue projections up to 10 years based on historical trends.")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    file_path = 'SuperMarket Analysis.csv' 
    try:
        # Detects Tab vs Comma automatically
        df = pd.read_csv(file_path, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        
        # Identify necessary columns
        date_col = next((c for c in ['sale_date', 'date', 'transaction_date'] if c in df.columns), None)
        rev_col = next((c for c in ['revenue_usd', 'total', 'revenue', 'gross income'] if c in df.columns), None)
        cat_col = next((c for c in ['category', 'product line'] if c in df.columns), df.columns[-1])

        df[date_col] = pd.to_datetime(df[date_col])
        return df, date_col, rev_col, cat_col
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

df, date_col, rev_col, cat_col = load_data()

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("Dashboard Controls")
categories = ["All Products"] + sorted(df[cat_col].unique().tolist())
selected_cat = st.sidebar.selectbox("Filter Category", categories)

# Forecast Period Slider (Up to 10 years / 3650 days)
st.sidebar.subheader("Forecasting Horizon")
period = st.sidebar.slider("Days to forecast (3650 = 10 Years):", 365, 3650, 1825)

# Filter Data
plot_df = df if selected_cat == "All Products" else df[df[cat_col] == selected_cat]

# --- 3. TOP KPI METRICS ---
m1, m2, m3, m4 = st.columns(4)
total_rev = plot_df[rev_col].sum()
avg_val = plot_df[rev_col].mean()
max_date = plot_df[date_col].max().year

m1.metric("Historical Revenue", f"${total_rev/1e6:.2f}M")
m2.metric("Avg Transaction Value", f"${avg_val:.2f}")
m3.metric("Last Data Point", f"Year {max_date}")
m4.metric("Total Records", len(plot_df))

# --- 4. FORECASTING ENGINE ---
# Prepare data for Prophet
ts_data = plot_df.groupby(date_col)[rev_col].sum().reset_index()
ts_data.columns = ['ds', 'y']

# Train Model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(ts_data)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# --- 5. ANNUAL PERFORMANCE BAR CHART (ACTUAL VS PREDICTED) ---
st.markdown("---")
st.header("Annual Revenue Performance Comparison")
st.info("The bar chart below compares actual historical annual totals with long-term projected growth.")

# Grouping by Year for clear 10-year visualization
hist_annual = ts_data.copy()
hist_annual['year'] = hist_annual['ds'].dt.year
hist_annual = hist_annual.groupby('year')['y'].sum().reset_index()

fore_annual = forecast[['ds', 'yhat']].copy()
fore_annual['year'] = fore_annual['ds'].dt.year
fore_annual = fore_annual.groupby('year')['yhat'].sum().reset_index()

# Merge for comparison
comparison_df = pd.merge(fore_annual, hist_annual, on='year', how='left')
comparison_df.columns = ['Year', 'Predicted Revenue', 'Actual Revenue']

# Create Interactive Bar Chart
fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(
    x=comparison_df['Year'],
    y=comparison_df['Actual Revenue'],
    name='Actual (Historical Data)',
    marker_color='#0071e3'
))

fig_bar.add_trace(go.Bar(
    x=comparison_df['Year'],
    y=comparison_df['Predicted Revenue'],
    name='Projected (Future Prediction)',
    marker_color='#ccd6dd'
))

fig_bar.update_layout(
    barmode='group',
    template='plotly_white',
    xaxis_title="Year",
    yaxis_title="Revenue (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- 6. LONG TERM TRAJECTORY (LINE VIEW) ---
st.markdown("---")
st.header("10-Year Growth Trajectory")
fig_line = plot_plotly(model, forecast)
fig_line.update_layout(template='plotly_white', title="")
st.plotly_chart(fig_line, use_container_width=True)

# --- 7. YEARLY BREAKDOWN TABLE & SUMMARY ---
st.markdown("---")
st.header("Annual Projected Revenue Summary")
st.write("Below is the predicted revenue for specific future milestones based on current growth trends.")

# Calculate future annual totals for specific years
summary_years = [max_date + 1, max_date + 3, max_date + 5, max_date + 7, max_date + 10]
summary_data = []

for yr in summary_years:
    yearly_val = fore_annual[fore_annual['year'] == yr]['yhat'].values
    val = f"${yearly_val[0]/1e6:.2f} Million" if len(yearly_val) > 0 else "Out of slider range"
    summary_data.append({"Milestone": f"After {yr - max_date} Year(s)", "Year": yr, "Estimated Revenue": val})

summary_table = pd.DataFrame(summary_data)

c1, c2 = st.columns([1, 2])
with c1:
    st.table(summary_table)
    st.warning("Note: Predictions beyond 5 years are based on trend continuity and carry higher uncertainty.")

with c2:
    # Seasonal Analysis
    st.subheader("Seasonal Trends Analysis")
    fig_comp = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_comp, use_container_width=True)

st.caption("Developed by Sales Intelligence BI | Engine: Prophet Time Series Prediction")
