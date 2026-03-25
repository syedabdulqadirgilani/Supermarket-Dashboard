import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly

st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("📊 Sales Time Series Forecasting")

@st.cache_data
def load_data():
    # Load the specific file in your repo
    df = pd.read_csv('SuperMarket Analysis.csv')
    
    # Clean headers (lowercase and remove spaces)
    df.columns = df.columns.str.strip().str.lower()
    
    # --- AUTO-DETECT COLUMNS ---
    # Detect Date Column (checks for 'sale_date' or just 'date')
    date_col = None
    for col in ['sale_date', 'date', 'transaction_date']:
        if col in df.columns:
            date_col = col
            break
            
    # Detect Revenue Column (checks for 'revenue_usd', 'total', or 'revenue')
    rev_col = None
    for col in ['revenue_usd', 'total', 'revenue', 'gross income']:
        if col in df.columns:
            rev_col = col
            break

    if not date_col or not rev_col:
        st.error(f"Missing required columns! Found: {list(df.columns)}")
        st.info("Ensure your CSV has a column for Date and a column for Revenue/Total.")
        st.stop()

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    return df, date_col, rev_col

try:
    df, date_col, rev_col = load_data()
    
    st.sidebar.header("Forecast Settings")
    
    # Detect Category Column
    cat_col = 'category' if 'category' in df.columns else ('product line' if 'product line' in df.columns else df.columns[-1])
    
    category_list = ["All"] + list(df[cat_col].unique())
    selected_category = st.sidebar.selectbox("Select Filter", category_list)
    period = st.sidebar.slider("Days to forecast:", 30, 365, 90)

    # Filter Data
    plot_df = df if selected_category == "All" else df[df[cat_col] == selected_category]

    # Prepare for Prophet (needs 'ds' and 'y')
    ts_data = plot_df.groupby(date_col)[rev_col].sum().reset_index()
    ts_data.columns = ['ds', 'y']

    st.subheader(f"Historical Trend: {selected_category}")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name="Actual"))
    st.plotly_chart(fig_hist, use_container_width=True)

    if st.button("Generate Forecast"):
        with st.spinner('Analyzing patterns...'):
            # Prophet model setup
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            model.fit(ts_data)
            
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)
            
            st.subheader(f"Prediction for next {period} days")
            fig_forecast = plot_plotly(model, forecast)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            st.subheader("Seasonal Breakdown")
            fig_comp = plot_components_plotly(model, forecast)
            st.plotly_chart(fig_comp, use_container_width=True)

except Exception as e:
    st.error(f"Critical Error: {e}")
