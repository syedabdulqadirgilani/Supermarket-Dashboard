import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly

# Page configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 Sales Time Series Forecasting")
st.markdown("""
This application uses the **Prophet** model to analyze historical sales data and predict future revenue trends.
""")

# 1. Robust Data Loading Function
@st.cache_data
def load_data():
    # Ensure this filename matches the one in your GitHub repository
    file_path = 'SuperMarket Analysis.csv' 
    
    try:
        # sep=None and engine='python' allows pandas to auto-detect if the file uses Commas or Tabs (\t)
        df = pd.read_csv(file_path, sep=None, engine='python')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    # Clean headers: remove spaces and convert to lowercase for easier matching
    df.columns = df.columns.str.strip().str.lower()
    
    # --- AUTO-DETECT DATE COLUMN ---
    date_col = None
    for col in ['sale_date', 'date', 'transaction_date', 'timestamp']:
        if col in df.columns:
            date_col = col
            break
            
    # --- AUTO-DETECT REVENUE/VALUE COLUMN ---
    rev_col = None
    for col in ['revenue_usd', 'total', 'revenue', 'gross income', 'sales']:
        if col in df.columns:
            rev_col = col
            break

    # If detection fails, show exactly what was found so you can troubleshoot
    if not date_col or not rev_col:
        st.error("Column Detection Failed!")
        st.write("Found these columns in your file:", list(df.columns))
        st.info("The app expects a Date column and a Revenue/Total column.")
        st.stop()

    # Convert the date column to actual datetime objects
    df[date_col] = pd.to_datetime(df[date_col])
    
    return df, date_col, rev_col

try:
    df, date_col, rev_col = load_data()
    
    # 2. Sidebar Filters
    st.sidebar.header("Forecast Configuration")
    
    # Find a category column (looks for 'category', 'product line', or uses the last string column)
    cat_col = 'category' if 'category' in df.columns else ('product line' if 'product line' in df.columns else None)
    
    if cat_col:
        category_list = ["All Categories"] + sorted(list(df[cat_col].unique()))
        selected_category = st.sidebar.selectbox("Filter by Category", category_list)
    else:
        selected_category = "All Categories"

    period = st.sidebar.slider("Days to forecast into the future:", 30, 365, 90)

    # 3. Filter and Aggregate Data
    if selected_category != "All Categories":
        filtered_df = df[df[cat_col] == selected_category]
    else:
        filtered_df = df

    # Prophet requires a specific format: 'ds' for date and 'y' for values
    ts_data = filtered_df.groupby(date_col)[rev_col].sum().reset_index()
    ts_data.columns = ['ds', 'y']

    # 4. Display Historical Chart
    st.subheader(f"Historical Revenue: {selected_category}")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name="Actual Revenue", line=dict(color='#1f77b4')))
    fig_hist.update_layout(xaxis_title="Date", yaxis_title="Revenue (USD)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 5. Forecasting Logic
    if st.button("🚀 Run Forecast"):
        with st.spinner('Training forecasting model...'):
            # Initialize and fit the Prophet model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(ts_data)
            
            # Create a future dataframe for the number of days selected
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)
            
            # 6. Display Forecast Results
            st.subheader(f"Forecast for the next {period} days")
            fig_forecast = plot_plotly(model, forecast)
            fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Revenue (USD)")
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # 7. Seasonal Breakdown (Trends, Weekly, Yearly)
            st.subheader("Seasonal Patterns & Trends")
            st.markdown("This section breaks down when your sales peak (e.g., weekends vs weekdays, or certain months).")
            fig_comp = plot_components_plotly(model, forecast)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Preview the last few days of the forecast
            st.subheader("Forecast Data Preview")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
