import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly

# Page configuration
st.set_page_config(page_title="Apple Sales Forecasting", layout="wide")

st.title("🍎 Apple Global Product Sales: Time Series Forecasting")
st.markdown("""
This app performs time-series forecasting on the synthetic Apple sales dataset. 
It aggregates daily revenue and predicts future trends using the **Prophet** model.
""")

# 1. Load Data
@st.cache_data
def load_data():
    # Adjust filename if necessary
    df = pd.read_csv('SuperMarket Analysis.csv')
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    return df

try:
    df = load_data()
    
    # 2. Sidebar Filters
    st.sidebar.header("Forecast Settings")
    category_list = ["All"] + list(df['category'].unique())
    selected_category = st.sidebar.selectbox("Select Product Category", category_list)
    
    period = st.sidebar.slider("Days to forecast into the future:", 30, 365, 90)

    # Filter data based on selection
    if selected_category != "All":
        plot_df = df[df['category'] == selected_category]
    else:
        plot_df = df

    # 3. Data Preprocessing for Prophet
    # Prophet requires columns 'ds' (date) and 'y' (value)
    ts_data = plot_df.groupby('sale_date')['revenue_usd'].sum().reset_index()
    ts_data.columns = ['ds', 'y']

    # 4. Show Historical Data
    st.subheader(f"Historical Daily Revenue: {selected_category}")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name="Actual Revenue"))
    fig_hist.update_layout(xaxis_title="Date", yaxis_title="Revenue (USD)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 5. Forecasting Logic
    if st.button("Run Forecast"):
        with st.spinner('Training model...'):
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            model.fit(ts_data)
            
            # Create future dates
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)
            
            # 6. Display Results
            st.subheader(f"Forecast for next {period} days")
            
            # Interactive Forecast Plot
            fig_forecast = plot_plotly(model, forecast)
            fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Revenue (USD)")
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Components (Trend, Weekly, Yearly)
            st.subheader("Forecast Components")
            st.markdown("This shows the underlying trend and seasonal patterns (e.g., holiday peaks).")
            fig_comp = plot_components_plotly(model, forecast)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Show Raw Forecast Table
            st.subheader("Forecasted Data (Preview)")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

except FileNotFoundError:
    st.error("CSV file not found. Please ensure 'apple_global_sales_dataset.csv' is in the project folder.")
except Exception as e:
    st.error(f"An error occurred: {e}")
