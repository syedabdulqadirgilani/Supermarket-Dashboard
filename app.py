import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import datetime

# 1. PAGE SETUP
st.set_page_config(page_title="30-Day Sales Forecasting Dashboard", layout="wide", page_icon="🔮")

# Custom CSS for modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 32px; color: #1E88E5; font-weight: bold; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    }
    h1, h2, h3 { color: #0D47A1; }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA LOADING & PREPROCESSING
@st.cache_data
def load_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

try:
    df = load_data()

    # 3. GLOBAL FILTERS
    st.sidebar.title("🛠️ Forecasting Controls")
    st.sidebar.info("This dashboard uses Linear Regression to predict trends based on historical data.")
    
    # Filter by Branch/Customer Type for specific context
    branch_filter = st.sidebar.multiselect("Select Branch (Optional)", options=df["Branch"].unique(), default=df["Branch"].unique())
    df_filtered = df[df["Branch"].isin(branch_filter)]

    # 4. FORECASTING ENGINE
    # We define a function to predict future values for any specific grouping
    def get_forecast(dataframe, group_col=None, forecast_days=30):
        # Aggregate daily sales
        if group_col:
            daily_data = dataframe.groupby(['Date', group_col])['Sales'].sum().reset_index()
            unique_items = daily_data[group_col].unique()
        else:
            daily_data = dataframe.groupby('Date')['Sales'].sum().reset_index()
            unique_items = [None]

        last_date = daily_data['Date'].max()
        future_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=forecast_days)
        future_ords = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        
        forecast_results = []

        for item in unique_items:
            if item:
                subset = daily_data[daily_data[group_col] == item]
            else:
                subset = daily_data
            
            # Prepare X and y
            subset['Date_Ord'] = subset['Date'].map(datetime.datetime.toordinal)
            X = subset[['Date_Ord']]
            y = subset['Sales']
            
            if len(X) > 1: # Need at least 2 points to draw a line
                model = LinearRegression().fit(X, y)
                preds = model.predict(future_ords)
                # Clip negative values to 0 (sales can't be negative)
                preds = np.maximum(preds, 0)
                
                temp_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Sales': preds})
                if item:
                    temp_df[group_col] = item
                forecast_results.append(temp_df)
        
        return pd.concat(forecast_results) if forecast_results else pd.DataFrame()

    # Generate Forecasts
    total_forecast = get_forecast(df_filtered)
    city_forecast = get_forecast(df_filtered, group_col="City")
    product_forecast = get_forecast(df_filtered, group_col="Product line")

    # 5. DASHBOARD LAYOUT
    st.title("🔮 30-Day Sales Forecasting Dashboard")
    st.markdown("Predictive analytics for inventory and revenue planning.")

    # --- ROW 1: KEY METRICS ---
    col_m1, col_m2, col_m3 = st.columns(3)
    
    total_predicted_revenue = total_forecast['Forecasted_Sales'].sum()
    top_city = city_forecast.groupby("City")["Forecasted_Sales"].sum().idxmax()
    top_prod = product_forecast.groupby("Product line")["Forecasted_Sales"].sum().idxmax()

    col_m1.metric("Total Predicted Revenue (30 Days)", f"${total_predicted_revenue:,.2f}")
    col_m2.metric("Top Predicted City", top_city)
    col_m3.metric("Top Predicted Product Category", top_prod)

    st.markdown("---")

    # --- ROW 2: TIME SERIES FORECAST ---
    st.subheader("📈 Time Series for 30-Day Forecast")
    
    historical_total = df_filtered.groupby("Date")["Sales"].sum().reset_index()
    
    fig_ts = go.Figure()
    # Historical Line
    fig_ts.add_trace(go.Scatter(x=historical_total['Date'], y=historical_total['Sales'], 
                                name='Historical Sales', line=dict(color='#1E88E5', width=2)))
    # Forecast Line
    fig_ts.add_trace(go.Scatter(x=total_forecast['Date'], y=total_forecast['Forecasted_Sales'], 
                                name='Predicted Sales (Next 30 Days)', line=dict(color='#D32F2F', width=3, dash='dash')))
    
    fig_ts.update_layout(template="plotly_white", hovermode="x unified", 
                         xaxis_title="Timeline", yaxis_title="Sales ($)")
    st.plotly_chart(fig_ts, use_container_width=True)

    # --- ROW 3: CATEGORICAL FORECASTS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🛍️ Top Product Sales Forecast")
        # Sum of predicted sales per product for the 30 day period
        prod_sums = product_forecast.groupby("Product line")["Forecasted_Sales"].sum().reset_index().sort_values("Forecasted_Sales", ascending=True)
        fig_p = px.bar(prod_sums, x="Forecasted_Sales", y="Product line", orientation='h',
                       title="Predicted Total Revenue by Product (Next 30 Days)",
                       color="Forecasted_Sales", color_continuous_scale="Blues")
        st.plotly_chart(fig_p, use_container_width=True)

    with col_right:
        st.subheader("🌆 Revenue by City Forecast")
        city_sums = city_forecast.groupby("City")["Forecasted_Sales"].sum().reset_index()
        fig_c = px.pie(city_sums, values="Forecasted_Sales", names="City", hole=0.4,
                       title="Predicted Revenue Contribution by City",
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_c, use_container_width=True)

    # --- DATA TABLE ---
    with st.expander("📝 View Forecast Data Table"):
        st.write("Below are the raw predicted values for the next 30 days:")
        st.dataframe(total_forecast.set_index('Date'))

except FileNotFoundError:
    st.error("Error: 'SuperMarket Analysis.csv' not found. Please ensure the file is in the same directory.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
