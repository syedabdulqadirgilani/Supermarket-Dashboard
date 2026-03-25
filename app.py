import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly

st.set_page_config(page_title="Apple Sales Forecasting", layout="wide")
st.title("🍎 Apple Global Product Sales: Time Series Forecasting")

@st.cache_data
def load_data():
    # 1. Load the data
    df = pd.read_csv('SuperMarket Analaysis.csv')
    
    # 2. CLEAN HEADERS: Remove hidden spaces and make everything lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # 3. CONVERT DATE: Ensure the date column is in datetime format
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
    else:
        # Fallback: if 'sale_date' is missing, show the user what columns ARE there
        st.error(f"Could not find 'sale_date'. Available columns are: {list(df.columns)}")
        st.stop()
        
    return df

try:
    df = load_data()
    
    st.sidebar.header("Forecast Settings")
    # Clean the category selection as well
    category_col = 'category' if 'category' in df.columns else df.columns[-1]
    category_list = ["All"] + list(df[category_col].unique())
    selected_category = st.sidebar.selectbox("Select Product Category", category_list)
    
    period = st.sidebar.slider("Days to forecast:", 30, 365, 90)

    # Filter
    if selected_category != "All":
        plot_df = df[df[category_col] == selected_category]
    else:
        plot_df = df

    # Prepare for Prophet (Prophet needs 'ds' and 'y')
    # We aggregate 'revenue_usd' or whatever the revenue column is named
    revenue_col = 'revenue_usd' if 'revenue_usd' in df.columns else 'revenue'
    
    ts_data = plot_df.groupby('sale_date')[revenue_col].sum().reset_index()
    ts_data.columns = ['ds', 'y']

    st.subheader(f"Historical Trends: {selected_category}")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=ts_data['ds'], y=ts_data['y'], name="Actual Revenue"))
    st.plotly_chart(fig_hist, use_container_width=True)

    if st.button("Run Forecast"):
        with st.spinner('Calculating...'):
            model = Prophet(yearly_seasonality=True, interval_width=0.95)
            model.fit(ts_data)
            
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)
            
            st.subheader("Forecast Results")
            fig_forecast = plot_plotly(model, forecast)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            st.subheader("Seasonal Patterns (Yearly/Weekly)")
            fig_comp = plot_components_plotly(model, forecast)
            st.plotly_chart(fig_comp, use_container_width=True)

except Exception as e:
    st.error(f"Critical Error: {e}")
    st.info("Check if your CSV file name is exactly 'apple_global_sales_dataset.csv'")
