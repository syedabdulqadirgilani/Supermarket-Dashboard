import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

# Dashboard ko wide screen mode par set karna
st.set_page_config(page_title="Supermarket Dashboard", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Aapki file read ho rahi hai
    df = pd.read_csv("SuperMarket Analysis.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()

    # --- SIDEBAR FILTERS & BUTTONS ---
    st.sidebar.title("Control Panel")
    
    # City Filter
    city = st.sidebar.multiselect(
        "Shehar chunein (Select City):",
        options=df["City"].unique(),
        default=df["City"].unique()
    )

    # Product Filter
    product = st.sidebar.multiselect(
        "Product line chunein:",
        options=df["Product line"].unique(),
        default=df["Product line"].unique()
    )

    # Prediction Slider
    days_to_predict = st.sidebar.slider("Kitne din ki prediction chahiye?", 1, 30, 7)

    # Action Buttons in Sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Filters"):
        st.rerun() # Is button se sab kuch pehle jaisa ho jayega

    # --- DATA FILTERING ---
    df_selection = df.query("City == @city & `Product line` == @product")

    # --- MAIN DASHBOARD AREA ---
    st.title("📊 Supermarket Dashboard")
    
    # Top Row: KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sales", f"${int(df_selection['Sales'].sum()):,}")
    m2.metric("Average Rating", f"{round(df_selection['Rating'].mean(), 1)} ⭐")
    m3.metric("Transactions", len(df_selection))
    m4.metric("Avg Unit Price", f"${round(df_selection['Unit price'].mean(), 2)}")

    st.markdown("---")

    # --- MIDDLE AREA: BUTTONS FOR ACTION ---
    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        # 1. Download Button: Filtered data ko CSV mein save karne ke liye
        csv = df_selection.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name='filtered_sales_data.csv',
            mime='text/csv',
        )

    with col_b:
        # 2. Interactive Info Button
        if st.button("📝 Show Quick Summary"):
            top_product = df_selection.groupby("Product line")["Sales"].sum().idxmax()
            st.info(f"Summary: Sabse zyada sales '{top_product}' ki hui hain.")

    with col_c:
        # 3. Warning/Goal Button
        if st.button("🎯 Check Sales Target"):
            if df_selection['Sales'].sum() > 5000:
                st.success("Mubarak ho! Sales target mukammal ho gaya.")
            else:
                st.warning("Sales target abhi door hai.")

    st.markdown("---")

    # --- CHARTS AREA ---
    left, right = st.columns(2)
    
    daily_sales = df_selection.groupby('Date')['Sales'].sum().reset_index().sort_values('Date')

    with left:
        st.subheader("Sales Trend Over Time")
        st.line_chart(daily_sales.set_index('Date'))

    with right:
        st.subheader("Payment Methods Comparison")
        pay_method = df_selection.groupby("Payment")["Sales"].sum()
        st.bar_chart(pay_method)

    st.markdown("---")

    # --- FORECASTING AREA ---
    st.subheader(f"🔮 Future Forecast (Next {days_to_predict} Days)")
    
    if len(daily_sales) > 1:
        # Simple Math (No AI)
        daily_sales['Day_Number'] = np.arange(len(daily_sales))
        slope, intercept = np.polyfit(daily_sales['Day_Number'], daily_sales['Sales'], 1)

        last_date = daily_sales['Date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
        future_day_numbers = np.arange(len(daily_sales), len(daily_sales) + days_to_predict)
        predicted_sales = slope * future_day_numbers + intercept

        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': predicted_sales})

        c1, c2 = st.columns([1, 2])
        with c1:
            st.write(forecast_df)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(daily_sales['Date'], daily_sales['Sales'], label="Actual", marker='.')
            ax.plot(forecast_df['Date'], forecast_df['Predicted Sales'], label="Forecast", color='red', ls='--')
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
    else:
        st.error("Forecast ke liye data kam hai. Filters check karein.")

except Exception as e:
    st.error("CSV File nahi mili ya koi masla hai.")
    st.write(e)