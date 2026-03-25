import streamlit as st
import pandas as pd
import plotly.express as px

# 1. PAGE SETUP
st.set_page_config(page_title="Global Supermarket Intelligence", layout="wide", page_icon="📊")

# Custom CSS for UI
st.markdown("""
    <style>
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time_dt"] = pd.to_datetime(df["Time"], errors='coerce')
    df["Hour"] = df["Time_dt"].dt.hour
    return df

try:
    df = load_and_clean_data()

    # 3. GLOBAL SIDEBAR FILTERS
    st.sidebar.title("🕹️ Control Panel")
    
    # --- ADDED: Date Range Filter ---
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    date_range = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    city = st.sidebar.multiselect("📍 Select Cities", options=df["City"].unique(), default=df["City"].unique())
    customer_type = st.sidebar.multiselect("👥 Customer Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())
    branch = st.sidebar.multiselect("🏬 Branch", options=df["Branch"].unique(), default=df["Branch"].unique())

    # Apply Filters
    mask = (df["City"].isin(city)) & \
           (df["Customer type"].isin(customer_type)) & \
           (df["Branch"].isin(branch)) & \
           (df["Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])))
    
    df_selection = df[mask]

    st.title("📊 Supermarket Enterprise Dashboard")
    st.write(f"Analyzing {len(df_selection)} transactions globally.")

    if df_selection.empty:
        st.error("No data found! Adjust your filters.")
    else:
        # 5. KPI SUMMARY ROW (Updated with AOV)
        total_sales = df_selection["Sales"].sum()
        total_profit = df_selection["gross income"].sum()
        avg_rating = round(df_selection["Rating"].mean(), 1)
        total_trans = len(df_selection)
        aov = total_sales / total_trans if total_trans > 0 else 0

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Revenue", f"${total_sales:,.0f}")
        kpi2.metric("Gross Profit", f"${total_profit:,.0f}")
        kpi3.metric("AOV (Avg Bill)", f"${aov:,.2f}") # --- ADDED KPI ---
        kpi4.metric("Avg Rating", f"{avg_rating} ⭐")
        kpi5.metric("Transactions", f"{total_trans:,}")

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Financials", "📦 Operations", "👤 Customers", "📋 Raw Data"])

        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                sales_by_date = df_selection.groupby("Date")[["Sales"]].sum().reset_index()
                fig_trend = px.line(sales_by_date, x="Date", y="Sales", title="Daily Sales Trend", template="plotly_white")
                st.plotly_chart(fig_trend, use_container_width=True)
            with col2:
                city_rev = df_selection.groupby("City")[["Sales"]].sum().reset_index()
                fig_city = px.pie(city_rev, values="Sales", names="City", hole=0.5, title="Revenue by City")
                st.plotly_chart(fig_city, use_container_width=True)

        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                product_sales = df_selection.groupby("Product line")[["Sales"]].sum().sort_values("Sales")
                fig_prod = px.bar(product_sales, x="Sales", y=product_sales.index, orientation='h', title="Product Performance", color="Sales", color_continuous_scale="Blues")
                st.plotly_chart(fig_prod, use_container_width=True)
            with col4:
                hourly_sales = df_selection.groupby("Hour")[["Sales"]].count().reset_index()
                fig_hour = px.area(hourly_sales, x="Hour", y="Sales", title="Peak Traffic Hours (Invoices)", color_discrete_sequence=["#00CC96"])
                st.plotly_chart(fig_hour, use_container_width=True)

        with tab3:
            col5, col6 = st.columns(2)
            with col5:
                fig_pay = px.bar(df_selection, x="Payment", y="Sales", color="Customer type", barmode="group", title="Payment Preferences")
                st.plotly_chart(fig_pay, use_container_width=True)
            with col6:
                # Improved: Using Bar chart for ratings instead of scatter
                prod_rating = df_selection.groupby("Product line")[["Rating"]].mean().sort_values("Rating").reset_index()
                fig_rat = px.bar(prod_rating, x="Rating", y="Product line", orientation='h', title="Avg Rating per Category", color="Rating", range_x=[0,10])
                st.plotly_chart(fig_rat, use_container_width=True)

        with tab4:
            st.subheader("Filtered Data")
            # --- ADDED: Download CSV Button ---
            csv = df_selection.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Filtered Data as CSV", data=csv, file_name="supermarket_data.csv", mime="text/csv")
            st.dataframe(df_selection, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
