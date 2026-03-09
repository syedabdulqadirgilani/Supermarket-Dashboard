import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE SETUP
st.set_page_config(page_title="Global Supermarket Intelligence", layout="wide", page_icon="📊")

# Custom CSS for a professional look (Fixed the parameter name here)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stMetricValue"] { font-size: 28px; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA ENGINE
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    
    # Robust DateTime Conversion
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    
    # Handle both 12h (AM/PM) and 24h time formats
    df["Time_dt"] = pd.to_datetime(df["Time"], errors='coerce')
    df["Hour"] = df["Time_dt"].dt.hour
    
    # Month name for seasonal analysis
    df["Month"] = df["Date"].dt.month_name()
    return df

try:
    df = load_and_clean_data()

    # 3. GLOBAL SIDEBAR FILTERS
    st.sidebar.title("🕹️ Control Panel")
    
    city = st.sidebar.multiselect("📍 Select Cities", options=df["City"].unique(), default=df["City"].unique())
    customer_type = st.sidebar.multiselect("👥 Customer Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())
    gender = st.sidebar.multiselect("⚧ Gender", options=df["Gender"].unique(), default=df["Gender"].unique())
    branch = st.sidebar.multiselect("🏬 Branch", options=df["Branch"].unique(), default=df["Branch"].unique())

    # Apply all filters
    mask = (df["City"].isin(city)) & \
           (df["Customer type"].isin(customer_type)) & \
           (df["Gender"].isin(gender)) & \
           (df["Branch"].isin(branch))
    
    df_selection = df[mask]

    # 4. MAIN TITLE
    st.title("📊 Supermarket Enterprise Dashboard")
    st.write(f"Analyzing {len(df_selection)} transactions globally.")

    if df_selection.empty:
        st.error("No data found! Adjust your sidebar filters.")
    else:
        # 5. KPI SUMMARY ROW
        # Explicitly selecting numeric columns before sum to avoid Errors
        total_sales = df_selection["Sales"].sum()
        total_profit = df_selection["gross income"].sum()
        avg_rating = round(df_selection["Rating"].mean(), 1)
        total_units = df_selection["Quantity"].sum()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Revenue", f"${total_sales:,.0f}")
        kpi2.metric("Gross Profit", f"${total_profit:,.2f}")
        kpi3.metric("Avg Satisfaction", f"{avg_rating} / 10")
        kpi4.metric("Units Sold", f"{total_units:,} pcs")

        st.markdown("---")

        # 6. TABS FOR MULTI-LEVEL ANALYSIS
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Financials", "📦 Operations", "👤 Customers", "📋 Raw Data"])

        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                # Revenue Trends by Date
                sales_by_date = df_selection.groupby("Date")[["Sales"]].sum().reset_index()
                fig_trend = px.line(sales_by_date, x="Date", y="Sales", title="<b>Daily Sales Revenue Trend</b>", template="plotly_white")
                st.plotly_chart(fig_trend, use_container_width=True)
            with col2:
                # Revenue by City
                city_rev = df_selection.groupby("City")[["Sales"]].sum().reset_index()
                fig_city = px.pie(city_rev, values="Sales", names="City", hole=0.4, title="<b>Revenue Share by City</b>")
                st.plotly_chart(fig_city, use_container_width=True)

        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                # Sales by Product Line
                product_sales = df_selection.groupby("Product line")[["Sales"]].sum().sort_values("Sales")
                fig_prod = px.bar(product_sales, x="Sales", y=product_sales.index, orientation='h', title="<b>Product Category Performance</b>", color="Sales", color_continuous_scale="Viridis")
                st.plotly_chart(fig_prod, use_container_width=True)
            with col4:
                # Peak Hours
                hourly_sales = df_selection.groupby("Hour")[["Sales"]].sum().reset_index()
                fig_hour = px.area(hourly_sales, x="Hour", y="Sales", title="<b>Hourly Peak Traffic (24h)</b>", color_discrete_sequence=["#FF4B4B"])
                st.plotly_chart(fig_hour, use_container_width=True)

        with tab3:
            col5, col6 = st.columns(2)
            with col5:
                # Payment Methods
                fig_pay = px.bar(df_selection, x="Payment", y="Sales", color="Customer type", barmode="group", title="<b>Payment Method vs Customer Type</b>")
                st.plotly_chart(fig_pay, use_container_width=True)
            with col6:
                # Rating by Product Line
                prod_rating = df_selection.groupby("Product line")[["Rating"]].mean().reset_index()
                fig_rat = px.scatter(prod_rating, x="Product line", y="Rating", size="Rating", color="Rating", title="<b>Customer Rating by Category</b>")
                st.plotly_chart(fig_rat, use_container_width=True)

        with tab4:
            st.subheader("Filtered Transactional Records")
            # Drop the helper datetime columns before showing table
            st.dataframe(df_selection.drop(columns=["Time_dt", "Hour", "Month"]), use_container_width=True)

except Exception as e:
    st.error(f"Critical System Error: {e}")
