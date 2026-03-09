import streamlit as st
import pandas as pd
import plotly.express as px

# 1. PAGE SETUP
st.set_page_config(page_title="Executive Supermarket Dashboard", layout="wide", page_icon="📊")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    [data-testid="stMetricValue"] { font-size: 30px; color: #0083B8; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
    }
    h2 { color: #1f3b4d; border-bottom: 2px solid #0083B8; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA ENGINE
@st.cache_data
def load_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    # Robust time parsing for AM/PM formats
    df["Time_dt"] = pd.to_datetime(df["Time"], errors='coerce')
    df["Hour"] = df["Time_dt"].dt.hour
    return df

try:
    df = load_data()

    # 3. SIDEBAR FILTERS
    st.sidebar.title("🔍 Global Filters")
    city = st.sidebar.multiselect("Select City", options=df["City"].unique(), default=df["City"].unique())
    ctype = st.sidebar.multiselect("Customer Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())
    branch = st.sidebar.multiselect("Branch", options=df["Branch"].unique(), default=df["Branch"].unique())

    df_selection = df.query("City == @city & `Customer type` == @ctype & Branch == @branch")

    # 4. TOP KPI ROW
    st.title("🛒 Supermarket Business Intelligence")
    
    if not df_selection.empty:
        total_sales = df_selection["Sales"].sum()
        total_profit = df_selection["gross income"].sum()
        avg_rating = round(df_selection["Rating"].mean(), 1)
        total_qty = df_selection["Quantity"].sum()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Revenue", f"${total_sales:,.0f}")
        kpi2.metric("Gross Profit", f"${total_profit:,.2f}")
        kpi3.metric("Total Units Sold", f"{total_qty:,}")
        kpi4.metric("Avg Satisfaction", f"{avg_rating} / 10")

        st.markdown("---")

        # 5. FINANCIAL SECTION
        st.subheader("💰 Financial Performance")
        f_col1, f_col2 = st.columns([2, 1])
        
        with f_col1:
            # Sales Trend Line
            sales_by_date = df_selection.groupby("Date")[["Sales"]].sum().reset_index()
            fig_trend = px.line(sales_by_date, x="Date", y="Sales", title="Daily Revenue Trend", template="plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with f_col2:
            # Revenue by City Pie
            city_rev = df_selection.groupby("City")[["Sales"]].sum().reset_index()
            fig_city = px.pie(city_rev, values="Sales", names="City", hole=0.5, title="Revenue by City", color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_city, use_container_width=True)

        st.markdown("---")

        # 6. OPERATIONS SECTION
        st.subheader("📦 Operational Efficiency")
        o_col1, o_col2 = st.columns(2)
        
        with o_col1:
            # Sales by Product Line
            product_sales = df_selection.groupby("Product line")[["Sales"]].sum().sort_values("Sales")
            fig_prod = px.bar(product_sales, x="Sales", y=product_sales.index, orientation='h', title="Top Product Categories", color="Sales", color_continuous_scale="Blues")
            st.plotly_chart(fig_prod, use_container_width=True)
            
        with o_col2:
            # Hourly Peak Area Chart
            hourly_sales = df_selection.groupby("Hour")[["Sales"]].sum().reset_index()
            fig_hour = px.area(hourly_sales, x="Hour", y="Sales", title="Peak Shopping Hours", color_discrete_sequence=["#00CC96"])
            st.plotly_chart(fig_hour, use_container_width=True)

        st.markdown("---")

        # 7. CUSTOMER SECTION
        st.subheader("👤 Customer Insights")
        c_col1, c_col2 = st.columns(2)
        
        with c_col1:
            # Payment Methods
            fig_pay = px.bar(df_selection, x="Payment", y="Sales", color="Gender", barmode="group", title="Payment Preference by Gender")
            st.plotly_chart(fig_pay, use_container_width=True)
            
        with c_col2:
            # Rating Scatter
            prod_rating = df_selection.groupby("Product line")[["Rating"]].mean().reset_index()
            fig_rat = px.scatter(prod_rating, x="Product line", y="Rating", size="Rating", color="Rating", title="Avg Rating per Category", template="plotly_white")
            st.plotly_chart(fig_rat, use_container_width=True)

        # 8. DATA TABLE
        with st.expander("📝 View Detailed Transaction Log"):
            st.dataframe(df_selection.drop(columns=["Time_dt", "Hour"]))
    else:
        st.warning("Please adjust filters to view data.")

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
