import streamlit as st
import pandas as pd
import plotly.express as px

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Supermarket Analytics Dashboard", layout="wide", page_icon="🛒")

# 2. LOAD DATA
@st.cache_data
def get_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    # Convert Date to datetime object
    df["Date"] = pd.to_datetime(df["Date"])
    # Create an Hour column safely
    df["Hour"] = pd.to_datetime(df["Time"], format='%H:%M:%S').dt.hour if df["Time"].dtype == 'O' else pd.to_datetime(df["Time"]).dt.hour
    return df

try:
    df = get_data()

    # 3. SIDEBAR FILTERS
    st.sidebar.header("🕹️ Control Panel")
    city = st.sidebar.multiselect(
        "Select City:", options=df["City"].unique(), default=df["City"].unique()
    )
    customer_type = st.sidebar.multiselect(
        "Customer Type:", options=df["Customer type"].unique(), default=df["Customer type"].unique()
    )
    gender = st.sidebar.multiselect(
        "Gender:", options=df["Gender"].unique(), default=df["Gender"].unique()
    )

    # Apply Filters
    df_selection = df.query("City == @city & `Customer type` == @customer_type & Gender == @gender")

    # 4. MAIN HEADER
    st.title("🛒 Supermarket KPI Dashboard")
    st.markdown("### Interactive Performance Overview")
    st.markdown("---")

    if df_selection.empty:
        st.warning("No data available based on the current filters!")
    else:
        # 5. TOP LEVEL KPIs
        total_sales = df_selection["Sales"].sum()
        avg_rating = round(df_selection["Rating"].mean(), 1)
        total_gross_income = df_selection["gross income"].sum()
        total_qty = df_selection["Quantity"].sum()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Total Revenue 💰", value=f"${total_sales:,.0f}")
        kpi2.metric(label="Gross Income 📈", value=f"${total_gross_income:,.2f}")
        kpi3.metric(label="Items Sold 📦", value=f"{total_qty:,}")
        kpi4.metric(label="Avg Rating ⭐", value=f"{avg_rating} / 10")

        st.markdown("---")

        # 6. VISUALIZATIONS
        row1_col1, row1_col2 = st.columns(2)

        # FIXED CHART A: Select only numeric column ['Sales'] before summing
        sales_by_product = (
            df_selection.groupby(by=["Product line"])[["Sales"]].sum().sort_values(by="Sales")
        )
        fig_product_sales = px.bar(
            sales_by_product,
            x="Sales",
            y=sales_by_product.index,
            orientation="h",
            title="<b>Sales by Product Line</b>",
            color_discrete_sequence=["#0083B8"] * len(sales_by_product),
            template="plotly_white",
        )
        row1_col1.plotly_chart(fig_product_sales, use_container_width=True)

        # FIXED CHART B: Select only numeric column ['Sales'] before summing
        sales_by_hour = df_selection.groupby(by=["Hour"])[["Sales"]].sum()
        fig_hourly_sales = px.area(
            sales_by_hour,
            x=sales_by_hour.index,
            y="Sales",
            title="<b>Peak Sales Hours</b>",
            color_discrete_sequence=["#FFA500"],
            template="plotly_white",
        )
        row1_col2.plotly_chart(fig_hourly_sales, use_container_width=True)

        row2_col1, row2_col2, row2_col3 = st.columns(3)

        # CHART C: Payment Method Distribution
        fig_payment = px.pie(
            df_selection, values="Sales", names="Payment", 
            title="<b>Payment Methods</b>", hole=.3
        )
        row2_col1.plotly_chart(fig_payment, use_container_width=True)

        # FIXED CHART D: Revenue by City
        sales_by_city = df_selection.groupby(by=["City"])[["Sales"]].sum()
        fig_city = px.bar(
            sales_by_city, x=sales_by_city.index, y="Sales",
            title="<b>Revenue by City</b>",
            color=sales_by_city.index,
            template="plotly_white"
        )
        row2_col2.plotly_chart(fig_city, use_container_width=True)

        # CHART E: Avg Rating by Product Line
        rating_by_product = df_selection.groupby(by=["Product line"])[["Rating"]].mean()
        fig_rating = px.scatter(
            rating_by_product, x=rating_by_product.index, y="Rating",
            size="Rating", color="Rating",
            title="<b>Satisfaction Score by Category</b>",
            template="plotly_white"
        )
        row2_col3.plotly_chart(fig_rating, use_container_width=True)

        # 7. DATA TABLE VIEW
        with st.expander("👀 View Filtered Dataset"):
            st.dataframe(df_selection)

except Exception as e:
    st.error(f"Waiting for data... Please ensure 'SuperMarket_Analysis.csv' is uploaded. Error: {e}")


