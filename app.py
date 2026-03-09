import streamlit as st
import pandas as pd
import plotly.express as px

# Set Page Config
st.set_page_config(page_title="Supermarket KPI Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    # Reading the data (assuming the file is named SuperMarket_Analysis.csv)
    df = pd.read_csv("SuperMarket_Analysis.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Data")
    branch = st.sidebar.multiselect("Select Branch:", options=df["Branch"].unique(), default=df["Branch"].unique())
    customer_type = st.sidebar.multiselect("Customer Type:", options=df["Customer type"].unique(), default=df["Customer type"].unique())
    gender = st.sidebar.multiselect("Gender:", options=df["Gender"].unique(), default=df["Gender"].unique())

    df_selection = df.query("Branch == @branch & `Customer type` == @customer_type & Gender == @gender")

    # --- MAIN PAGE: KPIs ---
    st.title("📊 Supermarket Sales Dashboard")
    st.markdown("##")

    # Top Level Metrics
    total_sales = int(df_selection["Sales"].sum())
    average_rating = round(df_selection["Rating"].mean(), 1)
    total_gross_income = round(df_selection["gross income"].sum(), 2)
    avg_transaction = round(df_selection["Sales"].mean(), 2)

    left_column, middle_column, right_column, last_column = st.columns(4)
    with left_column:
        st.metric("Total Revenue", f"US ${total_sales:,}")
    with middle_column:
        st.metric("Avg Rating", f"{average_rating} ⭐")
    with right_column:
        st.metric("Total Gross Income", f"${total_gross_income:,}")
    with last_column:
        st.metric("Avg Spend/Invoice", f"${avg_transaction}")

    st.markdown("""---""")

    # --- CHARTS ---

    col1, col2 = st.columns(2)

    # 1. Sales by Product Line (Horizontal Bar Chart)
    sales_by_product_line = df_selection.groupby(by=["Product line"])[["Sales"]].sum().sort_values(by="Sales")
    fig_product_sales = px.bar(
        sales_by_product_line,
        x="Sales",
        y=sales_by_product_line.index,
        orientation="h",
        title="<b>Revenue by Product Line</b>",
        color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
        template="plotly_white",
    )
    col1.plotly_chart(fig_product_sales, use_container_width=True)

    # 2. Payment Method Distribution (Pie Chart)
    fig_payment = px.pie(
        df_selection, 
        values='Sales', 
        names='Payment', 
        title='<b>Payment Method Usage</b>',
        hole=0.4
    )
    col2.plotly_chart(fig_payment, use_container_width=True)

    col3, col4 = st.columns(2)

    # 3. Revenue by City (Bar Chart)
    sales_by_city = df_selection.groupby(by=["City"])[["Sales"]].sum()
    fig_city_sales = px.bar(
        sales_by_city,
        x=sales_by_city.index,
        y="Sales",
        title="<b>Revenue by City</b>",
        template="plotly_dark"
    )
    col3.plotly_chart(fig_city_sales, use_container_width=True)

    # 4. Rating by Product Line
    rating_by_product = df_selection.groupby(by=["Product line"])[["Rating"]].mean().sort_values(by="Rating")
    fig_rating = px.line(
        rating_by_product,
        x=rating_by_product.index,
        y="Rating",
        title="<b>Avg Rating per Product Category</b>",
        markers=True,
        template="plotly_white"
    )
    col4.plotly_chart(fig_rating, use_container_width=True)

    # Display Raw Data
    if st.checkbox("Show Raw Data"):
        st.dataframe(df_selection)

except Exception as e:
    st.error(f"Please ensure 'SuperMarket_Analysis.csv' is uploaded to Colab. Error: {e}")

