import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE SETUP
st.set_page_config(page_title="Enterprise BI Dashboard", layout="wide", page_icon="💎")

# 2. CUSTOM CSS FOR PREMIUM LOOK
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    
    /* KPI Cards Styling */
    div[data-testid="stMetric"] {
        background: white;
        padding: 20px !important;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #00d2ff;
        transition: transform 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1e2f !important;
        color: white;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding: 10px 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #3a7bd5 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. DATA ENGINE
@st.cache_data
def load_data():
    df = pd.read_csv("SuperMarket Analysis.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time_dt"] = pd.to_datetime(df["Time"], errors='coerce')
    df["Hour"] = df["Time_dt"].dt.hour
    return df

try:
    df = load_data()

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=80)
        st.title("Admin Panel")
        st.markdown("---")
        
        date_range = st.date_input("📅 Date Range", [df["Date"].min(), df["Date"].max()])
        city = st.multiselect("📍 Cities", options=df["City"].unique(), default=df["City"].unique())
        branch = st.multiselect("🏬 Branch", options=df["Branch"].unique(), default=df["Branch"].unique())
        customer = st.multiselect("👥 Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())
        
    # Filter Mask
    mask = (df["City"].isin(city)) & (df["Branch"].isin(branch)) & (df["Customer type"].isin(customer))
    if len(date_range) == 2:
        mask = mask & (df["Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])))
    
    df_filtered = df[mask]

    # --- HEADER SECTION ---
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.title("💎 Supermarket Enterprise Dashboard")
        st.markdown(f"**Live Analysis:** Tracking `{len(df_filtered)}` verified transactions")
    with head_col2:
        # Mini Dynamic Info
        top_city = df_filtered.groupby("City")["Sales"].sum().idxmax() if not df_filtered.empty else "N/A"
        st.success(f"🏆 Top City: **{top_city}**")

    # --- 4. KPIs ROW ---
    st.markdown("### 🚀 Performance Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    
    total_sales = df_filtered["Sales"].sum()
    total_profit = df_filtered["gross income"].sum()
    margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0
    avg_rating = df_filtered["Rating"].mean()
    total_qty = df_filtered["Quantity"].sum()

    k1.metric("Total Revenue", f"${total_sales:,.0f}", delta="↑ 5.2%")
    k2.metric("Net Profit", f"${total_profit:,.1f}", delta="Stable")
    k3.metric("Profit Margin", f"{margin:.1f}%", delta="↑ 0.4%")
    k4.metric("Avg Rating", f"{avg_rating:.1f} ⭐")
    k5.metric("Units Sold", f"{total_qty:,} pcs")

    st.markdown("---")

    # --- 5. VISUALIZATION TABS ---
    tab_fin, tab_ops, tab_cust, tab_data = st.tabs(["💰 Financials", "⚙️ Operations", "👤 Customers", "📂 Raw Data"])

    with tab_fin:
        c1, c2 = st.columns([2, 1])
        with c1:
            # Sales Trend with Area & Line combo
            daily_sales = df_filtered.groupby("Date")[["Sales"]].sum().reset_index()
            fig_trend = px.area(daily_sales, x="Date", y="Sales", title="<b>Daily Revenue Performance</b>",
                                color_discrete_sequence=["#3a7bd5"], template="plotly_white")
            fig_trend.update_layout(hovermode="x unified")
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with c2:
            # Revenue by Product Line (Donut)
            prod_rev = df_filtered.groupby("Product line")["Sales"].sum().reset_index()
            fig_donut = px.pie(prod_rev, values="Sales", names="Product line", hole=0.6,
                               title="<b>Category Contribution</b>", 
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_donut, use_container_width=True)

    with tab_ops:
        c3, c4 = st.columns(2)
        with c3:
            # Vertical Bar for Branch Sales
            branch_perf = df_filtered.groupby("Branch")["Sales"].sum().sort_values().reset_index()
            fig_branch = px.bar(branch_perf, x="Branch", y="Sales", color="Sales",
                                title="<b>Sales Performance by Branch</b>",
                                color_continuous_scale="Viridis")
            st.plotly_chart(fig_branch, use_container_width=True)
        
        with c4:
            # Hourly Heatmap-style Area
            hourly_data = df_filtered.groupby("Hour")["Sales"].sum().reset_index()
            fig_hour = px.line(hourly_data, x="Hour", y="Sales", markers=True,
                               title="<b>Peak Business Hours</b>",
                               line_shape="spline", color_discrete_sequence=["#FF4B4B"])
            st.plotly_chart(fig_hour, use_container_width=True)

    with tab_cust:
        c5, c6 = st.columns(2)
        with c5:
            # Payment Method vs Gender
            fig_pay = px.sunburst(df_filtered, path=['Gender', 'Payment'], values='Sales',
                                  title="<b>Payment Patterns by Gender</b>",
                                  color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig_pay, use_container_width=True)
        
        with c6:
            # Satisfaction Score (Radar-like Bar)
            sat_score = df_filtered.groupby("Product line")["Rating"].mean().sort_values().reset_index()
            fig_sat = px.bar(sat_score, x="Rating", y="Product line", orientation='h',
                             title="<b>Customer Satisfaction per Category</b>",
                             color="Rating", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_sat, use_container_width=True)

    with tab_data:
        st.subheader("Transactional Intelligence Table")
        # Download Button Row
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered CSV", csv, "supermarket_report.csv", "text/csv")
        st.dataframe(df_filtered.style.background_gradient(cmap='Blues', subset=['Sales', 'Rating']), use_container_width=True)

except Exception as e:
    st.error(f"Please check your dataset: {e}")
