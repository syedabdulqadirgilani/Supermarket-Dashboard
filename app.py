import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet  # Best Time-Series Model

# 1. PAGE SETUP
st.set_page_config(page_title="Enterprise BI Dashboard", layout="wide", page_icon="💎")

# 2. CUSTOM CSS (Dashboard ko khoobsurat banane ke liye)
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to right, #f8f9fa, #e9ecef); }
    div[data-testid="stMetric"] {
        background: white;
        padding: 20px !important;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #00d2ff;
    }
    section[data-testid="stSidebar"] { background-color: #1e1e2f !important; color: white; }
    .stTabs [aria-selected="true"] { background-color: #3a7bd5 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. DATA ENGINE (Data load aur saaf karne ke liye)
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
        date_range = st.date_input("📅 Date Range", [df["Date"].min(), df["Date"].max()])
        city = st.multiselect("📍 Cities", options=df["City"].unique(), default=df["City"].unique())
        branch = st.multiselect("🏬 Branch", options=df["Branch"].unique(), default=df["Branch"].unique())
        customer = st.multiselect("👥 Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())
        
    # Data Filtering Logic
    mask = (df["City"].isin(city)) & (df["Branch"].isin(branch)) & (df["Customer type"].isin(customer))
    if len(date_range) == 2:
        mask = mask & (df["Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])))
    
    df_filtered = df[mask]

    # --- HEADER ---
    st.title("💎 Supermarket Enterprise Dashboard")
    
    # --- 4. KPIs ROW ---
    k1, k2, k3, k4, k5 = st.columns(5)
    total_sales = df_filtered["Sales"].sum()
    total_profit = df_filtered["gross income"].sum()
    margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0
    avg_rating = df_filtered["Rating"].mean()
    total_qty = df_filtered["Quantity"].sum()

    k1.metric("Total Revenue", f"${total_sales:,.0f}")
    k2.metric("Net Profit", f"${total_profit:,.1f}")
    k3.metric("Profit Margin", f"{margin:.1f}%")
    k4.metric("Avg Rating", f"{avg_rating:.1f} ⭐")
    k5.metric("Units Sold", f"{total_qty:,} pcs")

    st.markdown("---")

    # --- 5. TABS (Including the New Forecast Tab) ---
    tab_fin, tab_ops, tab_cust, tab_forecast, tab_data = st.tabs([
        "💰 Financials", "⚙️ Operations", "👤 Customers", "🔮 Predictive Insights", "📂 Raw Data"
    ])

    # --- TAB: FINANCIALS ---
    with tab_fin:
        c1, c2 = st.columns([2, 1])
        daily_sales = df_filtered.groupby("Date")[["Sales"]].sum().reset_index()
        fig_trend = px.area(daily_sales, x="Date", y="Sales", title="Daily Revenue Trend", color_discrete_sequence=["#3a7bd5"])
        c1.plotly_chart(fig_trend, use_container_width=True)
        
        prod_rev = df_filtered.groupby("Product line")["Sales"].sum().reset_index()
        fig_donut = px.pie(prod_rev, values="Sales", names="Product line", hole=0.6, title="Category Share")
        c2.plotly_chart(fig_donut, use_container_width=True)

    # --- TAB: OPERATIONS ---
    with tab_ops:
        c3, c4 = st.columns(2)
        branch_perf = df_filtered.groupby("Branch")["Sales"].sum().reset_index()
        c3.plotly_chart(px.bar(branch_perf, x="Branch", y="Sales", color="Sales", title="Branch Performance"), use_container_width=True)
        
        hourly_data = df_filtered.groupby("Hour")["Sales"].sum().reset_index()
        c4.plotly_chart(px.line(hourly_data, x="Hour", y="Sales", title="Peak Hours", markers=True), use_container_width=True)

    # --- TAB: CUSTOMERS ---
    with tab_cust:
        c5, c6 = st.columns(2)
        fig_pay = px.sunburst(df_filtered, path=['Gender', 'Payment'], values='Sales', title="Payment by Gender")
        c5.plotly_chart(fig_pay, use_container_width=True)
        sat_score = df_filtered.groupby("Product line")["Rating"].mean().reset_index()
        c6.plotly_chart(px.bar(sat_score, x="Rating", y="Product line", orientation='h', title="Customer Satisfaction"), use_container_width=True)

    # --- NEW TAB: FORECASTING (Using Prophet) ---
    with tab_forecast:
        st.subheader("🔮 30-Day Sales Forecast")
        st.info("Meta Prophet model pichle data ko analyze karke future predict kar raha hai...")

        # 1. Data Prep for Prophet (Prophet ko 'ds' aur 'y' columns chahiye)
        df_forecast = daily_sales.rename(columns={'Date': 'ds', 'Sales': 'y'})
        
        if len(df_forecast) > 5: # Prediction ke liye thoda data hona zaroori hai
            # 2. Model Initialization & Fitting
            m = Prophet(interval_width=0.95, yearly_seasonality=False, daily_seasonality=True)
            m.fit(df_forecast)
            
            # 3. Future Dates create karna (30 days)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            # 4. Visualization (Actual vs Predicted)
            fig_fc = go.Figure()
            # Actual Data
            fig_fc.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['y'], name="Actual Sales", line=dict(color="#3a7bd5")))
            # Predicted Data
            fig_fc.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat'].iloc[-30:], 
                                        name="Forecast", line=dict(color="#FF4B4B", dash='dash')))
            # Confidence Interval (Shaded area)
            fig_fc.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat_upper'].iloc[-30:], 
                                        fill=None, mode='lines', line_color='rgba(255,75,75,0.2)', name="Upper Bound"))
            fig_fc.add_trace(go.Scatter(x=forecast['ds'].iloc[-30:], y=forecast['yhat_lower'].iloc[-30:], 
                                        fill='tonexty', mode='lines', line_color='rgba(255,75,75,0.2)', name="Lower Bound"))
            
            fig_fc.update_layout(title="Sales Forecast: Next 30 Days", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_fc, use_container_width=True)
            
            st.write("💡 **Insight:** Dotted line agle mahine ki mutawaqqe (expected) sales dikha rahi hai.")
        else:
            st.warning("Forecasting ke liye data kam hai. Please Date Range barhaein.")

    # --- TAB: RAW DATA ---
    with tab_data:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered CSV", csv, "report.csv", "text/csv")
        st.dataframe(df_filtered, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
