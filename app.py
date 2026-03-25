import streamlit as st  # Import streamlit to build the web app interface
import pandas as pd  # Import pandas to handle and clean the data tables
import plotly.express as px  # Import plotly express for fast and easy charts
import plotly.graph_objects as go  # Import plotly objects for custom complex charts
from prophet import Prophet  # Import prophet to predict future sales values

# PAGE SETUP
st.set_page_config(page_title="Enterprise Dashboard", layout="wide")  # Set the browser tab title and use full screen width

# DESIGN SECTION
st.markdown("""
    <style>
    /* Change the background of the whole app */
    .stApp { background: linear-gradient(to right, #f8f9fa, #e9ecef); }
    
    /* Style the metric boxes to look like white cards */
    div[data-testid="stMetric"] {
        background: white; /* White color for the card */
        padding: 20px !important; /* Add space inside the card */
        border-radius: 15px; /* Round the corners of the card */
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); /* Add a soft shadow effect */
        border-left: 5px solid #00d2ff; /* Add a blue colored line on the left */
    }
    
    /* Style the sidebar on the left */
    section[data-testid="stSidebar"] { background-color: #1e1e2f !important; color: white; }
    
    /* Style the active tab button */
    .stTabs [aria-selected="true"] { background-color: #3a7bd5 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)  # Allow HTML and CSS to be used in the app

# DATA LOADING SECTION
@st.cache_data  # Tell streamlit to remember this data so it does not reload every time
def load_data():  # Define a function to load the data
    df = pd.read_csv("SuperMarket Analysis.csv")  # Read the CSV file from your folder
    df["Date"] = pd.to_datetime(df["Date"])  # Convert the Date column into a real date format
    df["Time_dt"] = pd.to_datetime(df["Time"], errors='coerce')  # Convert the Time column and ignore errors
    df["Hour"] = df["Time_dt"].dt.hour  # Extract only the hour number from the time
    return df  # Send the cleaned data back to the app

try:  # Start a safe block to catch any errors
    df = load_data()  # Run the load_data function and save it into df

    # SIDEBAR FILTERS
    with st.sidebar:  # Place everything inside the sidebar
        st.title("Admin Controls")  # Add a title to the sidebar
        date_range = st.date_input("Date Range", [df["Date"].min(), df["Date"].max()])  # Create a date picker input
        city = st.multiselect("Cities", options=df["City"].unique(), default=df["City"].unique())  # Create a city filter
        branch = st.multiselect("Branch", options=df["Branch"].unique(), default=df["Branch"].unique())  # Create a branch filter
        customer = st.multiselect("Customer Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())  # Create a customer type filter
        
    # FILTERING LOGIC
    mask = (df["City"].isin(city)) & (df["Branch"].isin(branch)) & (df["Customer type"].isin(customer))  # Create a true/false list based on filters
    if len(date_range) == 2:  # Check if both start and end dates are selected
        mask = mask & (df["Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])))  # Filter data between the two dates
    
    df_filtered = df[mask]  # Apply the true/false list to the data and create a new filtered table

    # TOP HEADER
    st.title("Supermarket Enterprise Dashboard")  # Add a main title to the page
    st.write(f"Analyzing {len(df_filtered)} transactions.")  # Show a count of how many rows are found
    
    # KEY PERFORMANCE INDICATORS (KPIs)
    k1, k2, k3, k4, k5 = st.columns(5)  # Split the row into 5 equal columns
    total_sales = df_filtered["Sales"].sum()  # Calculate the sum of the Sales column
    total_profit = df_filtered["gross income"].sum()  # Calculate the sum of the Gross Income column
    margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0  # Calculate profit percentage
    avg_rating = df_filtered["Rating"].mean()  # Calculate the average rating score
    total_qty = df_filtered["Quantity"].sum()  # Calculate the sum of units sold

    k1.metric("Total Revenue", f"${total_sales:,.0f}")  # Show Revenue in the first column
    k2.metric("Net Profit", f"${total_profit:,.0f}")  # Show Profit in the second column
    k3.metric("Margin Percent", f"{margin:.1f}%")  # Show Margin in the third column
    k4.metric("Avg Rating", f"{avg_rating:.1f}")  # Show Rating in the fourth column
    k5.metric("Units Sold", f"{total_qty:,}")  # Show Quantity in the fifth column

    st.markdown("---")  # Add a horizontal line to separate sections

    # TABS FOR ORGANIZATION
    tab_fin, tab_ops, tab_cust, tab_forecast, tab_data = st.tabs(["Financials", "Operations", "Customers", "Forecast", "Raw Data"])  # Create 5 tabs

    # FINANCIALS TAB
    with tab_fin:  # Logic for the first tab
        c1, c2 = st.columns([2, 1])  # Create two columns with different widths
        daily_sales = df_filtered.groupby("Date")[["Sales"]].sum().reset_index()  # Calculate total sales for every single day
        fig_trend = px.area(daily_sales, x="Date", y="Sales", title="Daily Sales Trend", color_discrete_sequence=["#3a7bd5"])  # Create an area line chart
        c1.plotly_chart(fig_trend, use_container_width=True)  # Show the chart in the first column
        
        prod_rev = df_filtered.groupby("Product line")["Sales"].sum().reset_index()  # Calculate total sales for each product category
        fig_donut = px.pie(prod_rev, values="Sales", names="Product line", hole=0.5, title="Sales by Product Category")  # Create a donut pie chart
        c2.plotly_chart(fig_donut, use_container_width=True)  # Show the chart in the second column

    # OPERATIONS TAB
    with tab_ops:  # Logic for the second tab
        c3, c4 = st.columns(2)  # Create two equal columns
        branch_perf = df_filtered.groupby("Branch")["Sales"].sum().reset_index()  # Calculate total sales per branch
        fig_branch = px.bar(branch_perf, x="Branch", y="Sales", color="Sales", title="Branch Sales Performance")  # Create a bar chart
        c3.plotly_chart(fig_branch, use_container_width=True)  # Show the chart in the first column
        
        hourly_data = df_filtered.groupby("Hour")["Sales"].sum().reset_index()  # Calculate sales for every hour of the day
        fig_hour = px.line(hourly_data, x="Hour", y="Sales", title="Sales by Hour of Day", markers=True)  # Create a line chart with dots
        c4.plotly_chart(fig_hour, use_container_width=True)  # Show the chart in the second column

    # CUSTOMERS TAB
    with tab_cust:  # Logic for the third tab
        c5, c6 = st.columns(2)  # Create two equal columns
        fig_pay = px.sunburst(df_filtered, path=['Gender', 'Payment'], values='Sales', title="Payment Method and Gender")  # Create a multi-level circle chart
        c5.plotly_chart(fig_pay, use_container_width=True)  # Show the chart in the first column
        
        sat_score = df_filtered.groupby("Product line")["Rating"].mean().sort_values().reset_index()  # Calculate average rating per product
        fig_sat = px.bar(sat_score, x="Rating", y="Product line", orientation='h', title="Customer Satisfaction Score")  # Create a horizontal bar chart
        c6.plotly_chart(fig_sat, use_container_width=True)  # Show the chart in the second column

    # FORECAST TAB (PREDICTION)
    with tab_forecast:  # Logic for the fourth tab
        st.subheader("30 Day Sales Prediction")  # Add a small title
        st.write("The system is using the Prophet model to look at past sales and predict the next 30 days.")  # Add an explanation text
        
        df_fc_input = daily_sales.rename(columns={'Date': 'ds', 'Sales': 'y'})  # Rename columns because Prophet only accepts 'ds' and 'y'
        
        if len(df_fc_input) > 10:  # Only run if there is enough data for a prediction
            model = Prophet(yearly_seasonality=False, daily_seasonality=True)  # Create the Prophet model
            model.fit(df_fc_input)  # Train the model using your filtered data
            
            future_dates = model.make_future_dataframe(periods=30)  # Create a list of the next 30 days
            forecast_results = model.predict(future_dates)  # Predict the sales for those 30 days
            
            fig_forecast = go.Figure()  # Start a new blank figure
            fig_forecast.add_trace(go.Scatter(x=df_fc_input['ds'], y=df_fc_input['y'], name="Past Sales"))  # Add a line for the actual past sales
            fig_forecast.add_trace(go.Scatter(x=forecast_results['ds'].iloc[-30:], y=forecast_results['yhat'].iloc[-30:], name="Predicted Sales", line=dict(color="red", dash='dash')))  # Add a red dashed line for the future sales
            
            fig_forecast.update_layout(title="Future Sales Trend", template="plotly_white")  # Clean the chart design
            st.plotly_chart(fig_forecast, use_container_width=True)  # Show the forecast chart
        else:  # If there is not enough data
            st.warning("Not enough data to calculate a forecast. Please select a larger date range in the sidebar.")  # Show a warning message

    # RAW DATA TAB
    with tab_data:  # Logic for the fifth tab
        st.subheader("View Filtered Records")  # Add a small title
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')  # Convert the filtered table into a CSV format
        st.download_button("Download Data as CSV", csv_data, "sales_report.csv", "text/csv")  # Create a button to download the file
        st.dataframe(df_filtered, use_container_width=True)  # Show the actual data table on the screen

except Exception as e:  # If the app crashes for any reason
    st.error(f"Critical System Error: {e}")  # Show the error message on the screen so you can fix it
