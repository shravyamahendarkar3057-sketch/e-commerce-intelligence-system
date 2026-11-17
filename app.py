
import streamlit as st
import pandas as pd
import sqlite3
import joblib

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="E-Commerce Intelligence System",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------------------
# PURPLE DASHBOARD STYLING
# -----------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F7F3FC;
    }

    h1, h2, h3 {
        color: #54258A;
        font-weight: 800;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E5DDF2;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(70, 40, 100, 0.08);
    }

    [data-testid="stMetricValue"] {
        color: #54258A;
    }

    [data-testid="stSidebar"] {
        background-color: #EEE8F7;
    }

    .stButton > button {
        background-color: #54258A;
        color: white;
        border-radius: 10px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #7546A8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# LOAD DATABASE
# -----------------------------------------

@st.cache_data
def load_data():

    conn = sqlite3.connect("data/processed/ecommerce.db")

    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    items = pd.read_sql_query("SELECT * FROM order_items", conn)
    customers = pd.read_sql_query("SELECT * FROM customers", conn)
    payments = pd.read_sql_query("SELECT * FROM payments", conn)

    conn.close()

    return orders, items, customers, payments


orders, items, customers, payments = load_data()


# -----------------------------------------
# LOAD MACHINE LEARNING MODEL
# -----------------------------------------

@st.cache_resource
def load_model():

    return joblib.load("models/delivery_prediction_model.pkl")


model_bundle = load_model()


# -----------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------

st.sidebar.title("🛍️ E-Commerce Insights")

st.sidebar.markdown("### Welcome!")

page = st.sidebar.radio(
    "Navigate",
    [
        "Business Overview",
        "Sales Analytics",
        "Customer Insights",
        "Delivery Performance",
        "Delivery Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "💜 Powered by Python, SQL & Machine Learning"
)

st.sidebar.caption("E-Commerce Intelligence System")


# -----------------------------------------
# PAGE 1: BUSINESS OVERVIEW
# -----------------------------------------

if page == "Business Overview":

    st.title("🛍️ E-Commerce Intelligence System")

    st.subheader("Business Overview")

    st.write(
        "Welcome! Explore the business performance "
        "of the e-commerce platform."
    )

    total_orders = len(orders)

    total_revenue = items["price"].sum()

    total_customers = customers["customer_unique_id"].nunique()

    delivered_orders = orders[
        orders["order_status"] == "delivered"
    ]

    valid_deliveries = delivered_orders.dropna(
        subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )

    valid_deliveries = valid_deliveries.copy()

    valid_deliveries["order_delivered_customer_date"] = pd.to_datetime(
        valid_deliveries["order_delivered_customer_date"]
    )

    valid_deliveries["order_estimated_delivery_date"] = pd.to_datetime(
        valid_deliveries["order_estimated_delivery_date"]
    )

    on_time_percentage = (
        (
            valid_deliveries["order_delivered_customer_date"]
            <= valid_deliveries["order_estimated_delivery_date"]
        ).mean() * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Orders", f"{total_orders:,}")

    col2.metric("Total Revenue", f"R$ {total_revenue:,.2f}")

    col3.metric("Unique Customers", f"{total_customers:,}")

    col4.metric("On-Time Delivery", f"{on_time_percentage:.2f}%")

    st.markdown("---")

    st.subheader("📦 Order Status Distribution")

    status_counts = orders["order_status"].value_counts()

    st.bar_chart(status_counts)

    st.success("Business Overview Loaded Successfully!")


# -----------------------------------------
# PAGE 2: SALES ANALYTICS
# -----------------------------------------

elif page == "Sales Analytics":

    st.title("📈 Sales Analytics")

    st.write("Explore revenue trends and order performance.")

    st.subheader("Monthly Revenue Trend")

    revenue_data = items.merge(
        orders[["order_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner"
    )

    revenue_data["order_purchase_timestamp"] = pd.to_datetime(
        revenue_data["order_purchase_timestamp"]
    )

    revenue_data = revenue_data.dropna(
        subset=["order_purchase_timestamp"]
    ).copy()

    revenue_data["month"] = (
        revenue_data["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_revenue = (
        revenue_data.groupby("month")["price"]
        .sum()
        .sort_index()
    )

    st.line_chart(monthly_revenue)

    st.markdown("---")

    st.subheader("💳 Payment Method Distribution")

    payment_counts = payments["payment_type"].value_counts()

    st.bar_chart(payment_counts)

    st.markdown("---")

    st.subheader("💰 Revenue Summary")

    st.metric(
        "Total Product Revenue",
        f"R$ {items['price'].sum():,.2f}"
    )

    st.caption("Revenue excludes freight charges.")


# -----------------------------------------
# PAGE 3: CUSTOMER INSIGHTS
# -----------------------------------------

elif page == "Customer Insights":

    st.title("👥 Customer Insights")

    st.write("Understand customer distribution across Brazil.")

    total_customers = customers["customer_unique_id"].nunique()

    st.metric("Total Unique Customers", f"{total_customers:,}")

    st.markdown("---")

    st.subheader("Top 10 Customer States")

    state_counts = (
        customers["customer_state"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(state_counts)

    st.markdown("---")

    st.subheader("Top 10 Customer Cities")

    city_counts = (
        customers["customer_city"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(city_counts)


# -----------------------------------------
# PAGE 4: DELIVERY PERFORMANCE
# -----------------------------------------

elif page == "Delivery Performance":

    st.title("🚚 Delivery Performance")

    st.write("Analyze early and late delivery performance.")

    delivery_data = orders.dropna(
        subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    ).copy()

    delivery_data["order_delivered_customer_date"] = pd.to_datetime(
        delivery_data["order_delivered_customer_date"]
    )

    delivery_data["order_estimated_delivery_date"] = pd.to_datetime(
        delivery_data["order_estimated_delivery_date"]
    )

    delivery_data["delivery_status"] = delivery_data.apply(
        lambda row: (
            "Late"
            if row["order_delivered_customer_date"]
            > row["order_estimated_delivery_date"]
            else "On Time / Early"
        ),
        axis=1
    )

    delivery_counts = (
        delivery_data["delivery_status"].value_counts()
    )

    total_valid_deliveries = len(delivery_data)

    late_count = (
        delivery_data["delivery_status"] == "Late"
    ).sum()

    early_count = (
        delivery_data["delivery_status"] == "On Time / Early"
    ).sum()

    col1, col2 = st.columns(2)

    col1.metric("Early / On-Time Deliveries", f"{early_count:,}")

    col2.metric("Late Deliveries", f"{late_count:,}")

    st.markdown("---")

    st.subheader("Delivery Performance Chart")

    st.bar_chart(delivery_counts)

    st.metric(
        "Late Delivery Percentage",
        f"{late_count / total_valid_deliveries * 100:.2f}%"
    )


# -----------------------------------------
# PAGE 5: MACHINE LEARNING PREDICTION
# -----------------------------------------

elif page == "Delivery Prediction":

    st.title("🤖 Delivery Delay Prediction")

    st.write(
        "Enter order details to estimate the risk of late delivery."
    )

    col1, col2 = st.columns(2)

    with col1:

        purchase_hour = st.number_input(
            "Purchase Hour", 0, 23, 12
        )

        purchase_day = st.number_input(
            "Purchase Day (0 = Monday)", 0, 6, 2
        )

        purchase_month = st.number_input(
            "Purchase Month", 1, 12, 6
        )

        item_count = st.number_input(
            "Number of Items", 1, 20, 1
        )

        total_price = st.number_input(
            "Total Product Price (R$)",
            min_value=0.0,
            value=100.0
        )

    with col2:

        total_freight = st.number_input(
            "Total Freight (R$)",
            min_value=0.0,
            value=20.0
        )

        unique_sellers = st.number_input(
            "Number of Sellers", 1, 10, 1
        )

        payment_value = st.number_input(
            "Payment Value (R$)",
            min_value=0.0,
            value=120.0
        )

        avg_installments = st.number_input(
            "Average Installments", 1, 24, 1
        )

        customer_state = st.selectbox(
            "Customer State",
            sorted(customers["customer_state"].dropna().unique())
        )

    if st.button("🔮 Predict Delivery Risk"):

        feature_columns = model_bundle["feature_columns"]

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=feature_columns
        )

        input_values = {
            "purchase_hour": purchase_hour,
            "purchase_day": purchase_day,
            "purchase_month": purchase_month,
            "item_count": item_count,
            "total_price": total_price,
            "total_freight": total_freight,
            "unique_sellers": unique_sellers,
            "payment_value": payment_value,
            "avg_installments": avg_installments
        }

        for feature, value in input_values.items():

            input_data.loc[0, feature] = value

        state_column = "customer_state_" + customer_state

        if state_column in feature_columns:

            input_data.loc[0, state_column] = 1

        prediction_probability = (
            model_bundle["model"]
            .predict_proba(input_data)[0][1]
        )

        threshold = model_bundle["threshold"]

        st.markdown("---")

        st.subheader("🔮 Prediction Result")

        st.metric(
            "Estimated Late Delivery Risk",
            f"{prediction_probability * 100:.2f}%"
        )

        if prediction_probability >= threshold:

            st.warning(
                "⚠️ The model flags this order as potentially late."
            )

        else:

            st.success(
                "✅ The model does not flag this order as late."
            )

        st.caption(
            "This is an experimental prediction based on historical "
            "data. It is not a guarantee of actual delivery performance."
        )
# Project Footer
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#7546A8;">
        <h4>🛍️ E-Commerce Intelligence System</h4>
        <p>Built using Python | SQL | Machine Learning | Streamlit</p>
        <p>Data-driven insights for smarter business decisions 💜</p>
    </div>
    """,
    unsafe_allow_html=True
)