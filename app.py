import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib


# PAGE CONFIG


st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# LOAD DATA


@st.cache_data
def load_dashboard_data():
    return pd.read_csv("dashboard_data.csv")

@st.cache_data
def load_feature_importance():
    return pd.read_csv("feature_importance.csv")

df = load_dashboard_data()
importance_df = load_feature_importance()

# Load Model

@st.cache_resource
def load_model():
    return joblib.load(
        "telco_churn_probability_estimator_lr.joblib"
    )

model = load_model()


# SIDEBAR


page = st.sidebar.radio(
    "Select Page",
    [
        "Executive Overview",
        "Customer Risk Analysis",
        "Churn Drivers",
        "Live Prediction",
        "Batch Scoring"
    ]
)

st.sidebar.divider()

uploaded_batch_file = st.sidebar.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)

# EXECUTIVE OVERVIEW


if page == "Executive Overview":

    st.title("📊 Customer Churn Intelligence Platform")
    st.caption("Predict • Prioritize • Retain")


    # KPI SECTION


    total_customers = len(df)

    high_risk_customers = (
        df["risk_level"] == "High"
    ).sum()

    avg_probability = (
        df["churn_probability"].mean() * 100
    )

    revenue_at_risk = (
        df["monthly_revenue_at_risk"].sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "High Risk Customers",
        f"{high_risk_customers:,}"
    )

    col3.metric(
        "Revenue At Risk",
        f"${revenue_at_risk:,.0f}"
    )

    col4.metric(
        "Avg Churn Probability",
        f"{avg_probability:.1f}%"
    )

    st.divider()


    # CHARTS


    left, right = st.columns(2)

    with left:

        risk_counts = (
            df["risk_level"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Customers"
        ]

        fig_risk = px.pie(
            risk_counts,
            names="Risk Level",
            values="Customers",
            title="Customer Risk Segmentation"
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

    with right:

        fig_prob = px.histogram(
            df,
            x="churn_probability",
            nbins=25,
            title="Churn Probability Distribution"
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )


    # REVENUE AT RISK


    st.subheader(
        "Expected Revenue Loss by Risk Tier"
    )

    revenue_summary = (
        df.groupby("risk_level")
        ["monthly_revenue_at_risk"]
        .sum()
        .reset_index()
    )

    fig_rev = px.bar(
        revenue_summary,
        x="risk_level",
        y="monthly_revenue_at_risk",
        color="risk_level",
        title="Monthly Revenue At Risk"
    )

    st.plotly_chart(
        fig_rev,
        use_container_width=True
    )


# CUSTOMER RISK ANALYSIS


elif page == "Customer Risk Analysis":

    st.title("🔍 Customer Risk Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        risk_filter = st.multiselect(
            "Risk Level",
            options=df["risk_level"].unique(),
            default=df["risk_level"].unique()
        )

    with col2:

        contract_filter = st.multiselect(
            "Contract",
            options=df["Contract"].unique(),
            default=df["Contract"].unique()
        )

    with col3:

        internet_filter = st.multiselect(
            "Internet Service",
            options=df["InternetService"].unique(),
            default=df["InternetService"].unique()
        )

    filtered_df = df[
        df["risk_level"].isin(risk_filter)
    ]

    filtered_df = filtered_df[
        filtered_df["Contract"].isin(contract_filter)
    ]

    filtered_df = filtered_df[
        filtered_df["InternetService"].isin(
            internet_filter
        )
    ]

    st.subheader(
        "Retention Priority List"
    )

    display_cols = [
        "customerID",
        "risk_level",
        "churn_probability",
        "MonthlyCharges",
        "monthly_revenue_at_risk",
        "Contract",
        "InternetService",
        "tenure"
    ]

    st.dataframe(
        filtered_df[display_cols]
        .sort_values(
            "churn_probability",
            ascending=False
        ),
        use_container_width=True
    )

    csv = (
        filtered_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "📥 Download Retention List",
        csv,
        "retention_priority_list.csv",
        "text/csv"
    )





# CHURN DRIVERS


elif page == "Churn Drivers":

    st.title("📈 Churn Drivers")

    positive = (
        importance_df[
            importance_df["coefficient"] > 0
        ]
        .sort_values(
            "coefficient",
            ascending=False
        )
        .head(15)
    )

    negative = (
        importance_df[
            importance_df["coefficient"] < 0
        ]
        .sort_values(
            "coefficient"
        )
        .head(15)
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_pos = px.bar(
            positive,
            x="coefficient",
            y="feature",
            orientation="h",
            title="Features Increasing Churn"
        )

        st.plotly_chart(
            fig_pos,
            use_container_width=True
        )

    with col2:

        fig_neg = px.bar(
            negative,
            x="coefficient",
            y="feature",
            orientation="h",
            title="Features Reducing Churn"
        )

        st.plotly_chart(
            fig_neg,
            use_container_width=True
        )

    st.divider()

    st.subheader(
        "Business Interpretation"
    )

    st.info(
        """
        ### Key Findings

        • Month-to-month contracts are strongly associated with churn.

        • Customers with short tenure exhibit the highest churn risk.

        • Fiber optic customers experience elevated churn rates.

        • Long-term contracts significantly improve retention.

        • Higher customer lifetime value is associated with lower churn risk.

        ### Recommended Actions

        • Target customers during their first year.

        • Encourage migration from month-to-month to annual contracts.

        • Monitor high-value fiber customers for early warning signs.

        • Prioritize retention campaigns for customers classified as High Risk.
        """
    )

    
# LIVE PREDICTION


elif page == "Live Prediction":

    st.title("🤖 Live Churn Prediction")


    st.divider()

    st.subheader("📂 Batch Prediction")

    uploaded_live_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        key="live_prediction_upload"
    )

    if uploaded_live_file is not None:

        batch_df = pd.read_csv(uploaded_live_file)

        try:

            probs = model.predict_proba(batch_df)[:,1]

            batch_df["churn_probability"] = probs

            batch_df["risk_level"] = np.where(
                probs >= 0.60,
                "High",
                np.where(
                    probs >= 0.30,
                    "Medium",
                    "Low"
                )
            )

            batch_df["monthly_revenue_at_risk"] = (
                batch_df["MonthlyCharges"]
                * batch_df["churn_probability"]
            )

            st.success(
                f"{len(batch_df):,} customers scored successfully."
            )

            st.dataframe(
                batch_df.head(),
                use_container_width=True
            )

            csv = (
                batch_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "📥 Download Scored Dataset",
                csv,
                "scored_customers.csv",
                "text/csv"
            )

        except Exception as e:

            st.error(
                f"Dataset format error: {e}"
            )

    st.divider()








    st.markdown(
        """
        Enter customer information to estimate
        churn probability and business impact.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.divider()

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=1000.0
    )

    if st.button("Predict Churn Risk"):

        avg_monthly_spend = (
            monthly_charges / (tenure + 1)
        )

        customer_lifetime_value = (
            monthly_charges * tenure
        )

        input_df = pd.DataFrame(
            {
                "gender":[gender],
                "SeniorCitizen":[senior],
                "Partner":[partner],
                "Dependents":[dependents],
                "tenure":[tenure],
                "PhoneService":[phone_service],
                "MultipleLines":[multiple_lines],
                "InternetService":[internet_service],
                "OnlineSecurity":[online_security],
                "OnlineBackup":[online_backup],
                "DeviceProtection":[device_protection],
                "TechSupport":[tech_support],
                "StreamingTV":[streaming_tv],
                "StreamingMovies":[streaming_movies],
                "Contract":[contract],
                "PaperlessBilling":[paperless],
                "PaymentMethod":[payment_method],
                "MonthlyCharges":[monthly_charges],
                "TotalCharges":[total_charges],
                "avg_monthly_spend":[avg_monthly_spend],
                "customer_lifetime_value":[customer_lifetime_value]
            }
        )

        probability = (
            model.predict_proba(input_df)[0][1]
        )

        prediction = (
            "Likely to Churn"
            if probability >= 0.5
            else "Likely to Stay"
        )

        if probability >= 0.60:
            risk_level = "High"

        elif probability >= 0.30:
            risk_level = "Medium"

        else:
            risk_level = "Low"

        revenue_at_risk = (
            monthly_charges * probability
        )

        st.divider()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Churn Probability",
            f"{probability*100:.1f}%"
        )

        c2.metric(
            "Risk Level",
            risk_level
        )

        c3.metric(
            "Revenue At Risk",
            f"${revenue_at_risk:.2f}"
        )

        if probability >= 0.60:

            st.error(
                f"{prediction}"
            )

        elif probability >= 0.30:

            st.warning(
                f"{prediction}"
            )

        else:

            st.success(
                f"{prediction}"
            )




# BATCH SCORING


elif page == "Batch Scoring":

    st.title("📂 Batch Customer Scoring")

    st.markdown(
        """
        Upload a customer dataset to generate
        churn probabilities, risk tiers and
        revenue-at-risk estimates.
        """
    )

    if uploaded_batch_file is not None:

        batch_df = pd.read_csv(
            uploaded_batch_file
        )

        try:

            probs = model.predict_proba(
                batch_df
            )[:,1]

            batch_df["churn_probability"] = probs

            batch_df["risk_level"] = np.where(
                probs >= 0.60,
                "High",
                np.where(
                    probs >= 0.30,
                    "Medium",
                    "Low"
                )
            )

            batch_df["monthly_revenue_at_risk"] = (
                batch_df["MonthlyCharges"]
                * batch_df["churn_probability"]
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Customers",
                f"{len(batch_df):,}"
            )

            c2.metric(
                "High Risk",
                (
                    batch_df["risk_level"]
                    == "High"
                ).sum()
            )

            c3.metric(
                "Revenue At Risk",
                f"${batch_df['monthly_revenue_at_risk'].sum():,.0f}"
            )

            st.divider()

            st.dataframe(
                batch_df.sort_values(
                    "churn_probability",
                    ascending=False
                ),
                use_container_width=True
            )

            csv = (
                batch_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "📥 Download Predictions",
                csv,
                "batch_predictions.csv",
                "text/csv"
            )

        except Exception as e:

            st.error(
                f"Dataset format error: {e}"
            )

    else:

        st.info(
            "Upload a CSV file from the sidebar."
        )