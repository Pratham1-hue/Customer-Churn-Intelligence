import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    X_train = joblib.load('X_train.pkl')
    raw_df = joblib.load('raw_data.pkl')
    return model, explainer, X_train, raw_df

try:
    model, explainer, X_train, raw_df = load_artifacts()
except Exception as e:
    st.error("Model artifacts not found! Run `python train.py` first to create required models.")
    st.stop()

# Helper function to preprocess single user input for inference
def prepare_input(df_row):
    features = df_row.copy()
    if 'CustomerID' in features:
        features = features.drop(columns=['CustomerID'])
    if 'Churn' in features:
        features = features.drop(columns=['Churn'])
    
    # Align categorical dummy columns with training set
    encoded = pd.get_dummies(features, columns=['Contract_Type'], drop_first=True)
    for col in X_train.columns:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded[X_train.columns]

# Sidebar Navigation
st.sidebar.title("🛡️ Churn Intelligence")
page = st.sidebar.radio("Navigation", ["Global Executive Dashboard", "Account Diagnostic & SHAP", "What-If Simulator"])

# PAGE 1: GLOBAL EXECUTIVE DASHBOARD
if page == "Global Executive Dashboard":
    st.title("📊 Global Churn Risk Overview")
    st.markdown("Macro-level retention insights across enterprise account portfolio.")

    # High-level KPIs
    preprocessed_all = prepare_input(raw_df)
    churn_probs = model.predict_proba(preprocessed_all)[:, 1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Active Accounts", len(raw_df))
    col2.metric("Average Churn Risk", f"{np.mean(churn_probs)*100:.1f}%")
    col3.metric("High-Risk Accounts (>60%)", int(np.sum(churn_probs > 0.6)))
    col4.metric("Est. Revenue at Risk", f"${np.sum(raw_df['Monthly_Charges'] * (churn_probs > 0.5)):,.0f}")

    st.markdown("---")
    st.subheader("Global Feature Importance (SHAP Summary Plot)")
    st.write("Identifies top macroeconomic drivers behind account attrition across all clients.")
    
    shap_values_global = explainer(preprocessed_all)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values_global, preprocessed_all, show=False)
    st.pyplot(fig)

# PAGE 2: ACCOUNT DIAGNOSTIC & SHAP WATERFALL
elif page == "Account Diagnostic & SHAP":
    st.title("🔍 Individual Account Diagnostic Tool")
    st.markdown("Deep-dive attribution explaining why a specific customer is flagged for churn.")

    selected_cust = st.selectbox("Select Customer ID:", raw_df['CustomerID'].unique())
    cust_row = raw_df[raw_df['CustomerID'] == selected_cust].iloc[0]

    # Preprocess selected instance
    cust_processed = prepare_input(pd.DataFrame([cust_row]))
    risk_score = float(model.predict_proba(cust_processed)[:, 1][0])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Account Metadata")
        st.write(f"**Tenure:** {cust_row['Tenure_Months']} months")
        st.write(f"**Contract:** {cust_row['Contract_Type']}")
        st.write(f"**Monthly Charges:** ${cust_row['Monthly_Charges']:.2f}")
        st.write(f"**Support Tickets:** {cust_row['Support_Tickets']}")
        st.write(f"**NPS Score:** {cust_row['NPS_Score']}/10")
        
        st.markdown("---")
        st.subheader("Churn Probability Score")
        if risk_score > 0.6:
            st.error(f"🔴 HIGH RISK: {risk_score*100:.1f}%")
        elif risk_score > 0.3:
            st.warning(f"🟡 MEDIUM RISK: {risk_score*100:.1f}%")
        else:
            st.success(f"🟢 LOW RISK: {risk_score*100:.1f}%")

    with col2:
        st.subheader("SHAP Waterfall Explanation")
        st.caption("Shows positive (red/push up) and negative (blue/push down) factors driving this account's probability from the baseline.")

        shap_vals_single = explainer(cust_processed)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_vals_single[0], max_display=7, show=False)
        st.pyplot(fig)

    # Prescriptive Retention Recommendations
    st.markdown("---")
    st.subheader("💡 Prescriptive Action Plan for Account Manager")
    
    # Calculate top positive SHAP driver
    instance_shap = shap_vals_single.values[0]
    top_driver_idx = np.argmax(instance_shap)
    top_driver_name = X_train.columns[top_driver_idx]

    recommendations = []
    if cust_row['Contract_Type'] == 'Month-to-Month':
        recommendations.append("📌 **Contract Action:** Offer 15% discount for upgrading to an annual contract.")
    if cust_row['Support_Tickets'] > 3:
        recommendations.append("📌 **Support Intervention:** Schedule a dedicated call with Senior Customer Success Lead.")
    if cust_row['NPS_Score'] <= 5:
        recommendations.append("📌 **Satisfaction Outreach:** Trigger executive check-in and product feedback survey.")
    if cust_row['Payment_Delay_Days'] > 5:
        recommendations.append("📌 **Finance Action:** Enable automated payment reminders or flexible billing cycles.")

    for rec in recommendations:
        st.write(rec)

# PAGE 3: WHAT-IF SIMULATOR
elif page == "What-If Simulator":
    st.title("🎛️ Interactive Risk Simulator")
    st.markdown("Simulate how operational interventions impact customer churn risk in real-time.")

    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
        charges = st.slider("Monthly Charges ($)", 30.0, 300.0, 120.0)
        tickets = st.slider("Support Tickets", 0, 10, 2)
        login = st.slider("Monthly Login Frequency", 1, 30, 15)
        
    with col2:
        nps = st.slider("NPS Score", 1, 10, 7)
        delay = st.slider("Payment Delay (Days)", 0, 30, 2)
        contract = st.selectbox("Contract Type", ['Month-to-Month', 'One-Year', 'Two-Year'])

    simulated_df = pd.DataFrame([{
        'Tenure_Months': tenure,
        'Monthly_Charges': charges,
        'Support_Tickets': tickets,
        'Login_Frequency': login,
        'NPS_Score': nps,
        'Contract_Type': contract,
        'Payment_Delay_Days': delay,
        'Ticket_Density': tickets / (tenure + 1)
    }])

    sim_processed = prepare_input(simulated_df)
    sim_risk = model.predict_proba(sim_processed)[:, 1][0]

    st.markdown("---")
    st.metric("Simulated Risk Score", f"{sim_risk*100:.1f}%", delta=f"{-(sim_risk - 0.5)*100:.1f}% relative to baseline")