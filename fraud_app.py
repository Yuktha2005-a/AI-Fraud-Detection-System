import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide")

st.title("💳 AI Fraud Detection Dashboard - Karnataka")

# =========================
# LOAD MODEL
# =========================
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# KARNATAKA CITIES
# =========================
cities = [
    "Bangalore", "Mysore", "Mangalore", "Hubli", "Belgaum",
    "Gulbarga", "Davanagere", "Tumkur", "Udupi", "Shimoga",
    "Bellary", "Bijapur", "Chikmagalur", "Mandya", "Hassan",
    "Raichur", "Bidar", "Karwar", "Kolar", "Chitradurga"
]

city_mapping = {city: index for index, city in enumerate(cities)}

pincode_map = {
    "Bangalore":560001,"Mysore":570001,"Mangalore":575001,"Hubli":580020,
    "Belgaum":590001,"Gulbarga":585101,"Davanagere":577001,"Tumkur":572101,
    "Udupi":576101,"Shimoga":577201,"Bellary":583101,"Bijapur":586101,
    "Chikmagalur":577101,"Mandya":571401,"Hassan":573201,"Raichur":584101,
    "Bidar":585401,"Karwar":581301,"Kolar":563101,"Chitradurga":577501
}

# =========================
# SESSION STORAGE
# =========================
if "transactions" not in st.session_state:
    st.session_state.transactions = []

# =========================
# USER INPUT
# =========================
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=0.0)
    location = st.selectbox("Select Karnataka City", cities)
    st.write("📮 Pincode:", pincode_map[location])

with col2:
    transaction_time = st.time_input("Transaction Time")

# =========================
# FRAUD CHECK
# =========================
if st.button("Check Fraud"):

    city_code = city_mapping[location]
    hour = transaction_time.hour

    input_data = [[amount, city_code, hour]]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    fraud_score = round(probability * 100)

    if fraud_score >= 70:
        status = "BLOCK"
        st.error(f"🚫 BLOCKED | Fraud Score: {fraud_score}")
    elif fraud_score >= 40:
        status = "REVIEW"
        st.warning(f"⚠ UNDER REVIEW | Fraud Score: {fraud_score}")
    else:
        status = "SAFE"
        st.success(f"✅ SAFE | Fraud Score: {fraud_score}")

    # Save transaction
    st.session_state.transactions.append({
        "Amount": amount,
        "City": location,
        "Time": str(transaction_time),
        "Fraud Score": fraud_score,
        "Status": status
    })

# =========================
# DASHBOARD
# =========================
if st.session_state.transactions:

    df = pd.DataFrame(st.session_state.transactions)

    st.markdown("---")
    st.header("📊 Fraud Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Transactions", len(df))
        fraud_rate = round((df["Status"] == "BLOCK").mean() * 100, 2)
        st.metric("Fraud Rate (%)", fraud_rate)

    with col2:
        fig_status = px.pie(df, names="Status", title="Fraud Distribution")
        st.plotly_chart(fig_status)

    fig_city = px.bar(df, x="City", title="Transactions by City")
    st.plotly_chart(fig_city)

    st.subheader("📄 Transaction History")
    st.dataframe(df)