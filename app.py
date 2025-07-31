import streamlit as st
import joblib

# Load the model
model = joblib.load("essay_scoring_model.pkl")

st.title("📘 Automated Essay Scoring")
st.markdown("Enter your essay below and get a score between 1 and 6.")

# Input text box
essay_text = st.text_area("✍️ Write or paste your essay here:", height=300)

# Predict button
if st.button("📊 Predict Essay Score"):
    if essay_text.strip() == "":
        st.warning("Please enter an essay before submitting.")
    else:
        # Predict
        prediction = model.predict([essay_text])[0]
        st.success(f"✅ Predicted Essay Score: **{prediction}**")

