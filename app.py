# clinical_chatbot.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load ClinicalBERT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForMaskedLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return pipeline("fill-mask", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()

# UI
st.title("🩺 ClinicalBERT Health Assistant")
st.write("Ask health-related questions. E.g., 'Metformin may cause [MASK].'")

user_input = st.text_input("Enter a sentence with [MASK] to predict:")

if user_input:
    with st.spinner("Thinking..."):
        results = qa_pipeline(user_input, top_k=3)
        for r in results:
            st.success(f"Prediction: {r['sequence']} (Score: {r['score']:.4f})")
