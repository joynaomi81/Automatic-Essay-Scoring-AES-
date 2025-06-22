import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Set page config
st.set_page_config(page_title="🩺 FLAN-T5 Health Chatbot")

# Load FLAN-T5 model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()

# Streamlit UI
st.title("🩺 Ask Me Anything: FLAN-T5 Health Assistant")
st.write("Ask any health-related question below, and I'll do my best to help.")

user_question = st.text_input("💬 Your medical question:")

if user_question:
    with st.spinner("Thinking..."):
        prompt = f"Answer this health question: {user_question}"
        result = qa_pipeline(prompt, max_new_tokens=100)[0]['generated_text']
        st.success(f"**Answer:** {result}")
