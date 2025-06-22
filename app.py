#import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="🩺 Health Q&A Assistant")

# Load GPT-2
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# Function to generate response
def generate_answer(question):
    prompt = f"Answer this health question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated.replace(prompt, "").strip()

# UI
st.title("🩺 Health Q&A Assistant")
st.write("Ask me any health-related question and I'll try my best to answer.")

user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer = generate_answer(user_question)
        st.success(f"**Answer:** {answer}")
