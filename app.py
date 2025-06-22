import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="🦙 LLaMA-3 Health Chat")

# Load LLaMA-3 8B (this will be slow unless you have good hardware or use quantization)
@st.cache_resource
def load_llama3():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

tokenizer, model = load_llama3()

# Function to generate response
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# UI
st.title("🦙 LLaMA-3 Medical Q&A")
st.write("Ask any health-related question and get answers from Meta’s LLaMA-3 model.")

user_input = st.text_input("Ask your question:")

if user_input:
    with st.spinner("Generating answer..."):
        response = generate_answer(f"Q: {user_input}\nA:")
        st.success(f"**Answer:** {response}")
