import streamlit as st
import openai
import os

# Page config
st.set_page_config(page_title="🧠 Medical Q&A Assistant")

# Input your OpenAI API key
openai_api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Only proceed if key is entered
if openai_api_key:
    openai.api_key = openai_api_key

    st.title("🩺 Ask a Medical Question")
    st.write("Ask me anything related to health and medicine!")

    user_question = st.text_input("💬 What would you like to know?")

    if user_question:
        with st.spinner("Thinking..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # or "gpt-4" if you have access
                    messages=[
                        {"role": "system", "content": "You are a helpful and medically knowledgeable assistant. Always explain in simple terms, but avoid giving medical advice. Stick to facts."},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.5,
                    max_tokens=300
                )
                answer = response["choices"][0]["message"]["content"].strip()
                st.success(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")
else:
    st.info("Please enter your OpenAI API key to begin.")
