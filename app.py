import streamlit as st
import openai
import os
from openai import OpenAI

# --- Streamlit UI ---
st.set_page_config(page_title="MedBot Assistant", page_icon="💊")
st.title("💬 MedBot – Your Health Assistant")

# --- User inputs API key manually ---
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    st.success("✅ API Key Loaded")
else:
    st.warning("Please enter your OpenAI API Key to continue")

# --- Set a custom system prompt ---
system_prompt = """
You are a friendly and intelligent in-app assistant named MedBot.
You help users:
1. Set medication reminders.
2. Log meals.
3. Answer questions about health and the app features.
Always be concise, polite, and use emojis where helpful.
"""

# --- Initialize Chat if API key is present ---
if openai_api_key:
    # Session state to store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": system_prompt}
        ]

    # Display previous messages
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Chat input box
    user_input = st.chat_input("Ask MedBot anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Call OpenAI API (v1.0+)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=st.session_state.messages
            )
            bot_reply = response.choices[0].message.content
        except Exception as e:
            bot_reply = "❌ Error: " + str(e)

        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.chat_message("assistant").markdown(bot_reply)

    # Optional: Simulate Meal Logging or Reminders
    with st.expander("📝 Simulate Meal Logging or Reminders"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Log Meal 🍱"):
                st.success("Meal logged successfully at this time.")
        with col2:
            if st.button("Set Med Reminder ⏰"):
                st.info("Reminder set for 8PM. We'll notify you then!")
