import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import datetime

st.set_page_config(page_title="My Health Assistant", page_icon="🩺")

# Load BioBERT QA Model
@st.cache_resource
def load_biobert_qa():
    tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_biobert_qa()

# Static context (can be replaced with medical DB)
context = """
Ulcers are open sores that develop on the inside lining of your stomach and the upper portion of your small intestine.
The most common cause of ulcers is infection with the bacterium Helicobacter pylori (H. pylori) and long-term use of NSAIDs.
Malaria is caused by Plasmodium parasites. Symptoms include fever, chills, and flu-like illness.
Treatment involves antimalarial medications.
"""

# Initial session state
if "meals" not in st.session_state:
    st.session_state.meals = []

if "reminder_time" not in st.session_state:
    st.session_state.reminder_time = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# App UI
st.title("🩺 My Health Assistant")
st.markdown("Hello! I'm your health buddy 😊 How can I help you today?")

# Feature Selection
option = st.selectbox(
    "What would you like to do?",
    [
        "Ask a medical question",
        "Log my meal",
        "Remind me about meds",
        "Help me use the app"
    ]
)

# 1️⃣ Ask a Question → BioBERT
if option == "Ask a medical question":
    question = st.text_input("Ask your question (e.g., What causes ulcer?)")
    if question:
        with st.spinner("Thinking..."):
            try:
                result = qa_pipeline(question=question, context=context)
                st.success(f"**Answer:** {result['answer']}")
            except:
                st.error("Sorry, I couldn't find an answer.")

# 2️⃣ Log My Meal
elif option == "Log my meal":
    meal = st.selectbox("Which meal are you logging?", ["Breakfast", "Lunch", "Dinner"])
    note = st.text_input("Add a note (optional):")
    if st.button("Log Meal"):
        entry = {
            "meal": meal,
            "note": note,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.meals.append(entry)
        st.success(f"{meal} logged!")

    if st.session_state.meals:
        st.markdown("### 🍽️ Your Meal Logs:")
        for m in st.session_state.meals:
            st.markdown(f"- **{m['meal']}** at {m['time']} — {m['note']}")

# 3️⃣ Med Reminder
elif option == "Remind me about meds":
    reminder_time = st.time_input("Set medication time")
    if st.button("Set Reminder"):
        st.session_state.reminder_time = reminder_time
        st.success(f"⏰ Reminder set for {reminder_time}.")

    # Check reminder
    now = datetime.datetime.now().time()
    if st.session_state.reminder_time and now >= st.session_state.reminder_time:
        st.warning("⏰ Time to take your meds!")
        # Reset reminder
        st.session_state.reminder_time = None

# 4️⃣ Help Navigate App
elif option == "Help me use the app":
    st.markdown("🗺️ **App Help Guide**")
    st.markdown("""
    - **Ask a medical question** → Get answers using BioBERT model trained on biomedical texts.
    - **Log my meal** → Save logs of what you ate and when.
    - **Remind me about meds** → Set a time to be reminded.
    - **Help me use the app** → You're here! 😄
    """)

