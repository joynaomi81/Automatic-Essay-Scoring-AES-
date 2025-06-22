import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load BioBERT QA model
@st.cache_resource
def load_biobert_qa():
    tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_biobert_qa()

# Fixed context for now (can be replaced with full knowledge base)
context = """
Malaria is caused by Plasmodium parasites. The parasites are spread to people through the bites of infected female Anopheles mosquitoes.
Symptoms include fever, chills, and flu-like illness. Without treatment, it can lead to severe complications and death.
Treatment typically involves antimalarial medications.
"""

st.title("🩺 BioBERT Medical Assistant")
st.markdown("Ask any **medical question** (e.g., 'What causes malaria?')")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if question := st.chat_input("Type your health question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_pipeline(question=question, context=context)
                answer = result['answer']
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                answer = "Sorry, I couldn't find an answer to that."
                st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": f"**Answer:** {answer}"})
