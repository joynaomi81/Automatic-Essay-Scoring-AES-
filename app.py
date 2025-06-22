import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Set up the Streamlit page
st.set_page_config(page_title="🩺 Ask Me Anything - ClinicalBERT")

# Load the ClinicalBERT model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()

# Built-in general medical knowledge context
context = """
Ulcers are open sores on the lining of the stomach or small intestine. They are caused by Helicobacter pylori infection or overuse of NSAIDs. 
Malaria is caused by Plasmodium parasites and spread through mosquito bites. It is treated with antimalarial medications.
High blood pressure, or hypertension, increases the risk of heart disease and stroke. It is managed with lifestyle changes and medication.
Asthma is a chronic condition that affects the airways, leading to wheezing and difficulty breathing.
Diabetes is a disease that affects how the body uses blood sugar and is managed with insulin, oral medications, and diet.
Typhoid fever is caused by Salmonella typhi and is treated with antibiotics.
Preventing ulcers involves avoiding NSAIDs, treating infections, managing stress, and eating healthily.
"""

# Streamlit UI
st.title("🩺 Ask Me Anything (ClinicalBERT)")
st.write("Ask a health-related question. The model will answer using trusted medical knowledge.")

question = st.text_input("💬 Ask your medical question:")

if question:
    with st.spinner("Thinking..."):
        try:
            result = qa_pipeline(question=question, context=context)
            answer = result.get("answer", "").strip()
            if answer:
                st.success(f"**Answer:** {answer}")
            else:
                st.warning("I couldn't find an exact answer. Try rephrasing your question.")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
