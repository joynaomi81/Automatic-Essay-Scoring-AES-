import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Set page config
st.set_page_config(page_title="🩺 ClinicalBERT Health Q&A")

# Load ClinicalBERT QA pipeline
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()

# Richer context: multiple conditions, causes, symptoms, treatments
context = """
Ulcers are sores on the lining of the stomach or small intestine. They are commonly caused by Helicobacter pylori infection or overuse of NSAIDs.
Symptoms include burning stomach pain, bloating, and nausea. Treatment includes antibiotics and acid-reducing medications.

Malaria is a mosquito-borne disease caused by Plasmodium parasites. Symptoms include fever, chills, sweating, and vomiting.
It is treated with antimalarial drugs like chloroquine or artemisinin-based therapies.

Diabetes is a chronic condition affecting insulin production or use. Symptoms include frequent urination, thirst, fatigue, and blurred vision.
Treatment includes insulin therapy, lifestyle changes, and oral medications.

High blood pressure, or hypertension, is when the force of blood against artery walls is too high. It can lead to heart disease and stroke.
It is often treated with lifestyle modifications and antihypertensive drugs.

Asthma is a respiratory condition where airways become inflamed and narrowed. Triggers include allergens, cold air, and exercise.
Symptoms include wheezing, shortness of breath, and coughing. Treatment includes inhalers and corticosteroids.

Cancer is a group of diseases involving abnormal cell growth. Causes vary but include genetic factors, carcinogens, and lifestyle.
Treatment may involve surgery, chemotherapy, radiation, or immunotherapy.
"""

# Streamlit UI
st.title("🩺 ClinicalBERT Medical Q&A")
st.markdown("Ask any health-related question. This assistant will answer using biomedical knowledge.")

question = st.text_input("Your medical question:")

if question:
    with st.spinner("Finding the answer..."):
        try:
            result = qa_pipeline(question=question, context=context)
            st.success(f"**Answer:** {result['answer']}")
        except Exception as e:
            st.error("Sorry, I couldn't answer that. Try a simpler or clearer question.")
