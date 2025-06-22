from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

context = "Malaria is caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes."
question = "What causes malaria?"

result = qa_pipeline(question=question, context=context)
print(result['answer'])  # Output: "parasites"
