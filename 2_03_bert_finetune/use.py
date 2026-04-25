from transformers import pipeline

classifier = pipeline("token-classification", "AXKuhta/bert-finetuned-ner")
