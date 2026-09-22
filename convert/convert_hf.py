"""
convert_hf.py — вариант с моделью из Hugging Face.

DistilBERT (sentiment, SST-2) -> Core ML. Сценарий NaturalLanguage:
классификация тональности английского текста.

Запуск:
    pip install transformers
    python convert_hf.py
"""
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True).eval()

enc = tok("A great movie", return_tensors="pt", padding="max_length", max_length=64)
example = (enc["input_ids"], enc["attention_mask"])
traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=example[0].shape, dtype=int),
        ct.TensorType(name="attention_mask", shape=example[1].shape, dtype=int),
    ],
    minimum_deployment_target=ct.target.iOS16,
)
mlmodel.save("Sentiment.mlpackage")
print("Готово: Sentiment.mlpackage")
