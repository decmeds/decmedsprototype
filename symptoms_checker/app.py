import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import joblib

# Initialize the model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_dir = "D:\\Streamlit\\symptoms_checker\\model"
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForSequenceClassification.from_pretrained(output_dir)
model.to(device)

class BioBERTWithDropout(torch.nn.Module):
    def __init__(self, model, dropout_prob=0.3):
        super(BioBERTWithDropout, self).__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.dropout(outputs.logits)
        return logits

classifier_model = BioBERTWithDropout(model)
classifier_model.to(device)

label_encoder = joblib.load('D:\\Streamlit\\symptoms_checker\\model\\label_encoder.joblib')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_disease(prompt):
    symptoms = preprocess_text(prompt)
    tokenized_prompt = tokenizer(symptoms, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = classifier_model(**tokenized_prompt)
    predicted_label = torch.argmax(outputs, dim=1).item()
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_disease

# Streamlit UI
st.title("Disease Prediction")

prompt = st.text_area("Enter symptoms:")

if st.button("Predict Disease"):
    if prompt:
        predicted_disease = predict_disease(prompt)
        st.write(f"**Predicted Disease:** {predicted_disease}")
    else:
        st.write("Please enter symptoms to predict.")
