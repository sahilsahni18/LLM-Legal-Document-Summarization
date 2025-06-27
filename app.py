# app.py
import streamlit as st
from src.clause_detector import inference
import shap

@st.cache_resource
def load_explainer(model_dir="clause_model"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    explainer = shap.Explainer(model, tokenizer)
    return model, tokenizer, explainer

st.title("Legal Clause Risk Detector")
text = st.text_area("Paste clause text here", height=200)

if st.button("Analyze"):
    model, tokenizer, explainer = load_explainer()
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    shap_values = explainer([text])
    st.write("⚠️ Risk score:", float(shap_values.values[:,:,1].mean()))
    st.subheader("SHAP explanation")
    st.pyplot(shap.plots.text(shap_values[0]))
