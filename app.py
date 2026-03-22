import streamlit as st
from transformers import pipeline

st.title("🧠 Tweet Semantic Analysis")

text = st.text_area("Enter Tweet")

task = st.selectbox(
    "Select Task",
    ["Sentiment", "Emotion", "Hate Speech", "Irony"]
)

@st.cache_resource
def load_models():
    return {
        "Sentiment": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment"),
        "Emotion": pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base"),
        "Hate Speech": pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english"),
        "Irony": pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")
    }

models = load_models()

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        result = models[task](text)
        st.success("Result:")
        st.write(result)
