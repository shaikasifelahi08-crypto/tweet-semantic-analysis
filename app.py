import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Tweet Semantic Analysis",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 Tweet Semantic Analysis")
st.markdown("### Analyze Tweets using Multiple AI Models")
st.divider()

# Input
text = st.text_area("✍️ Enter Tweet / Comment")

# Model selection
model_choice = st.selectbox(
    "🤖 Select Model",
    ["RoBERTa", "BERT", "ALBERT"]
)

# Task selection
task = st.selectbox(
    "📌 Select Task",
    ["Sentiment", "Emotion", "Hate Speech", "Irony"]
)

# Load model dynamically
@st.cache_resource
def load_model(task, model_choice):

    if model_choice == "RoBERTa":
        models = {
            "Sentiment": "cardiffnlp/twitter-roberta-base-sentiment",
            "Emotion": "j-hartmann/emotion-english-distilroberta-base",
            "Hate Speech": "Hate-speech-CNERG/dehatebert-mono-english",
            "Irony": "cardiffnlp/twitter-roberta-base-irony"
        }

    elif model_choice == "BERT":
        models = {
            "Sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
            "Emotion": "bhadresh-savani/bert-base-uncased-emotion",
            "Hate Speech": "unitary/toxic-bert",
            "Irony": "cardiffnlp/twitter-roberta-base-irony"  # fallback
        }

    elif model_choice == "ALBERT":
        models = {
            "Sentiment": "textattack/albert-base-v2-imdb",
            "Emotion": "bhadresh-savani/albert-base-v2-emotion",
            "Hate Speech": "unitary/toxic-bert",
            "Irony": "cardiffnlp/twitter-roberta-base-irony"
        }

    return pipeline("text-classification", model=models[task])

# Label mapping (optional improvement)
label_map = {
    "LABEL_0": "Negative 😡",
    "LABEL_1": "Neutral 😐",
    "LABEL_2": "Positive 😊"
}

# Analyze button
if st.button("🚀 Analyze"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("🔍 Analyzing..."):
            model = load_model(task, model_choice)
            result = model(text)[0]

        label = result['label']
        score = result['score']

        # Convert label if possible
        label = label_map.get(label, label)

        st.success("✅ Analysis Complete")

        st.markdown(f"### 🤖 Model: {model_choice}")
        st.markdown(f"### 📌 Task: {task}")
        st.markdown(f"### 🏷️ Result: {label}")
        st.markdown(f"### 📊 Confidence: {score:.2f}")

        # Progress bar
        st.progress(float(score))
