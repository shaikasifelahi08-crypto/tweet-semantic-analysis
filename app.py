import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(page_title="Tweet Semantic Analysis", page_icon="🧠")

# Sidebar
with st.sidebar:
    st.title(" Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["Auto", "RoBERTa", "BERT", "ALBERT"]
    )

# Title
st.title("Tweet Semantic Analysis")
st.markdown("### Analyze Tweets using AI Models")
st.divider()

# Example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button(" Example: Positive"):
        st.session_state.text = "I love this product!"
with col2:
    if st.button(" Example: Sad"):
        st.session_state.text = "I feel very sad today"

# Input
text = st.text_area(" Enter Tweet / Comment", value=st.session_state.get("text", ""))

# Task selection
task = st.selectbox(
    " Select Task",
    ["Sentiment", "Emotion", "Hate Speech", "Irony"]
)

# Best models (Auto mode)
BEST_MODELS = {
    "Sentiment": "cardiffnlp/twitter-roberta-base-sentiment",
    "Emotion": "j-hartmann/emotion-english-distilroberta-base",
    "Hate Speech": "unitary/toxic-bert",
    "Irony": "cardiffnlp/twitter-roberta-base-irony"
}

# Model mapping
def get_model(task, model_choice):
    if model_choice == "Auto":
        return BEST_MODELS[task]

    elif model_choice == "RoBERTa":
        return BEST_MODELS[task]

    elif model_choice == "BERT":
        models = {
            "Sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
            "Emotion": "bhadresh-savani/bert-base-uncased-emotion",
            "Hate Speech": "unitary/toxic-bert",
            "Irony": BEST_MODELS["Irony"]
        }
        return models[task]

    elif model_choice == "ALBERT":
        models = {
            "Sentiment": "textattack/albert-base-v2-imdb",
            "Emotion": "bhadresh-savani/albert-base-v2-emotion",
            "Hate Speech": "unitary/toxic-bert",
            "Irony": BEST_MODELS["Irony"]
        }
        return models[task]

# Load model
@st.cache_resource
def load_pipeline(model_name):
    return pipeline("text-classification", model=model_name)

# Emoji mapping
emoji_map = {
    "positive": "😊",
    "negative": "😡",
    "neutral": "😐",
    "joy": "😁",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "toxic": "⚠️",
    "non-toxic": "✅",
    "irony": "😏"
}

# Confidence interpretation
def confidence_text(score):
    if score > 0.8:
        return "High Confidence "
    elif score > 0.5:
        return "Medium Confidence "
    else:
        return "Low Confidence "

# History
if "history" not in st.session_state:
    st.session_state.history = []

# Analyze
if st.button(" Analyze"):
    if text.strip() == "":
        st.warning(" Please enter some text")
    elif len(text.split()) < 3:
        st.warning(" Enter a meaningful sentence")
    else:
        model_name = get_model(task, model_choice)

        with st.spinner(" Analyzing..."):
            model = load_pipeline(model_name)
            results = model(text)   # ✅ Only top prediction

        st.success(" Analysis Complete")

        # Show only one result
        res = results[0]
        label = res["label"].lower()
        score = res["score"]

        emoji = emoji_map.get(label, "")

        st.markdown(f"""
### 🔹 Result
 **Label:** {label} {emoji}  
 **Confidence:** {score:.4f}  
 **{confidence_text(score)}**
""")

        st.progress(float(score))

        # Save history
        st.session_state.history.append((text, res["label"]))

# Show history
if st.session_state.history:
    st.divider()
    st.markdown("### 🧾 Previous Predictions")
    for t, r in st.session_state.history[-5:]:
        st.write(f" {t} → {r}")
