import string
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------------------------------------
# Page Setup & Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Intent Analytics & Chatbot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data securely
@st.cache_resource
def setup_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords")
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

setup_nltk()

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# ---------------------------------------------------------
# Utility & Pipeline Functions
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Preprocess raw input text matching the training pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

@st.cache_data
def load_dataset():
    """Load and preprocess dataset for analysis."""
    try:
        df = pd.read_csv("chatbot-dataset.csv")
        df["clean_text"] = df["text"].apply(clean_text)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_artifacts():
    """Load vectorizer and trained Logistic Regression model."""
    try:
        with open("chatbot_model.pkl", "rb") as f_model:
            model = pickle.load(f_model)
        with open("vectorizer.pkl", "rb") as f_vec:
            vectorizer = pickle.load(f_vec)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

df_data = load_dataset()
model, vectorizer = load_artifacts()

# ---------------------------------------------------------
# Navigation / Sidebar Options
# ---------------------------------------------------------
st.sidebar.title("🤖 Navigation")
app_mode = st.sidebar.radio(
    "Choose View Mode",
    ["Interactive Chatbot", "Dataset Analytics", "Model Inspection"]
)

# System Health Checks on Sidebar
st.sidebar.markdown("---")
st.sidebar.caption("### System Status")
st.sidebar.markdown(f"**Dataset Loaded:** {'✅' if not df_data.empty else '❌'}")
st.sidebar.markdown(f"**Model Loaded:** {'✅' if model is not None else '❌'}")
st.sidebar.markdown(f"**Vectorizer Loaded:** {'✅' if vectorizer is not None else '❌'}")

# ---------------------------------------------------------
# View 1: Interactive Chatbot
# ---------------------------------------------------------
if app_mode == "Interactive Chatbot":
    st.title("💬 Intent Classification Assistant")
    st.markdown("Interact with the trained NLP model in real-time.")

    if model is None or vectorizer is None or df_data.empty:
        st.warning(
            "Model, vectorizer, or dataset files not found. "
            "Please ensure `chatbot_model.pkl`, `vectorizer.pkl`, and `chatbot-dataset.csv` exist."
        )
    else:
        # Initialize conversation state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! How can I help you today?"}
            ]

        # Render conversation thread (includes intent & confidence if present)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "intent" in msg and "confidence" in msg:
                    st.caption(f"🎯 **Predicted Intent:** `{msg['intent']}` | 📊 **Confidence:** `{msg['confidence']:.2f}%`")

        # Chat Input Area
        if prompt := st.chat_input("Type your message here..."):
            # Append user utterance to state
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Inference
            clean_input = clean_text(prompt)
            vec_input = vectorizer.transform([clean_input])
            predicted_intent = model.predict(vec_input)[0]
            
            # Predict Probabilities for Confidence
            probs = model.predict_proba(vec_input)[0]
            max_prob = np.max(probs) * 100

            # Retrieve response mapped to predicted intent
            matched_responses = df_data[df_data["intent"] == predicted_intent]["response"]
            bot_response = matched_responses.iloc[0] if not matched_responses.empty else "I couldn't process that request."

            # Append assistant response along with intent and confidence metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "intent": predicted_intent,
                "confidence": max_prob
            })

            # Rerun so Streamlit updates and renders the new state cleanly
            st.rerun()

# ---------------------------------------------------------
# View 2: Dataset Analytics
# ---------------------------------------------------------
elif app_mode == "Dataset Analytics":
    st.title("📊 Dataset Exploratory Analytics")
    
    if df_data.empty:
        st.error("No dataset available to render analytics.")
    else:
        # Top KPI Cards
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Samples", len(df_data))
        kpi2.metric("Unique Intents", df_data["intent"].nunique())
        kpi3.metric("Vocabulary Size", len(vectorizer.get_feature_names_out()) if vectorizer else "N/A")

        st.markdown("---")

        col1, col2 = st.columns(2)

        # Chart 1: Intent Class Distribution
        with col1:
            st.subheader("Class Distribution (Intents)")
            intent_counts = df_data["intent"].value_counts().reset_index()
            intent_counts.columns = ["Intent", "Count"]
            fig_bar = px.bar(
                intent_counts,
                x="Count",
                y="Intent",
                orientation="h",
                color="Count",
                color_continuous_scale="Viridis",
                title="Sample Count per Intent"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Chart 2: Utterance Length Distribution
        with col2:
            st.subheader("Utterance Word Lengths")
            df_data["char_len"] = df_data["text"].apply(lambda x: len(str(x)))
            fig_hist = px.histogram(
                df_data,
                x="char_len",
                nbins=15,
                color="intent",
                title="Text Character Length by Intent"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Data Table View
        st.subheader("📋 Dataset Inspection")
        st.dataframe(df_data[["text", "clean_text", "intent", "response"]], use_container_width=True)

# ---------------------------------------------------------
# View 3: Model Inspection & Pipeline Technicals
# ---------------------------------------------------------
elif app_mode == "Model Inspection":
    st.title("🔬 Model & Pipeline Architecture")

    st.markdown("""
    ### Pipeline Workflow
    1. **Preprocessing:** Lowercasing $\\rightarrow$ Punctuation Stripping $\\rightarrow$ Tokenization $\\rightarrow$ Porter Stemming & English Stop-Words Filter.
    2. **Feature Extraction:** `TfidfVectorizer` mapping cleaned text to high-dimensional sparse representations.
    3. **Classification:** Multi-class `LogisticRegression` optimized via L-BFGS.
    """)

    st.markdown("---")

    if vectorizer and model:
        st.subheader("TF-IDF Vectorizer Parameters")
        vocab = vectorizer.get_feature_names_out()
        
        c1, c2 = st.columns(2)
        with c1:
            st.json({
                "Total Features Extracted": len(vocab),
                "Stop Words Applied": "NLTK English",
                "Stemmer Algorithm": "PorterStemmer"
            })
        with c2:
            st.write("#### Extracted Vocabulary Sample")
            st.write(list(vocab[:20]))

        st.subheader("Logistic Regression Coefficients Analysis")
        selected_intent = st.selectbox("Select Target Intent to View Top Features", model.classes_)

        # Extract coefficient safely (works for both binary and multi-class models)
        class_idx = list(model.classes_).index(selected_intent)
        
        if model.coef_.ndim == 1 or model.coef_.shape[0] == 1:
            coefficients = model.coef_[0] if class_idx == 1 else -model.coef_[0]
        else:
            coefficients = model.coef_[class_idx]
        
        top_indices = np.argsort(coefficients)[-10:]
        top_features = [vocab[i] for i in top_indices]
        top_scores = coefficients[top_indices]

        fig_coef = px.bar(
            x=top_scores,
            y=top_features,
            orientation="h",
            labels={"x": "Coefficient Weight", "y": "Feature Token"},
            title=f"Top 10 Feature Weights for Intent: '{selected_intent}'"
        )
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        st.error("Model or vectorizer pickles not found. Please train and export models first.")