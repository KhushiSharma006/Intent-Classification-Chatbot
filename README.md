# 🤖 Intent Classification Chatbot

An NLP-based chatbot that classifies user queries into predefined intents using **TF-IDF Vectorization** and **Logistic Regression**, with an interactive **Streamlit** dashboard for predictions and dataset analytics.

---

## 🚀 Features

- 💬 Interactive chatbot interface
- 🎯 Intent prediction with confidence score
- 📊 Dataset analytics dashboard
- 🔍 Model inspection and vocabulary analysis
- ⚡ Fast and responsive Streamlit application

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Framework | Streamlit |
| Machine Learning | Scikit-learn |
| NLP | NLTK |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |

---

## 📂 Project Structure

```text
Intent-Classification-Chatbot/
│── app.py
│── chatbot_model.pkl
│── vectorizer.pkl
│── chatbot-dataset.csv
│── requirements.txt
│── README.md
└── assets/
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/KhushiSharma006/Intent-Classification-Chatbot.git
cd Intent-Classification-Chatbot
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📊 Dashboard

### 💬 Interactive Chatbot
- Predicts user intent
- Displays confidence score
- Returns the appropriate response

### 📈 Dataset Analytics
- Intent distribution
- Vocabulary size
- Text length analysis
- Dataset preview

### 🔬 Model Inspection
- TF-IDF vocabulary
- Top feature weights
- Pipeline overview

---

## 🧠 NLP Pipeline

```text
Input Text
    │
    ▼
Lowercase
    │
    ▼
Remove Punctuation
    │
    ▼
Tokenization
    │
    ▼
Stopword Removal
    │
    ▼
Stemming
    │
    ▼
TF-IDF Vectorizer
    │
    ▼
Logistic Regression
    │
    ▼
Predicted Intent
```


---

## 🔮 Future Enhancements

- 🎙️ Voice-based interaction
- 🌐 Multi-language support
- 🤖 Transformer-based models (BERT)
- 💾 Database integration
- 👤 User authentication

---

## 👩‍💻 Author

**Khushi Sharma**

- GitHub: https://github.com/KhushiSharma006

---

⭐ **If you found this project helpful, consider giving it a star!**
