# 🧠 Advanced Multi-Model NER Research System

A transformer-based Named Entity Recognition (NER) research platform built using Hugging Face Transformers, FastAPI, and Streamlit.

This system performs real-time entity extraction from text using multiple transformer architectures including:

- 🤖 BERT
- 🤖 RoBERTa
- 🤖 DeBERTa

The platform identifies:

- 👤 Persons
- 🏢 Organizations
- 📍 Locations
- 🏷️ Miscellaneous entities

The project combines modern NLP research workflows with interactive deployment-ready architecture.

---

# 🚀 Features

## ✅ Multi-Model Transformer Comparison
Compare predictions across:
- BERT
- RoBERTa
- DeBERTa

## ✅ Interactive Streamlit Frontend
Modern UI with:
- highlighted entities
- charts
- confidence scores
- statistics dashboard

## ✅ FastAPI Backend
REST API serving transformer inference in real time.

## ✅ Real-Time NER Inference
Input custom text and instantly receive entity predictions.

## ✅ Confidence Score Tracking
Displays prediction confidence for each detected entity.

## ✅ Entity Highlighting
Entities are visually highlighted directly inside text.

## ✅ Comparative Transformer Benchmarking
Analyze behavioral differences between transformer architectures.

---

# 🏗️ Project Architecture

User Input  
↓  
Streamlit Frontend  
↓  
FastAPI Backend  
↓  
Selected Transformer Model  
(BERT / RoBERTa / DeBERTa)  
↓  
Entity Predictions  
↓  
Visualization + Analytics  

---

# 🤖 Model Status

| Model | Status |
|---|---|
| BERT | Stable baseline model |
| RoBERTa | Best performing model |
| DeBERTa | Experimental model |

---

# 📊 Research Observations

- RoBERTa demonstrated stronger real-world entity recognition performance.
- BERT provided stable and reliable baseline predictions.
- DeBERTa encountered training instability on constrained hardware environments.

Example observation:

> RoBERTa successfully identified entities such as “Elon Musk” in cases where BERT failed.

---

# 🛠️ Tech Stack

## Machine Learning
- Hugging Face Transformers
- PyTorch
- BERT
- RoBERTa
- DeBERTa

## Backend
- FastAPI
- Uvicorn

## Frontend
- Streamlit

## Dataset
- CoNLL-2003

---

# 📂 Project Structure

```text
advanced-ner-research/
│
├── src/
│   ├── training/
│   ├── inference/
│   ├── api/
│   └── frontend/
│
├── models/
│   ├── bert_ner/
│   ├── roberta_ner/
│   └── deberta_ner/
│
├── notebooks/
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <your-repository-link>
cd advanced-ner-research
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Application

## Start FastAPI Backend

```bash
uvicorn src.api.app:app --reload
```

## Start Streamlit Frontend

```bash
streamlit run src/frontend/app.py
```

---

# 🧪 Example Input

```text
Google hired Sundar Pichai in California.
```

Example entities detected:
- Google → ORG
- Sundar Pichai → PER
- California → LOC

---

# 📈 Future Improvements

- Domain-specific NER
- Resume parsing NER
- Biomedical entity recognition
- Model fine-tuning optimization
- Cloud deployment
- Hugging Face Spaces integration

---

# 🌟 Project Highlights

- Multi-model transformer inference
- Research-oriented NLP workflow
- Interactive visualization dashboard
- Real-time entity extraction
- Deployment-ready architecture

---

# 📄 License

This project is intended for research and educational purposes.