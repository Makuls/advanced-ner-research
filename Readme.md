# 🧠 Advanced NER Research System

A transformer-based Named Entity Recognition (NER) system built using Hugging Face Transformers, FastAPI, and Streamlit.

This project performs real-time entity extraction from text and identifies:

- 👤 Persons
- 🏢 Organizations
- 📍 Locations

The system includes:

- Fine-tuned BERT NER model
- FastAPI backend
- Streamlit frontend
- Real-time inference pipeline
- Interactive entity visualization
- Confidence scoring
- Entity statistics dashboard

---

# 🚀 Features

## ✅ Transformer-Based NER
Fine-tuned BERT model for accurate entity recognition.

## ✅ Interactive Frontend
Beautiful Streamlit UI with highlighted entities and analytics.

## ✅ FastAPI Backend
REST API serving predictions in real time.

## ✅ Real-Time Inference
Input text and instantly receive predictions.

## ✅ Confidence Scores
Displays prediction confidence for each entity.

## ✅ Entity Highlighting
Entities are highlighted directly inside text.

---

# 🏗️ Project Architecture

User Input
↓
Streamlit Frontend
↓
FastAPI Backend
↓
BERT NER Model
↓
Entity Predictions
↓
Visualization + Statistics

---

# 🛠️ Tech Stack

## Machine Learning
- Hugging Face Transformers
- PyTorch
- BERT

## Backend
- FastAPI
- Uvicorn

## Frontend
- Streamlit

## Dataset
- CoNLL-2003

---

# 📂 Project Structure

advanced-ner-research/
│
├── src/
│   ├── training/
│   ├── inference/
│   ├── api/
│   └── frontend/
│
├── models/
├── notebooks/
├── requirements.txt
└── README.md

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <your-repo-link>
cd advanced-ner-research