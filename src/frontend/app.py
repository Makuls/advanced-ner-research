import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Advanced NER Research System",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #FFFFFF;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 20px;
    color: #BBBBBB;
    margin-bottom: 30px;
}

.entity-card {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    background-color: #1C1F26;
}

.highlight-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1C1F26;
    font-size: 22px;
    line-height: 2.2;
}

.stats-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1C1F26;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<div class='big-title'>🧠 Advanced NER Research System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Compare BERT, RoBERTa, and DeBERTa for Named Entity Recognition</div>",
    unsafe_allow_html=True
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.info("""
Available Models:

• BERT  
• RoBERTa  
• DeBERTa  

Framework: Hugging Face Transformers  
Backend: FastAPI  
Frontend: Streamlit  
Task: Named Entity Recognition
""")

st.sidebar.success("✅ Multi-Model System Ready")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Model Status")

st.sidebar.markdown("""
🟢 **BERT** → Stable baseline model

🟢 **RoBERTa** → Best performing model

🟡 **DeBERTa** → Experimental model
""")

# -----------------------------
# MODEL SELECTION
# -----------------------------
selected_model = st.selectbox(
    "🤖 Choose Transformer Model",
    ["bert", "roberta", "deberta"]
)

# -----------------------------
# ABOUT PROJECT
# -----------------------------
st.markdown("---")

st.subheader("📚 About This Project")

st.write("""
This project is a multi-model Named Entity Recognition (NER) system built using
Transformer architectures including BERT, RoBERTa, and DeBERTa.

The system performs real-time entity extraction for:
- 👤 Persons
- 🏢 Organizations
- 📍 Locations
- 🏷️ Miscellaneous entities

Key Features:
- Multi-model transformer comparison
- Real-time NER inference
- Interactive Streamlit dashboard
- FastAPI backend integration
- Entity visualization and analytics
- Confidence score tracking

Current Model Status:
- 🟢 BERT → Stable baseline model
- 🟢 RoBERTa → Best performing model
- 🟡 DeBERTa → Experimental model

RoBERTa currently demonstrates the strongest real-world entity recognition performance.
""")

# -----------------------------
# INPUT
# -----------------------------
text_input = st.text_area(
    "✍️ Enter Text",
    height=180,
    placeholder="Example: Microsoft hired Satya Nadella in Seattle."
)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Analyze Entities"):

    if text_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        payload = {
            "text": text_input,
            "model": selected_model
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            data = response.json()

            entities = data["entities"]

            st.success(f"✅ Using {selected_model.upper()} model")

            # -----------------------------
            # HIGHLIGHTED TEXT
            # -----------------------------
            highlighted_text = text_input

            for entity in entities:

                entity_text = entity["entity"]
                label = entity["label"]

                if label == "ORG":
                    color = "#90EE90"

                elif label == "PER":
                    color = "#87CEFA"

                elif label == "LOC":
                    color = "#FFD580"

                else:
                    color = "#DDA0DD"

                highlighted_entity = (
                    f"<span style='background-color:{color}; "
                    f"padding:6px; border-radius:6px; color:black; "
                    f"font-weight:bold;'>"
                    f"{entity_text} ({label})</span>"
                )

                highlighted_text = highlighted_text.replace(
                    entity_text,
                    highlighted_entity
                )

            st.subheader("✨ Highlighted Text")

            st.markdown(
                f"<div class='highlight-box'>{highlighted_text}</div>",
                unsafe_allow_html=True
            )

            # -----------------------------
            # ENTITY STATS
            # -----------------------------
            st.subheader("📈 Entity Statistics")

            col1, col2, col3 = st.columns(3)

            org_count = sum(
                1 for e in entities
                if e["label"] == "ORG"
            )

            per_count = sum(
                1 for e in entities
                if e["label"] == "PER"
            )

            loc_count = sum(
                1 for e in entities
                if e["label"] == "LOC"
            )

            with col1:
                st.metric(
                    "🏢 Organizations",
                    org_count
                )

            with col2:
                st.metric(
                    "👤 Persons",
                    per_count
                )

            with col3:
                st.metric(
                    "📍 Locations",
                    loc_count
                )

            # -----------------------------
            # ENTITY DISTRIBUTION CHART
            # -----------------------------
            st.subheader("📊 Entity Distribution")

            label_counts = {}

            for entity in entities:

                label = entity["label"]

                if label not in label_counts:
                    label_counts[label] = 0

                label_counts[label] += 1

            chart_data = pd.DataFrame({
                "Entity Type": list(label_counts.keys()),
                "Count": list(label_counts.values())
            })

            st.bar_chart(
                chart_data.set_index(
                    "Entity Type"
                )
            )

            # -----------------------------
            # ENTITY CARDS
            # -----------------------------
            st.subheader("📋 Detected Entities")

            for entity in entities:

                label = entity["label"]

                if label == "ORG":
                    icon = "🏢"

                elif label == "PER":
                    icon = "👤"

                elif label == "LOC":
                    icon = "📍"

                else:
                    icon = "🏷️"

                with st.container():

                    st.markdown("---")

                    col1, col2 = st.columns(
                        [3, 2]
                    )

                    with col1:

                        st.markdown(
                            f"### {icon} {entity['entity']}"
                        )

                        st.info(
                            f"Label: {label}"
                        )

                    with col2:

                        st.success(
                            f"Confidence: {entity['score']:.4f}"
                        )

        else:
            st.error("❌ API request failed.")