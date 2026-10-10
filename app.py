import os
import urllib.request

import faiss
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

MODEL_DIR = "predictive_maintenance_model"
RELEASE_URL = "https://github.com/anjith66/PREDICTIVE-MAINTENANCE-SYSTEM-USING-MACHINE-LEARNING/releases/download/v1.0.0"

MODEL_FILES = [
    "rf_model.joblib",
    "isolation_model.joblib",
    "maintenance_documents.joblib",
    "maintenance_index.faiss",
    "latest_engine_state.csv",
]


# --------------------------------------------------
# Automatic Model File Downloader (for GitHub Releases)
# --------------------------------------------------

def download_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename in MODEL_FILES:
        file_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(file_path):
            file_url = f"{RELEASE_URL}/{filename}"
            print(f"Downloading {filename} from GitHub Release...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except Exception as e:
                st.error(f"Failed to download {filename}: {e}")


# --------------------------------------------------
# Load saved artifacts
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    # Automatically download large model/FAISS artifacts if running on Streamlit Cloud
    download_model_files()

    rf_model = joblib.load(
        os.path.join(MODEL_DIR, "rf_model.joblib")
    )
    isolation_model = joblib.load(
        os.path.join(MODEL_DIR, "isolation_model.joblib")
    )
    engines = pd.read_csv(
        os.path.join(MODEL_DIR, "latest_engine_state.csv")
    )
    documents = joblib.load(
        os.path.join(MODEL_DIR, "maintenance_documents.joblib")
    )
    index = faiss.read_index(
        os.path.join(MODEL_DIR, "maintenance_index.faiss")
    )
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
    explainer = shap.TreeExplainer(rf_model)

    return (
        rf_model,
        isolation_model,
        engines,
        documents,
        index,
        embedding_model,
        explainer,
    )


# --------------------------------------------------
# RAG retrieval
# --------------------------------------------------

def retrieve_documents(query, documents, index, embedding_model, k=3):
    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    ).astype("float32")

    k = min(k, len(documents))
    distances, indices = index.search(query_vector, k)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(documents):
            results.append(documents[idx])

    return "\n\n".join(results)


# --------------------------------------------------
# SHAP explanation
# --------------------------------------------------

def explain_engine(engine_row, explainer):
    excluded = {
        "RUL",
        "engine_id",
        "predicted_RUL",
        "anomaly",
        "anomaly_score",
        "risk_level",
        "health_status",
    }

    feature_names = [
        c for c in engine_row.index if c not in excluded
    ]
    features = engine_row[feature_names].to_frame().T

    values = np.asarray(
        explainer.shap_values(features)
    ).reshape(-1)

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP value": values,
    })
    explanation["Absolute impact"] = explanation[
        "SHAP value"
    ].abs()

    return explanation.sort_values(
        "Absolute impact", ascending=False
    )


# --------------------------------------------------
# Gemini recommendation
# --------------------------------------------------

def create_recommendation(
    engine_row,
    explanation,
    documents,
    index,
    embedding_model,
):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return (
            "Gemini API key is missing. Please add GEMINI_API_KEY "
            "to your .streamlit/secrets.toml file and restart the app."
        )

    client = genai.Client(api_key=api_key)

    top_features = explanation.head(5)
    feature_text = "\n".join(
        f"- {row['Feature']}: SHAP={row['SHAP value']:.3f}"
        for _, row in top_features.iterrows()
    )

    anomaly_text = (
        "YES" if int(engine_row["anomaly"]) == -1 else "NO"
    )

    query = (
        f"Maintenance guidance for predicted RUL "
        f"{engine_row['predicted_RUL']:.1f}, "
        f"risk {engine_row['risk_level']}, "
        f"anomaly {anomaly_text}. "
        f"Important features: "
        f"{', '.join(top_features['Feature'].tolist())}"
    )

    context = retrieve_documents(
        query, documents, index, embedding_model
    )

    prompt = f"""
You are an AI predictive maintenance assistant.

ENGINE STATUS
Engine ID: {int(engine_row['engine_id'])}
Cycle: {int(engine_row['cycle'])}
Predicted RUL: {engine_row['predicted_RUL']:.1f} cycles
Risk level: {engine_row['risk_level']}
Health status: {engine_row['health_status']}
Anomaly detected: {anomaly_text}
Anomaly score: {engine_row['anomaly_score']:.4f}

TOP MODEL CONTRIBUTORS
{feature_text}

RETRIEVED MAINTENANCE KNOWLEDGE
{context}

Write these sections:
1. Machine condition
2. Key indicators
3. Anomaly assessment
4. Possible causes
5. Recommended inspection
6. Risk summary

Rules:
- Treat RUL as a model estimate, not a guaranteed failure time.
- Do not claim a confirmed root cause.
- Do not invent sensor meanings or procedures.
- Distinguish model evidence from possibilities.
- Use the retrieved knowledge as guidance, not proof.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    return response.text or "No recommendation was returned."


# --------------------------------------------------
# Main dashboard
# --------------------------------------------------

st.title("⚙️ AI-Powered Predictive Maintenance")
st.caption(
    "Random Forest • Isolation Forest • SHAP • FAISS RAG • Gemini"
)

try:
    (
        rf_model,
        isolation_model,
        engines,
        documents,
        index,
        embedding_model,
        explainer,
    ) = load_artifacts()
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.info(
        "Check that the predictive_maintenance_model folder "
        "is accessible and release download links are valid."
    )
    st.stop()

engine_ids = sorted(engines["engine_id"].astype(int).unique())

selected_id = st.sidebar.selectbox(
    "Select engine",
    engine_ids,
)

engine_row = engines[
    engines["engine_id"].astype(int) == selected_id
].iloc[0]

rul = float(engine_row["predicted_RUL"])
risk = str(engine_row["risk_level"])
anomaly = int(engine_row["anomaly"]) == -1
health = str(engine_row["health_status"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted RUL", f"{rul:.1f} cycles")
c2.metric("Risk level", risk)
c3.metric("Anomaly", "Detected" if anomaly else "Not detected")
c4.metric("Current cycle", int(engine_row["cycle"]))

if health == "CRITICAL":
    st.error(f"Health status: {health}")
elif health in {"HIGH RISK", "MEDIUM RISK", "ANOMALY DETECTED"}:
    st.warning(f"Health status: {health}")
else:
    st.success(f"Health status: {health}")

st.caption(
    "RUL and risk are model outputs for this prototype. "
    "An anomaly indicates unusual sensor behavior, not confirmed failure."
)

st.divider()
st.subheader("Model explainability")

try:
    explanation = explain_engine(engine_row, explainer)
    chart_data = explanation.head(10).set_index("Feature")[
        ["SHAP value"]
    ]
    st.bar_chart(chart_data)
    st.dataframe(
        explanation.head(10),
        use_container_width=True,
        hide_index=True,
    )
except Exception as exc:
    st.error(f"SHAP explanation failed: {exc}")
    st.stop()

st.divider()
st.subheader("AI maintenance recommendation")

if st.button("Analyze selected engine", type="primary"):
    with st.spinner("Retrieving maintenance guidance and querying Gemini..."):
        try:
            recommendation = create_recommendation(
                engine_row,
                explanation,
                documents,
                index,
                embedding_model,
            )
            st.markdown(recommendation)
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            st.caption(
                "Check your API key, model availability, and internet connection."
            )

with st.expander("View saved engine data"):
    st.dataframe(
        engine_row.to_frame("Value"),
        use_container_width=True,
    )

st.divider()
st.caption("Predictive Maintenance Prototype | NASA C-MAPSS FD001")
