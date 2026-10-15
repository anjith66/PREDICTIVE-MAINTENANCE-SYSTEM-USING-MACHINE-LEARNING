# ⚙️ AI-Powered Predictive Maintenance System Using Machine Learning

An intelligent predictive maintenance application that estimates engine Remaining Useful Life (RUL), detects abnormal engine behaviour, explains machine learning predictions, and generates AI-powered maintenance recommendations.

**Live Demo:** https://ztggxxls8iq2krz6dwpvin.streamlit.app/

---

## 📌 Project Overview

Unexpected equipment failures can cause costly downtime, production losses, and maintenance expenses. Traditional maintenance approaches often depend on fixed schedules or respond only after equipment has failed.

This project explores a data-driven approach to predictive maintenance using machine learning and artificial intelligence.

The system uses engine sensor data from the NASA C-MAPSS dataset to estimate Remaining Useful Life (RUL). It combines a Random Forest regression model, Isolation Forest anomaly detection, SHAP explainability, FAISS-based Retrieval-Augmented Generation (RAG), and Google's Gemini API to provide engine health insights and contextual maintenance recommendations.

The application is built with Python and Streamlit, providing an interactive dashboard for exploring predictions and maintenance insights.

## 🎯 Project Objectives

- Predict the estimated Remaining Useful Life of an engine.
- Identify potentially abnormal engine behaviour using anomaly detection.
- Classify engine conditions into risk categories.
- Explain predictions using SHAP feature importance.
- Retrieve relevant maintenance information using FAISS.
- Generate contextual maintenance recommendations using Gemini AI.
- Provide an interactive web application for viewing engine health insights.

## ✨ Key Features

### 1. Remaining Useful Life Prediction

Uses a trained Random Forest regression model to estimate how many operating cycles an engine may have remaining before reaching the dataset's defined end-of-life condition.

### 2. Anomaly Detection

Uses an Isolation Forest model to identify sensor patterns that differ from the patterns learned during training.

### 3. Engine Risk Classification

Converts the predicted RUL into prototype risk categories:

| Estimated RUL | Risk category |
|---|---|
| 10 cycles or less | Critical |
| 11–30 cycles | High |
| 31–60 cycles | Medium |
| More than 60 cycles | Low |

These thresholds are project-specific demonstration settings, not universal industrial safety limits.

### 4. Explainable AI with SHAP

Uses SHAP (SHapley Additive exPlanations) to help interpret model predictions and identify which input features contribute to an individual prediction.

This makes the model's output easier to investigate than a prediction alone.

### 5. AI Maintenance Recommendations

Integrates Google's Gemini API to generate maintenance guidance based on the information provided to the AI component.

### 6. Retrieval-Augmented Generation (RAG)

Uses FAISS to search a maintenance knowledge collection and retrieve relevant information that can provide context for AI-generated responses.

### 7. Interactive Streamlit Dashboard

Provides a web interface for inspecting engine predictions, risk classifications, anomaly results, and maintenance insights.

---

## 🏗️ System Architecture

```text
NASA C-MAPSS Dataset
        |
        v
Engine Sensor Data Processing
        |
        v
Trained Random Forest Model
        |
        v
Remaining Useful Life Prediction
        |
        v
Engine Risk Classification
        |
        +----------------------+
        |                      |
        v                      v
Isolation Forest          SHAP Explainability
Anomaly Detection         Feature Contributions
        |                      |
        +----------+-----------+
                   |
                   v
       FAISS Maintenance Knowledge Search
                   |
                   v
           Gemini AI Integration
                   |
                   v
      Contextual Maintenance Recommendations
                   |
                   v
          Streamlit Web Dashboard
```

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and processing |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and anomaly detection |
| Random Forest | Remaining Useful Life regression |
| Isolation Forest | Anomaly detection |
| SHAP | Model explainability |
| FAISS | Similarity search for maintenance information |
| Sentence Transformers | Text embeddings for semantic retrieval |
| Google Gemini API | AI-generated maintenance insights |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| Git and GitHub | Version control and project hosting |
| Streamlit Community Cloud | Web application deployment |

## 📊 Dataset

This project uses the **NASA C-MAPSS turbofan engine degradation dataset**, specifically the FD001 subset.

The dataset contains simulated engine operating conditions and sensor measurements across operating cycles.

The data includes:

- Engine ID
- Operating cycle
- Three operational setting features
- Twenty-one sensor measurements

The features are used to study engine degradation patterns and estimate Remaining Useful Life.

**Dataset source:** [NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

> The dataset is simulated engine degradation data. Model performance on this dataset does not automatically establish performance on real industrial equipment.

## 🤖 Machine Learning Models

Three regression approaches were evaluated during the project.

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Random Forest Regressor | 23.854 | 31.413 | 0.771 |
| XGBoost Regressor | 24.189 | 31.993 | 0.763 |
| Linear Regression | 25.155 | 31.656 | 0.768 |

*Values are approximate project evaluation results. Reproduce the evaluation with the same dataset split and preprocessing to verify them.*

### Selected Model: Random Forest

The Random Forest model achieved the lowest MAE and RMSE among the three evaluated models and the highest R² score in this comparison.

**Model configuration:**

- Algorithm: Random Forest Regressor
- Number of estimators: 200
- Maximum depth: 15
- Random state: 42

### Evaluation Metrics

- **MAE (Mean Absolute Error):** The average absolute difference between actual and predicted RUL.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily.
- **R² Score:** Measures how well the model explains variation in the target values.

## 🔍 Anomaly Detection

The Isolation Forest model is used to identify potentially unusual engine sensor patterns.

The model uses an anomaly score and prediction output to distinguish observations considered unusual from those considered normal under its learned data distribution.

An anomaly alert does not by itself confirm that an engine is faulty. Further investigation and appropriate engineering checks are required.

## 🧠 Explainable AI

SHAP is used to analyse feature contributions to predictions.

Features such as operating cycle and individual sensor readings can influence the estimated RUL. The contribution of each feature depends on the particular observation and the trained model.

SHAP explanations describe model behaviour; they do not establish the physical cause of equipment degradation.

## 🔎 Retrieval-Augmented Generation

The maintenance information retrieval component uses the following workflow:

1. Represent maintenance text as numerical embeddings using Sentence Transformers.
2. Search the FAISS index for relevant information.
3. Retrieve matching maintenance knowledge.
4. Provide relevant context to the Gemini integration.
5. Generate a contextual maintenance response.

The quality of the response depends on the retrieved documents, prompt design, model availability, and the accuracy of the underlying information.

## 🛠️ Installation and Setup

### Prerequisites

Install the following before running the application:

- Python 3.10 or a compatible Python version supported by your dependencies
- pip
- Git
- A Google Gemini API key for the generative AI functionality

### Step 1: Clone the repository

```bash
git clone https://github.com/anjith66/PREDICTIVE-MAINTENANCE-SYSTEM-USING-MACHINE-LEARNING.git
cd PREDICTIVE-MAINTENANCE-SYSTEM-USING-MACHINE-LEARNING
```

If your actual repository name differs, replace the repository URL and folder name with your own.

### Step 2: Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS or Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

If your repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the packages used by the application:

```bash
pip install streamlit pandas numpy scikit-learn joblib shap faiss-cpu sentence-transformers google-genai python-dotenv
```

Use the compatible versions you tested with your project. Some packages may require additional system support or a compatible Python version.

### Step 4: Configure the Gemini API key

Create a file named `.env` in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the example value with your own API key.

Obtain an API key through [Google AI Studio](https://aistudio.google.com/).

**Security:** Never commit `.env` files, API keys, or other secrets to GitHub. Add `.env` to `.gitignore`.

### Step 5: Check model and index files

The application requires its trained model artifacts and any configured retrieval files.

Depending on your code and file paths, these may include:

```text
rf_model.joblib
isolation_model.joblib
latest_engine_state.csv
maintenance_documents.joblib
maintenance_index.faiss
```

Make sure the files are available at the locations expected by your application.

The Random Forest model file is large, so it may need to be hosted separately rather than committed directly to a normal GitHub repository. If you store it externally, document the download and setup steps here.

Do not load model files from untrusted sources.

### Step 6: Run the application

Run the Streamlit entry-point file:

```bash
streamlit run app.py
```

Replace `app.py` with the actual Python filename that contains your Streamlit application if it is named differently.

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

---

## 📁 Project Structure

An example project structure is shown below. Adjust the filenames to match your actual repository.

```text
PREDICTIVE-MAINTENANCE-SYSTEM-USING-MACHINE-LEARNING/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── predictive_maintenance_model/
│   ├── rf_model.joblib
│   ├── isolation_model.joblib
│   ├── latest_engine_state.csv
│   ├── maintenance_documents.joblib
│   └── maintenance_index.faiss
│
└── notebooks/
    └── model_training_and_evaluation.ipynb
```

**Important:** This is a suggested structure, not a guarantee that every listed file exists in the repository. Keep `.env` out of version control. Do not include large generated artifacts unless your hosting strategy supports them.

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

**Live application:** https://xjgnwxklxlpskdadthcyss.streamlit.app/

To deploy your own copy:

1. Push your project code to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select the repository, branch, and Streamlit entry-point file.
5. Add `GEMINI_API_KEY` under the application's secrets settings.
6. Ensure all required model files and retrieval assets are accessible.
7. Deploy and test the application.

Refer to the [Streamlit deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

### Large Model File Handling

GitHub blocks ordinary Git files larger than 100 MB. The Random Forest artifact used in this project is approximately 127 MB.

Possible approaches include:

- Git LFS, subject to its storage and bandwidth limits.
- External model or artifact hosting.
- A separate download step during deployment.

Keep the model-loading paths in the code consistent with the selected approach. Avoid committing a large model file and then assuming it has been removed from Git history simply because it was deleted from the current folder.

## ⚠️ Limitations

- Predictions depend on the dataset, training procedure, preprocessing, and model quality.
- RUL is an estimate, not a guaranteed failure time.
- Risk thresholds are demonstration settings and require domain-specific validation for real use.
- Anomaly detection identifies unusual patterns, not necessarily confirmed failures.
- SHAP explains model predictions but does not prove physical causation.
- Gemini-generated recommendations can be inaccurate or incomplete and should be independently verified.
- The application depends on the availability and configuration of its AI API and supporting artifacts.
- This project is a prototype for educational and analytical purposes, not a certified industrial safety system.

## 🚀 Future Enhancements

- Add real-time sensor data ingestion.
- Develop trend monitoring and historical engine health analysis.
- Improve validation using additional engine degradation datasets.
- Add automated alerts for high-risk observations.
- Improve maintenance knowledge retrieval and source attribution.
- Add model monitoring and retraining workflows.
- Evaluate prediction performance on independently collected industrial data.
- Add authentication, audit logging, and operational monitoring for production deployments.

## 🎓 Learning Outcomes

Through this project, I explored and integrated:

- Machine learning model training and evaluation.
- Regression-based Remaining Useful Life prediction.
- Unsupervised anomaly detection.
- Explainable AI using SHAP.
- Semantic search and Retrieval-Augmented Generation.
- Integration of a generative AI API.
- Building and deploying an interactive Streamlit application.
- Model artifact management and GitHub deployment considerations.

## 📚 References

1. NASA Prognostics Center of Excellence — [Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
2. Scikit-learn — [Machine Learning Documentation](https://scikit-learn.org/stable/)
3. SHAP — [Explainable AI Documentation](https://shap.readthedocs.io/)
4. FAISS — [Similarity Search Library](https://github.com/facebookresearch/faiss)
5. Sentence Transformers — [Documentation](https://www.sbert.net/)
6. Google — [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
7. Streamlit — [Documentation](https://docs.streamlit.io/)

## 👨‍💻 Author

**Anjith Shetty**

Engineering student focused on Artificial Intelligence, Data Science, and Machine Learning.

- GitHub: https://github.com/anjith66
- Live Project: https://xjgnwxklxlpskdadthcyss.streamlit.app/

---

⭐ If you find this project useful, consider starring the repository.

**Disclaimer:** This project is developed for learning, experimentation, and demonstration. Its predictions and generated recommendations should not be used as the sole basis for real-world maintenance or safety-critical decisions.
