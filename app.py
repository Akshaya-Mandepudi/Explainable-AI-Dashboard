import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# --------------------------------------------------
# 🌈 PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Dashboard",
    layout="wide"
)

# --------------------------------------------------
# 🎨 CUSTOM CSS (CENTERED & PREMIUM UI)
# --------------------------------------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Center container */
.main .block-container {
    max-width: 1100px;
    margin: auto;
    padding-top: 2rem;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff;
    font-weight: 700;
}

/* Cards */
.card {
    background-color: rgba(255, 255, 255, 0.08);
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
}

/* Prediction badges */
.pred-good {
    background-color: #2ecc71;
    padding: 12px;
    border-radius: 12px;
    color: black;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}

.pred-bad {
    background-color: #e74c3c;
    padding: 12px;
    border-radius: 12px;
    color: white;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 🧠 CENTERED TITLE
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-bottom:35px;">
        <h1> Explainable AI Dashboard</h1>
        <h4 style="color:#dcdcdc;">
            Understand <i>why</i> a model makes decisions using <b>SHAP</b> and <b>LIME</b>
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 📦 LOAD MODEL & DATA
# --------------------------------------------------
model = joblib.load("model/rf_model.pkl")
shap_explainer = joblib.load("model/shap_explainer.pkl")
X_train = joblib.load("model/X_train.pkl")

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["No Default", "Default"],
    mode="classification"
)

# --------------------------------------------------
# 📤 FILE UPLOADER (CENTERED)
# --------------------------------------------------
st.markdown('<div class="card" style="text-align:center;">📂 <b>Upload CSV File</b></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # --------------------------------------------------
    # 📊 UPLOADED DATA
    # --------------------------------------------------
    st.markdown('<div class="card">📊 <b>Uploaded Data</b></div>', unsafe_allow_html=True)
    st.dataframe(data, use_container_width=True)

    # Drop target column if present
    if "target" in data.columns:
        input_data = data.drop(columns=["target"])
    else:
        input_data = data.copy()

    st.markdown('<div class="card">🧾 <b>Input Data Used for Prediction</b></div>', unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True)

    # --------------------------------------------------
    # ✅ MODEL PREDICTIONS
    # --------------------------------------------------
    predictions = model.predict(input_data)

    st.markdown('<div class="card">✅ <b>Model Predictions</b></div>', unsafe_allow_html=True)

    for i, pred in enumerate(predictions):
        if pred == 0:
            st.markdown(
                f'<div class="pred-good">Row {i+1}: No Default ✅</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="pred-bad">Row {i+1}: Default Risk ⚠️</div>',
                unsafe_allow_html=True
            )

    # --------------------------------------------------
    # 🔍 SHAP EXPLANATION (GLOBAL)
    # --------------------------------------------------
    st.markdown('<div class="card">🔎 <b>SHAP – Global Feature Importance</b></div>', unsafe_allow_html=True)

    shap_values = shap_explainer.shap_values(input_data)

    # Handle SHAP versions safely
    if isinstance(shap_values, list):
        shap_plot_values = shap_values[1]
    else:
        shap_plot_values = shap_values[:, :, 1]

    fig = plt.figure()
    shap.summary_plot(shap_plot_values, input_data, show=False)
    st.pyplot(fig)

    # --------------------------------------------------
    # 🧩 LIME EXPLANATION (LOCAL)
    # --------------------------------------------------
    st.markdown('<div class="card">🧩 <b>LIME – Local Explanation (First Instance)</b></div>', unsafe_allow_html=True)

    sample = input_data.iloc[0]

    lime_exp = lime_explainer.explain_instance(
        sample.values,
        model.predict_proba,
        num_features=5
    )

    lime_df = pd.DataFrame(
        lime_exp.as_list(),
        columns=["Feature Condition", "Impact on Prediction"]
    )

    st.dataframe(lime_df, use_container_width=True)

# --------------------------------------------------
# 🎉 FOOTER
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:#cfcfcf;">
        Built using <b>Streamlit</b>, <b>SHAP</b> & <b>LIME</b>
    </div>
    """,
    unsafe_allow_html=True
)
