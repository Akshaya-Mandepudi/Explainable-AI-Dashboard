#  Explainable AI Dashboard

An interactive **Explainable AI Dashboard** built using **Streamlit**, **SHAP**, and **LIME** to interpret predictions made by a machine learning model.

This project focuses on **credit risk prediction using structured financial (tabular) data**, enabling transparency and interpretability in model decisions.


##  Features

- Upload CSV files containing credit-related data
- Predict credit default risk using a trained **Random Forest classifier**
- **Global explanations** using SHAP (feature importance)
- **Local explanations** using LIME (instance-level interpretability)
- Clean, professional, and user-friendly dashboard UI
- Automatically removes the target column during inference to avoid feature mismatch


##  Technologies Used

- Python  
- Streamlit  
- scikit-learn  
- SHAP  
- LIME  
- pandas  
- matplotlib  
- joblib  


## 📂 Project Structure

Explainable-AI-Dashboard/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── credit_data.csv
│
├── model/
│   ├── train_model.py          # Trains Random Forest model
│   ├── explain_shap.py         # Creates SHAP explainer & values
│   ├── explain_lime.py         # Generates LIME explanations
│   ├── rf_model.pkl            # Saved trained model
│   ├── X_train.pkl             # Training data for explainers
│   ├── shap_explainer.pkl      # Saved SHAP explainer
│   └── shap_values.pkl         # Saved SHAP values

  

##  Dataset

The dataset represents a **credit risk prediction problem** and includes the following columns:

- age  
- income  
- credit_score  
- loan_amount  
- employment_years  
- target (0 = No Default, 1 = Default)

⚠️ The `target` column is used only during training and is automatically removed during prediction.


##  How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<Akshaya-Mandepudi>/Explainable-AI-Dashboard.git

cd Explainable-AI-Dashboard

1. Create and activate virtual environment
    python -m venv venv
    venv\Scripts\activate

2. Install dependencies
    pip install -r requirements.txt

3. Run the Streamlit dashboard
    python -m streamlit run app.py


Explainability:

SHAP provides a global view of feature importance and explains overall model behavior.

LIME explains individual predictions, highlighting which features influenced a specific decision.

Together, they make the model transparent and trustworthy.


Use Cases:

1. Credit risk assessment

2. Model transparency and interpretability

3. Explainable AI demonstrations

4. Academic projects and interviews