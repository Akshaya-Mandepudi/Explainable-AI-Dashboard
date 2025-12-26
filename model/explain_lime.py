from lime.lime_tabular import LimeTabularExplainer
import joblib

# Load trained model and training data
model = joblib.load("model/rf_model.pkl")
X_train = joblib.load("model/X_train.pkl")

# Create LIME explainer (DO NOT SAVE IT)
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["No Default", "Default"],
    mode="classification"
)

# Explain one sample instance
sample = X_train.iloc[0]

explanation = lime_explainer.explain_instance(
    sample.values,
    model.predict_proba,
    num_features=5
)

print("✅ LIME explanation generated successfully!")
print(explanation.as_list())
