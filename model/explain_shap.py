import shap
import joblib

model = joblib.load("model/rf_model.pkl")
X_train = joblib.load("model/X_train.pkl")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

joblib.dump(explainer, "model/shap_explainer.pkl")
joblib.dump(shap_values, "model/shap_values.pkl")

print("✅ SHAP explainer created")
