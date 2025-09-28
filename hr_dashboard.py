# HR Analytics - Employee Attrition Project Template with Streamlit Dashboard, Prediction Form, Filters & Time-Series Analysis

# =============================
# 1. Import Libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import shap

import streamlit as st

# =============================
# 2. Load Data
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

df = load_data()
st.title("HR Analytics - Employee Attrition Dashboard")
st.write("Dataset Preview:")
st.dataframe(df.head())

# =============================
# 3. Preprocessing
# =============================
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes=1, No=0

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# =============================
# 4. Modeling
# =============================
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_res, y_train_res)
y_pred_log = log_reg.predict(X_test)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)

xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_clf.predict(X_test)

# =============================
# 5. Dashboard - Model Results
# =============================
st.subheader("Model Performance")
st.write("Logistic Regression:")
st.text(classification_report(y_test, y_pred_log))

st.write("Random Forest:")
st.text(classification_report(y_test, y_pred_rf))

st.write("XGBoost:")
st.text(classification_report(y_test, y_pred_xgb))

# Confusion Matrix - XGBoost
st.subheader("Confusion Matrix - XGBoost")
cm = confusion_matrix(y_test, y_pred_xgb)
st.dataframe(pd.DataFrame(cm, index=["No", "Yes"], columns=["Pred No", "Pred Yes"]))

# ROC Curve - XGBoost
st.subheader("ROC Curve - XGBoost")
xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, xgb_probs)
roc_auc = roc_auc_score(y_test, xgb_probs)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0,1],[0,1],'--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# Feature Importance
st.subheader("Top 15 Feature Importances - Random Forest")
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
feat_importances.nlargest(15).plot(kind='barh', ax=ax)
st.pyplot(fig)

# =============================
# 6. SHAP Analysis
# =============================
st.subheader("SHAP Analysis - XGBoost")
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test)

st.write("SHAP Summary Plot:")
fig_summary = shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
st.pyplot(bbox_inches='tight', dpi=100)

st.write("SHAP Bar Plot:")
fig_bar = shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight', dpi=100)

# =============================
# 7. Employee Prediction Form
# =============================
st.subheader("Employee Attrition Risk Prediction")
st.write("Enter employee details to predict attrition risk:")

user_input = {}
for col in X.columns:
    if df[col].dtype in [np.int64, np.float64]:
        user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    else:
        options = sorted(df[col].unique())
        user_input[col] = st.selectbox(f"{col}", options)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Attrition Risk"):
    prob = xgb_clf.predict_proba(input_scaled)[0,1]
    prediction = xgb_clf.predict(input_scaled)[0]
    st.write(f"**Predicted Attrition:** {'Yes' if prediction == 1 else 'No'}")
    st.write(f"**Attrition Risk Score:** {prob:.2f}")

# =============================
# 8. Filters for Interactive Analysis
# =============================
st.subheader("Interactive Attrition Analysis")

# Department filter
departments = df['Department'].unique()
selected_dept = st.selectbox("Select Department", departments)

dept_data = df[df['Department'] == selected_dept]
st.write(f"Attrition Distribution in {selected_dept}:")
st.bar_chart(dept_data['Attrition'].value_counts())

# Job Role filter
job_roles = df['JobRole'].unique()
selected_role = st.selectbox("Select Job Role", job_roles)

role_data = df[df['JobRole'] == selected_role]
st.write(f"Attrition Distribution in {selected_role}:")
st.bar_chart(role_data['Attrition'].value_counts())

# =============================
# 9. Time-Series Attrition Trends
# =============================
st.subheader("Attrition Trends Over Tenure & Age")

# Attrition by Years at Company
st.write("Attrition by Years at Company:")
tenure_trend = df.groupby('YearsAtCompany')['Attrition'].mean()
fig, ax = plt.subplots()
tenure_trend.plot(ax=ax)
ax.set_xlabel("Years at Company")
ax.set_ylabel("Attrition Rate")
st.pyplot(fig)

# Attrition by Age
st.write("Attrition by Age:")
age_trend = df.groupby('Age')['Attrition'].mean()
fig, ax = plt.subplots()
age_trend.plot(ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("Attrition Rate")
st.pyplot(fig)

# =============================
# 10. Next Steps
# =============================
# - Enhance time-series with smoothing/rolling averages
# - Add cohort analysis by department or job role
# - Deploy app on Streamlit Cloud or internal HR system
