import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from model.common import load_data
from model.logistic_model import run_logistic
from model.decision_tree_model import run_decision_tree
from model.knn_model import run_knn
from model.naive_bayes_model import run_naive_bayes
from model.random_forest_model import run_random_forest
from model.xgboost_model import run_xgboost


st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.title("🧬 Breast Cancer Classification App")
st.write("Machine Learning Assignment - 2")


# ===============================
# Sidebar
# ===============================
st.sidebar.header("Upload Test Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)


# ===============================
# Run Selected Model
# ===============================

if model_choice == "Logistic Regression":
    result = run_logistic()

elif model_choice == "Decision Tree":
    result = run_decision_tree()

elif model_choice == "KNN":
    result = run_knn()

elif model_choice == "Naive Bayes":
    result = run_naive_bayes()

elif model_choice == "Random Forest":
    result = run_random_forest()

elif model_choice == "XGBoost":
    result = run_xgboost()


st.subheader("📊 Model Evaluation Metrics")

metrics_df = pd.DataFrame(result.items(), columns=["Metric", "Value"])
st.dataframe(metrics_df)


# ===============================
# Confusion Matrix (Using internal test split)
# ===============================

st.subheader("📉 Confusion Matrix")

X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_data()

# Re-train model to get predictions for confusion matrix
if model_choice == "Logistic Regression":
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_choice == "Decision Tree":
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif model_choice == "KNN":
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_choice == "Naive Bayes":
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_choice == "Random Forest":
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif model_choice == "XGBoost":
    from xgboost import XGBClassifier
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)


# ===============================
# Classification Report
# ===============================

st.subheader("📝 Classification Report")
st.text(classification_report(y_test, y_pred))


# ===============================
# CSV Upload Prediction (Optional Bonus)
# ===============================

if uploaded_file is not None:
    st.subheader("📂 Predictions on Uploaded File")

    df = pd.read_csv(uploaded_file)

    if 'diagnosis' in df.columns:
        df = df.drop(columns=['diagnosis'])

    scaler = None

    if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = df

    predictions = model.predict(df_scaled)

    df["Prediction"] = predictions
    st.dataframe(df.head())