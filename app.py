import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Ensure pip install xgboost
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix)

# Page Layout
st.set_page_config(page_title="ML Assignment UI", layout="wide")
st.title("📊 Breast Cancer Model Evaluation Dashboard")

# --- DATA PREPARATION FUNCTION ---
@st.cache_resource
def train_all_models():
    """Trains all 6 models using the local training file."""
    try:
        train_df = pd.read_csv('breast-cancer_train.csv')
        if 'id' in train_df.columns:
            train_df = train_df.drop(columns=['id'])
            
        X_train = train_df.drop(columns=['diagnosis'])
        y_train_raw = train_df['diagnosis']
        
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=10000).fit(X_train_scaled, y_train)
        # 2. Decision Tree
        dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        # 3. K-Nearest Neighbor
        knn = KNeighborsClassifier().fit(X_train_scaled, y_train)
        # 4. Naive Bayes
        nb = GaussianNB().fit(X_train_scaled, y_train)
        # 5. Random Forest
        rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        # 6. XGBoost
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(X_train, y_train)
        
        trained_models = {
            "Logistic Regression": lr,
            "Decision Tree Classifier": dt,
            "K-Nearest Neighbor Classifier": knn,
            "Naive Bayes Classifier": nb,
            "Ensemble Model - Random Forest": rf,
            "Ensemble Model - XGBoost": xgb
        }
        
        return trained_models, scaler, le
    except Exception as e:
        st.error(f"Error training models: {e}")
        return None, None, None

# Run training
models_dict, main_scaler, main_le = train_all_models()

# --- SIDEBAR ---
st.sidebar.header("Step 1: Upload Data")
uploaded_test_file = st.sidebar.file_uploader("Upload 'breast-cancer_test.csv'", type=["csv"])

if uploaded_test_file is not None and models_dict is not None:
    test_df = pd.read_csv(uploaded_test_file)
    test_display_df = test_df.copy() # Keep copy for top 10 display
    
    # Preprocess Uploaded Test Data for prediction
    if 'id' in test_df.columns:
        test_df = test_df.drop(columns=['id'])
    
    X_test = test_df.drop(columns=['diagnosis'])
    y_test = main_le.transform(test_df['diagnosis'])
    X_test_scaled = main_scaler.transform(X_test)
    
    st.sidebar.header("Step 2: Selection")
    selected_model_name = st.sidebar.selectbox("Choose Model", list(models_dict.keys()))
    
    # Get predictions
    current_model = models_dict[selected_model_name]
    
    if selected_model_name in ["Logistic Regression", "K-Nearest Neighbor Classifier", "Naive Bayes Classifier"]:
        y_pred = current_model.predict(X_test_scaled)
        y_prob = current_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = current_model.predict(X_test)
        y_prob = current_model.predict_proba(X_test)[:, 1]

    # --- PERFORMANCE METRICS TABLE ---
    st.subheader(f"Performance Metrics: {selected_model_name}")
    
    metrics_data = {
        "Metric": [
            "1. Accuracy", "2. AUC Score", "3. Precision", 
            "4. Recall", "5. F1 Score", "6. MCC Score"
        ],
        "Value": [
            accuracy_score(y_test, y_pred),
            roc_auc_score(y_test, y_prob),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            matthews_corrcoef(y_test, y_pred)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    # Displaying metrics in a clean table
    st.table(metrics_df.set_index("Metric"))

    # --- TOP 10 PREDICTIONS SECTION ---
    st.divider()
    st.subheader("Top 10 Test Records: Actual vs Prediction")
    
    # Create comparison dataframe
    top_10_df = test_display_df.head(10).copy()
    top_10_df['Actual Value'] = top_10_df['diagnosis']
    top_10_df['Predicted Value'] = main_le.inverse_transform(y_pred[:10])
    
    # Highlight prediction column for visibility
    st.dataframe(top_10_df)

    # --- CONFUSION MATRIX ---
    st.divider()
    st.subheader("Confusion Matrix")
    col_cm, _ = st.columns([1, 1])
    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=main_le.classes_, yticklabels=main_le.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

else:
    st.info("Awaiting 'breast-cancer_test.csv' upload in the sidebar.")