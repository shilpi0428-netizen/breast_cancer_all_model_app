# 🧬 Breast Cancer Classification – Machine Learning Assignment 2

## 👩‍🎓 Student Information
BITS ID: 2025ab05104
Name: Shilpi
Email: 2025ab05104@wilp.bits-pilani.ac.in
Date: 8, Feb, 2026

---

# 📌 Problem Statement

The objective of this project is to build multiple machine learning classification models to predict whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** based on various medical diagnostic features.

The goal is to compare the performance of different classification algorithms using standard evaluation metrics and deploy an interactive Streamlit web application for real-time predictions and visualization.

---

# 📊 Dataset Description

The Breast Cancer dataset contains diagnostic measurements computed from digitized images of breast mass samples.  

- Total Instances: 569  
- Total Features: 30 numerical features  
- Target Variable: `diagnosis`  
    - M → Malignant (1)  
    - B → Benign (0)  

### 🔍 Features Include:
- Radius (mean, worst, se)
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

### 🎯 Preprocessing Performed:
- Removed ID column
- Converted diagnosis (M/B) into binary (1/0)
- Train-test split (80% training, 20% testing)
- Feature scaling using StandardScaler (for Logistic Regression, KNN, Naive Bayes)

---

# 🤖 Models Implemented

The following six machine learning models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

---

# 📈 Evaluation Metrics Used

For each model, the following metrics were calculated:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

# 🖥️ Streamlit Web Application Features

The deployed Streamlit app includes:

- CSV dataset upload option
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization
- Classification report
- Prediction on uploaded dataset

---

# 🚀 Deployment

The application has been deployed using:

Streamlit Community Cloud  

Live App Link: [*(Streamlit link)*  ](https://shilpibits.streamlit.app/)

GitHub Repository Link: [*(GitHub repo link)*  ](https://github.com/shilpi0428-netizen/breast_cancer_all_model_app)

---

# 📂 Project Structure
breast-cancer-ml-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│ └── breast-cancer.csv
│
└── model/
├── common.py
├── logistic_model.py
├── decision_tree_model.py
├── knn_model.py
├── naive_bayes_model.py
├── random_forest_model.py
└── xgboost_model.py


# 📊 Model Comparison Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|---------------|----------|-----------|-----------|--------|----------|-----------|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9285 | 0.9512 | 0.9245 |
| Decision Tree Classifier | 0.9298 | 0.9246 | 0.9047 | 0.9047 | 0.9047 | 0.8492 |
| K-Nearest Neighbors (KNN) | 0.9561 | 0.9823 | 0.9743 | 0.9047 | 0.9382 | 0.9058 |
| Naive Bayes (GaussianNB) | 0.9210 | 0.9890 | 0.9230 | 0.8571 | 0.8888 | 0.8291 |
| Random Forest (Ensemble) | 0.9736 | 0.9928 | 1.0000 | 0.9285 | 0.9629 | 0.9441 |
| XGBoost (Ensemble) | 0.9736 | 0.9940 | 1.0000 | 0.9285 | 0.9629 | 0.9441 |

---

# 📌 Observations on Model Performance

| Model Name | Key Observations |
|------------|------------------|
| **Logistic Regression** | Demonstrates the **highest AUC Score (0.9960)** among all classifiers, indicating exceptional ability to distinguish between classes. Provides very high Accuracy (0.9649) and balanced Recall (0.9285), making it one of the most reliable base learners for this dataset. |
| **Decision Tree Classifier** | Shows **lower performance** compared to ensemble methods. While it achieves decent Accuracy (0.9298), its AUC (0.9246) and MCC (0.8492) are significantly lower than others, suggesting it may not be as robust at generalizing as more complex models. |
| **K-Nearest Neighbors (KNN)** | Shows **strong performance** with Accuracy of 0.9561 and AUC of 0.9823. Has high Precision (0.9743), but Recall (0.9047) is slightly lower than top-performing models, meaning it might miss a small percentage of positive cases. |
| **Naive Bayes (GaussianNB)** | Yielded the **lowest overall performance** across almost all metrics, including lowest Accuracy (0.9210) and Recall (0.8571). The drop in MCC (0.8291) suggests that the assumption of feature independence may not hold perfectly for this clinical dataset. |
| **Random Forest (Ensemble)** | One of the **two top-performing models**. Achieved highest Accuracy (0.9736) and **perfect Precision (1.0)**, indicating zero False Positives. With high MCC (0.9441) and Recall (0.9285), it is highly effective at minimizing misclassifications. |
| **XGBoost (Ensemble)** | **Tied with Random Forest** for best performance with Accuracy of 0.9736 and Precision of 1.0. Has slightly higher AUC (0.9940) than Random Forest, suggesting a very refined decision boundary. **Ideal model** for this classification task due to high reliability and balanced metrics. |

---

## 🏆 Best Performing Models

**Top 2 Models:**
1. **XGBoost** - Best overall with highest AUC (0.9940) and perfect precision
2. **Random Forest** - Tied for accuracy with XGBoost, perfect precision (1.0)

**Key Insights:**
- Ensemble methods (Random Forest & XGBoost) significantly outperform individual classifiers
- Logistic Regression shows excellent AUC despite being a simpler model
- Naive Bayes has the weakest performance, likely due to feature correlation
- All models achieve >92% accuracy, indicating the dataset is well-suited for classification