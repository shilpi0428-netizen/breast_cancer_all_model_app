import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/breast-cancer.csv")

    # Drop ID column if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Use diagnosis column as target
    y = df["diagnosis"]

    # Convert M/B to 1/0
    y = y.map({"M": 1, "B": 0})

    # Drop target column from features
    X = df.drop(columns=["diagnosis"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test