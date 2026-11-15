"""Train a Random Forest on the Heart Disease dataset and save it with joblib."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

df = pd.read_csv("data/heart.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["target"], test_size=0.2, random_state=42, stratify=df["target"])

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
print("test accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump({"model": model, "features": FEATURES}, "model/heart_model.joblib")
