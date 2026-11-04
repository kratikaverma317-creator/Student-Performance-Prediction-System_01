from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_PATH = ROOT / "models" / "student_performance_model.joblib"

def train():
    data = pd.read_csv(DATA_PATH)
    features = [
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "assignments_completed",
        "class_participation",
    ]
    X = data[features]
    y = data["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150, random_state=42, max_depth=8
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Model evaluation")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.2f}")
    print(f"R2:   {r2_score(y_test, predictions):.3f}")
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
