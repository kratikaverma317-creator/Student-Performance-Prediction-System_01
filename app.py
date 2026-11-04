from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from train_model import train

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_performance_model.joblib"
DATA_PATH = ROOT / "data" / "student_performance.csv"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]
data = pd.read_csv(DATA_PATH)

st.title("🎓 Student Performance Prediction System")
st.write("A machine-learning prototype that estimates a student's expected final score from academic and engagement indicators.")
st.info("Demo notice: this project uses generated sample data for demonstration. Predictions are estimates, not official grades or decisions.")

with st.sidebar:
    st.header("Student details")
    study_hours = st.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    previous = st.slider("Previous score", 0, 100, 65)
    assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)
    participation = st.slider("Class participation (1–5)", 1, 5, 3)
    predict_clicked = st.button("Predict performance", type="primary", use_container_width=True)

st.subheader("Prediction")
input_data = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance_percentage": attendance,
    "previous_score": previous,
    "assignments_completed": assignments,
    "class_participation": participation
}])[features]
prediction = float(model.predict(input_data)[0])
prediction = max(0, min(100, prediction))

c1, c2, c3 = st.columns(3)
c1.metric("Estimated final score", f"{prediction:.1f}/100")
c2.metric("Attendance", f"{attendance}%")
c3.metric("Daily study time", f"{study_hours:.1f} hrs")

if prediction >= 75:
    st.success("Estimated performance: High")
elif prediction >= 50:
    st.warning("Estimated performance: Moderate")
else:
    st.error("Estimated performance: Needs improvement")

st.progress(int(prediction))
st.caption("The estimate is based on patterns learned from the demonstration dataset.")

left, right = st.columns(2)
with left:
    st.subheader("Dataset overview")
    st.write(f"Number of sample records: **{len(data)}**")
    st.write(f"Average final score: **{data['final_score'].mean():.1f}**")
    st.write(f"Average attendance: **{data['attendance_percentage'].mean():.1f}%**")
    st.dataframe(data.head(8), use_container_width=True)

with right:
    st.subheader("Feature importance")
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Input feature")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("How the system works")
st.markdown("""
1. Student details are entered through the dashboard.
2. The input values are passed to the trained Random Forest model.
3. The model estimates the expected final score.
4. The dashboard displays the estimate and feature-importance chart.
""")
st.caption("Academic prototype only. Do not use this demo model for real student grading or high-stakes decisions.")
