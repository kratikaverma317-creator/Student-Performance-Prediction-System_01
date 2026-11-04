# Student Performance Prediction System

A beginner-friendly machine-learning web application that estimates a student's final score from study hours, attendance, previous score, assignments completed, and class participation.

## Technology
- Python
- Pandas and NumPy
- Scikit-learn (RandomForestRegressor)
- Streamlit
- Matplotlib and Joblib

## Important dataset note
`data/student_performance.csv` is a **generated demonstration dataset**, not real student records. The model is a learning prototype and should not be used for actual grading or high-stakes decisions.

## Run on Windows (VS Code)
Open this folder in VS Code, then open **Terminal > New Terminal**.

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py train_model.py
py -m streamlit run app.py
```

If `py` is not recognized, try replacing `py` with `python`.

Streamlit will show a local URL (usually http://localhost:8501). Open it in your browser.

## Project structure
```text
Student_Performance_Prediction_System/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── presentation_notes.md
├── data/
│   └── student_performance.csv
└── models/
    └── (trained model is created here)
```

## Workflow
1. Load the CSV dataset using Pandas.
2. Separate input features and target (`final_score`).
3. Split data into training and testing sets (80:20).
4. Train a Random Forest regression model.
5. Evaluate using MAE, RMSE, and R-squared.
6. Save the trained model using Joblib.
7. Use Streamlit to accept inputs and display a prediction.

## Future scope
- Replace generated data with a properly licensed, verified real dataset.
- Validate model performance on independent real-world data.
- Add teacher/admin login and student progress history.
- Add privacy, fairness, and model-monitoring safeguards.
