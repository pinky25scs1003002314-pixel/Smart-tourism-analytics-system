import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["year", "month", "state", "destination"]
TARGET = "total_visitors"

def train_tourism_model(data: pd.DataFrame):
    """Train a baseline Random Forest model and return model + evaluation metrics."""
    usable = data.dropna(subset=FEATURES + [TARGET]).copy()
    if len(usable) < 8:
        raise ValueError("At least 8 valid dataset rows are required to train the model.")

    X = usable[FEATURES]
    y = usable[TARGET]

    categorical = ["state", "destination"]
    numeric = ["year", "month"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150, random_state=42, min_samples_leaf=1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else float("nan"),
        "test_rows": int(len(y_test)),
    }
    return model, metrics

def predict_visitors(model, year: int, month: int, state: str, destination: str) -> int:
    sample = pd.DataFrame([{
        "year": int(year),
        "month": int(month),
        "state": state,
        "destination": destination
    }])
    prediction = model.predict(sample)[0]
    return max(0, int(round(prediction)))
