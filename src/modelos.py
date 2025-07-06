from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def entrenar_modelo(df: pd.DataFrame) -> Tuple[Pipeline, float, float, float]:
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = ColumnTransformer(
        transformers=[("num", RobustScaler(), X.columns)],
        remainder="drop",
    )

    modelo = Pipeline(
        steps=[
            ("scale", scaler),
            ("rf", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    modelo.fit(X_tr, y_tr)

    y_pred   = modelo.predict(X_te)
    r2_train = modelo.score(X_tr, y_tr)
    r2_test  = modelo.score(X_te, y_te)
    rmse     = np.sqrt(mean_squared_error(y_te, y_pred))

    print(f"[DEBUG] R² train: {r2_train:.4f} — R² test: {r2_test:.4f} — RMSE: {rmse:,.0f}")
    return modelo, r2_train, r2_test, rmse