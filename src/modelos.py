from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

def entrenar_modelo(df: pd.DataFrame) -> Tuple[Pipeline, float, float, float]:
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 2, 20), cv=5))
    ])

    modelo.fit(X_tr, y_tr)

    y_pred   = modelo.predict(X_te)
    r2_train = modelo.score(X_tr, y_tr)
    r2_test  = modelo.score(X_te, y_te)
    rmse     = np.sqrt(mean_squared_error(y_te, y_pred))

    return modelo, r2_train, r2_test, rmse
