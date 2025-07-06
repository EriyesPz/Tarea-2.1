import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

CAP_AGE, CAP_INC, CAP_VAL = 50, 15, 500_000

def cargar_datos(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def procesar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df["total_bedrooms"] = (
        SimpleImputer(strategy="median")
        .fit_transform(df[["total_bedrooms"]])
    )

    df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

    df["age_capped"]    = (df["housing_median_age"] >= CAP_AGE).astype(int)
    df["income_capped"] = (df["median_income"] >= CAP_INC).astype(int)
    df["value_capped"]  = (df["median_house_value"] >= CAP_VAL).astype(int)

    df["rooms_per_household"]      = df["total_rooms"] / df["households"]
    df["population_per_household"] = df["population"] / df["households"]
    df["bedroom_ratio"]            = df["total_rooms"] / df["total_bedrooms"]
    df["income_sq"]                = df["median_income"] ** 2
    df["log_income"]               = np.log1p(df["median_income"])
    df["median_income_cubed"] = df["median_income"] ** 3
    df["rooms_density"] = df["total_rooms"] / (df["population"] + 1)
    df["income_times_rooms"] = df["median_income"] * df["total_rooms"]


    cols = [c for c in df.columns if c != "median_house_value"] + ["median_house_value"]
    return df[cols]
