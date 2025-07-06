import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def cargar_datos(path: str) -> pd.DataFrame:
    """Lee el CSV original."""
    return pd.read_csv(path)


def procesar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y genera las mismas features del ejemplo RF."""

    df = df[(df["housing_median_age"] < 50) &
            (df["median_income"]     < 15) &
            (df["median_house_value"]< 500_000)].copy()

    df["total_bedrooms"] = SimpleImputer(strategy="median") \
                               .fit_transform(df[["total_bedrooms"]])

    df["rooms_per_household"]      = df["total_rooms"]     / df["households"]
    df["bedrooms_per_household"]   = df["total_bedrooms"]  / df["households"]
    df["population_per_household"] = df["population"]      / df["households"]
    df["income_squared"]           = df["median_income"] ** 2

    df = df[df["rooms_per_household"]      < 15]
    df = df[df["population_per_household"] < 10]

    df = pd.concat(
        [df, pd.get_dummies(df["ocean_proximity"], dtype=int)],
        axis=1
    ).drop(columns="ocean_proximity")

    cols = [c for c in df.columns if c != "median_house_value"] + ["median_house_value"]
    return df[cols]
