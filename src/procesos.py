import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

CAP_AGE, CAP_INC = 50, 15

def cargar_datos(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def procesar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df["total_bedrooms"] = (
        SimpleImputer(strategy="median")
        .fit_transform(df[["total_bedrooms"]])
    )

    df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

    df["age_capped"]    = (df["housing_median_age"] >= CAP_AGE).astype(int)
    df["income_capped"] = (df["median_income"]     >= CAP_INC).astype(int)

    df["rooms_per_household"]      = df["total_rooms"] / df["households"]
    df["population_per_household"] = df["population"]  / df["households"]
    df["bedroom_ratio"]            = df["total_rooms"] / df["total_bedrooms"]
    
    df["bedrooms_per_household"]   = df["total_bedrooms"] / df["households"]
    df["income_per_capita"]        = df["median_income"] / df["population_per_household"]
    df["density"]                  = df["population"] / (df["total_rooms"] + 1)  # +1 para evitar división por 0
    
    df["income_age_interaction"]   = df["median_income"] * df["housing_median_age"]
    df["location_income"]          = df["longitude"] * df["median_income"]
    df["location_age"]             = df["latitude"] * df["housing_median_age"]
    
    df["log_median_income"]        = np.log1p(df["median_income"])
    df["log_population"]           = np.log1p(df["population"])
    df["log_households"]           = np.log1p(df["households"])
    
    df["median_income_sq"]         = df["median_income"] ** 2
    df["housing_median_age_sq"]    = df["housing_median_age"] ** 2
    
    cols = [c for c in df.columns if c != "median_house_value"] + ["median_house_value"]
    return df[cols]