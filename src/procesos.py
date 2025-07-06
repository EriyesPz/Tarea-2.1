import pandas as pd
from sklearn.impute import SimpleImputer

CAP_AGE     = 50  
CAP_INCOME  = 15 
CAP_VALUE   = 500000


def cargar_datos(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def procesar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df["total_bedrooms"] = SimpleImputer(strategy="median") \
                               .fit_transform(df[["total_bedrooms"]])

    df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

    df["age_capped"]    = (df["housing_median_age"] >= CAP_AGE).astype(int)
    df["income_capped"] = (df["median_income"]     >= CAP_INCOME).astype(int)
    df["value_capped"]  = (df["median_house_value"]>= CAP_VALUE).astype(int)

    df["rooms_per_household"]      = df["total_rooms"] / df["households"]
    df["population_per_household"] = df["population"]  / df["households"]
    df["bedroom_ratio"]            = df["total_rooms"] / df["total_bedrooms"]

    cols = [c for c in df.columns if c != "median_house_value"] + ["median_house_value"]
    return df[cols]
