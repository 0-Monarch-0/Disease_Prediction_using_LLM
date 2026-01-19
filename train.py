#!/usr/bin/env python3
"""
model_training/train.py

- Cleans the raw CSV (Final_data.csv) and writes cleaned_disease_data.csv
- Trains a RandomForestClassifier to predict Disease and saves:
    - disease_model.pkl
    - model_columns.pkl
- Trains a RandomForestRegressor to predict Cases and saves:
    - unified_reg_model.pkl
    - unified_reg_columns.pkl

Run:
    python3 model_training/train.py
"""

import os
import sys
import argparse
import logging
from typing import Optional

from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import joblib

# -------------------------
# Config / Paths (relative)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "..", "Final_data.csv")
CLEANED_CSV = os.path.join(BASE_DIR, "cleaned_disease_data.csv")

DISEASE_MODEL_PKL = os.path.join(BASE_DIR, "disease_model.pkl")
MODEL_COLUMNS_PKL = os.path.join(BASE_DIR, "model_columns.pkl")

UNIFIED_REG_PKL = os.path.join(BASE_DIR, "unified_reg_model.pkl")
UNIFIED_REG_COLS_PKL = os.path.join(BASE_DIR, "unified_reg_columns.pkl")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

RANDOM_STATE = 42

def clean_and_save():
    import pandas as pd
    import numpy as np
    import logging, os, sys

    logging.info("Loading raw CSV from: %s", DATA_CSV)
    try:
        df = pd.read_csv(DATA_CSV)
    except FileNotFoundError:
        logging.error("File not found: %s", DATA_CSV)
        sys.exit(1)

    # --- normalize raw headers ---
    rename_map = {
        "week": "week_of_year",
        "week_no": "week_of_year",
        "week_of_outbreak": "week_of_year",
        "Temp": "temp_celsius",
        "temperature": "temp_celsius",
        "precipitation": "preci",
        "precip": "preci",
        "district_name": "district",
        "state": "state_ut",
        "state_name": "state_ut",
        "disease_name": "Disease",
        "disease_type": "Disease",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    # standardize cases name
    for col in df.columns:
        if "case" in col.lower():
            df.rename(columns={col: "Cases"}, inplace=True)
            break

    # --- week extraction ---
    if "week_of_year" in df.columns:
        df["week_of_year"] = (
            df["week_of_year"]
            .astype(str)
            .str.extract(r"(\d{1,2})", expand=False)
        )
        df["week_of_year"] = pd.to_numeric(df["week_of_year"], errors="coerce").astype("Int64")

    # temperature numeric + Kelvin heuristic
    if "temp_celsius" in df.columns:
        df["temp_celsius"] = pd.to_numeric(df["temp_celsius"], errors="coerce")
        if df["temp_celsius"].dropna().gt(100).any():
            df["temp_celsius"] = df["temp_celsius"] - 273.15

    if "preci" in df.columns:
        df["preci"] = pd.to_numeric(df["preci"], errors="coerce")

    # remove COVID period
    if "year" in df.columns:
        df = df[~df["year"].isin([2020, 2021, 2022])]

    # --- disease merge rules (PERMANENT) ---
    merge_map = {
        "Dengue/Chikungunya": "Dengue",
        "Dengue And Chikungunya": "Dengue",
        "Suspected Dengue": "Dengue",
        "Dengue And Malaria": "Dengue",
        "Suspected Dengue And Chikungunya": "Dengue",

        "Chikungunya/Dengue": "Chikungunya",
        "Chikungunya/ Dengue": "Chikungunya",
        "Suspected Chikungunya": "Chikungunya",

        "Gastroenteritis": "Acute Gastroenteritis",
        "Suspected Cholera": "Cholera",
        "Diarrhea": "Acute Diarrhoeal Disease",
    }
    if "Disease" in df.columns:
        df["Disease"] = df["Disease"].replace(merge_map)

    # --- keep only modeling cols ---
    cols = ["Disease", "week_of_year", "district", "state_ut", "temp_celsius", "preci", "Cases"]
    keep = [c for c in cols if c in df.columns]
    df = df[keep]

    # drop rows missing core features
    core = ["Disease", "week_of_year", "state_ut", "temp_celsius", "preci"]
    core = [c for c in core if c in df.columns]
    df = df.dropna(subset=core)

    # remove disease labels <2 samples
    vc = df["Disease"].value_counts()
    rare = vc[vc < 2].index
    df = df[~df["Disease"].isin(rare)]

    # store
    df.to_csv(CLEANED_CSV, index=False)
    logging.info("Cleaned data saved → %s", CLEANED_CSV)
    return df


def build_district_lookup():
    import pandas as pd
    import numpy as np
    import logging, sys, os

    logging.info("Building disease→district lookup table…")

    # 1) LOAD RAW CSV
    try:
        df = pd.read_csv(DATA_CSV)
    except FileNotFoundError:
        logging.error("File not found: %s", DATA_CSV)
        sys.exit(1)

    print("Lookup DF columns BEFORE rename:", df.columns.tolist())

    # 2) === SAME CLEANING AS clean_and_save() ===

    # normalize headers
    rename_map = {
        "week": "week_of_year",
        "week_no": "week_of_year",
        "week_of_outbreak": "week_of_year",

        "Temp": "temp_celsius",
        "temperature": "temp_celsius",

        "precipitation": "preci",
        "precip": "preci",

        "district_name": "district",
        "District": "district",

        "state": "state_ut",
        "state_name": "state_ut",
        "State/UT": "state_ut",
        "STNAME": "state_ut",
        "ST_NM": "state_ut",

        "disease_name": "Disease",
        "disease_type": "Disease",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    # Standardize Cases column
    for col in df.columns:
        if "case" in col.lower():
            df.rename(columns={col: "Cases"}, inplace=True)
            break

    # extract numeric week
    if "week_of_year" in df.columns:
        df["week_of_year"] = (
            df["week_of_year"]
            .astype(str)
            .str.extract(r"(\d{1,2})", expand=False)
        )
        df["week_of_year"] = pd.to_numeric(df["week_of_year"], errors="coerce").astype("Int64")

    # numeric weather
    if "temp_celsius" in df.columns:
        df["temp_celsius"] = pd.to_numeric(df["temp_celsius"], errors="coerce")
        if df["temp_celsius"].dropna().gt(100).any():
            df["temp_celsius"] = df["temp_celsius"] - 273.15

    if "preci" in df.columns:
        df["preci"] = pd.to_numeric(df["preci"], errors="coerce")

    # remove covid years
    if "year" in df.columns:
        df = df[~df["year"].isin([2020, 2021, 2022])]

    # 3) --- DISEASE MERGING ---
    merge_map = {
        "Dengue/Chikungunya": "Dengue",
        "Dengue And Chikungunya": "Dengue",
        "Suspected Dengue": "Dengue",
        "Dengue And Malaria": "Dengue",
        "Suspected Dengue And Chikungunya": "Dengue",

        "Chikungunya/Dengue": "Chikungunya",
        "Chikungunya/ Dengue": "Chikungunya",
        "Suspected Chikungunya": "Chikungunya",

        "Gastroenteritis": "Acute Gastroenteritis",
        "Suspected Cholera": "Cholera",
        "Diarrhea": "Acute Diarrhoeal Disease",
    }
    if "Disease" in df.columns:
        df["Disease"] = df["Disease"].replace(merge_map)

    # 4) --- KEEP NECESSARY COLS (IMPORTANT: include state_ut) ---
    keep = ["Disease", "state_ut", "district", "Cases"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    print("Lookup DF columns AFTER filtering:", df.columns.tolist())

    # drop missing
    df = df.dropna(subset=["Disease", "district", "Cases"])
    df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")
    df = df.dropna(subset=["Cases"])

    # remove rare disease labels (<2 samples)
    vc = df["Disease"].value_counts()
    rare = vc[vc < 2].index
    df = df[~df["Disease"].isin(rare)]

    # 5) --- BUILD LOOKUP TABLE (state-aware) ---
    if "state_ut" not in df.columns:
        raise KeyError("❌ 'state_ut' missing after cleaning → lookup cannot be built")

    stats = (
        df.groupby(["Disease", "state_ut", "district"], as_index=False)["Cases"]
        .mean()
        .sort_values(by="Cases", ascending=False)
    )

    OUT = os.path.join(BASE_DIR, "disease_district_stats.csv")
    stats.to_csv(OUT, index=False)
    logging.info("✅ District lookup saved → %s", OUT)

    return stats



def train_classification(save_models: bool = True) -> Optional[pd.Series]:
    """
    Train RandomForestClassifier on cleaned_disease_data.csv to predict Disease.

    Returns:
        The list/Index of model columns (one-hot encoded feature columns) saved as MODEL_COLUMNS_PKL.
    """
    logging.info("TRAINING: classification - loading cleaned CSV from: %s", CLEANED_CSV)
    try:
        df = pd.read_csv(CLEANED_CSV)
    except FileNotFoundError:
        logging.error("Cleaned CSV not found. Run clean_and_save() first.")
        return None

    print("\n=== DISEASE DISTRIBUTION ===")
    print(df["Disease"].value_counts().sort_values())
    print("============================\n")

    # Features and target
    features = ['week_of_year','state_ut', 'temp_celsius', 'preci']
    for f in features:
        if f not in df.columns:
            logging.error("Required feature '%s' not in cleaned CSV", f)
            return None

    X = df[features]
    y = df['Disease']

    # One-hot encode categorical columns (district, state_ut)
    X_encoded = pd.get_dummies(X, columns=['state_ut'], drop_first=True)



    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=RANDOM_STATE,stratify=y)

    from collections import Counter

    print("Before SMOTE:", Counter(y_train))

    vc = y_train.value_counts()
    to_drop = vc[vc < 2].index
    mask = ~y_train.isin(to_drop)
    X_train = X_train[mask]
    y_train = y_train[mask]

    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=1)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print("After SMOTE:", Counter(y_train))

    logging.info("Classification: training samples: %d, test samples: %d", X_train.shape[0], X_test.shape[0])

    # Model
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,class_weight="balanced")
    logging.info("Fitting RandomForestClassifier...")
    clf.fit(X_train, y_train)
    logging.info("Classifier training complete.")

    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info("Classification accuracy: %.4f", acc)
    logging.info("Classification report:\n%s", classification_report(y_test, preds, zero_division=0))

    # Save model and columns
    if save_models:
        joblib.dump(clf, DISEASE_MODEL_PKL)
        joblib.dump(list(X_encoded.columns), MODEL_COLUMNS_PKL)
        logging.info("Saved classifier to %s and columns to %s", DISEASE_MODEL_PKL, MODEL_COLUMNS_PKL)

    return X_encoded.columns


def train_regression(save_models: bool = True) -> Optional[pd.Series]:
    """
    Train RandomForestRegressor on cleaned_disease_data.csv to predict Cases.
    This requires the 'Cases' column to be present. We also include 'Disease' as a feature (one-hot).
    """
    logging.info("TRAINING: regression - loading cleaned CSV from: %s", CLEANED_CSV)
    try:
        df_full = pd.read_csv(CLEANED_CSV)
    except FileNotFoundError:
        logging.error("Cleaned CSV not found. Run clean_and_save() first.")
        return None

    # Ensure Cases column exists and is numeric
    if 'Cases' not in df_full.columns:
        logging.error("'Cases' column not found in cleaned CSV. Regression cannot proceed.")
        return None

    # drop NA and coerce numeric
    df_full['Cases'] = pd.to_numeric(df_full['Cases'], errors='coerce')
    df_full = df_full.dropna(subset=['Cases'])
    df_full['Cases'] = df_full['Cases'].astype(int)

    # Features: include Disease as predictor
    features = ['week_of_year', 'district', 'state_ut', 'temp_celsius', 'preci', 'Disease']
    for f in features:
        if f not in df_full.columns:
            logging.error("Required feature '%s' not in cleaned CSV for regression", f)
            return None

    X = df_full[features]
    y = df_full['Cases']

    # One-hot encode categorical columns including Disease
    X_encoded = pd.get_dummies(X, columns=['district', 'state_ut', 'Disease'], drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=RANDOM_STATE)
    logging.info("Regression: training samples: %d, test samples: %d", X_train.shape[0], X_test.shape[0])

    # Model
    regr = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    logging.info("Fitting RandomForestRegressor...")
    regr.fit(X_train, y_train)
    logging.info("Regressor training complete.")

    # Evaluate
    preds = regr.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    logging.info("Regression MAE: %.4f", mae)
    logging.info("Regression R^2: %.4f", r2)

    # Save model and columns
    if save_models:
        joblib.dump(regr, UNIFIED_REG_PKL)
        joblib.dump(list(X_encoded.columns), UNIFIED_REG_COLS_PKL)
        logging.info("Saved regressor to %s and columns to %s", UNIFIED_REG_PKL, UNIFIED_REG_COLS_PKL)

    return X_encoded.columns


def main(run_clean: bool = True, run_clf: bool = True, run_reg: bool = True):
    if run_clean:
        clean_and_save()
    build_district_lookup()
    if run_clf:
        clf_cols = train_classification(save_models=True)
    if run_reg:
        reg_cols = train_regression(save_models=True)

    logging.info("All requested steps finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train disease classification and regression models.")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning step (assumes cleaned CSV already exists)")
    parser.add_argument("--no-clf", action="store_true", help="Skip classification training")
    parser.add_argument("--no-reg", action="store_true", help="Skip regression training")
    args = parser.parse_args()

    main(run_clean=not args.no_clean, run_clf=not args.no_clf, run_reg=not args.no_reg)
