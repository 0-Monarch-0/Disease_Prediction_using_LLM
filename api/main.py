# main.py — FastAPI app (robust, self-contained)
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE = os.path.dirname(__file__)

# instantiate app
app = FastAPI(title="Disease & Case Prediction API", version="1.0")

# allow local dev CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Model paths
MODEL_DIR = os.path.abspath(os.path.join(BASE, "..", "model-training"))
DISEASE_MODEL_PKL = os.path.join(MODEL_DIR, "disease_model.pkl")
MODEL_COLUMNS_PKL = os.path.join(MODEL_DIR, "model_columns.pkl")
UNIFIED_REG_PKL = os.path.join(MODEL_DIR, "unified_reg_model.pkl")
UNIFIED_REG_COLS_PKL = os.path.join(MODEL_DIR, "unified_reg_columns.pkl")
DISTRICT_STATS_CSV = os.path.join(MODEL_DIR, "disease_district_stats.csv")

# Try load models
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        logging.warning("Could not load %s : %s", path, e)
        return None

disease_model = safe_load(DISEASE_MODEL_PKL)
disease_model_columns = safe_load(MODEL_COLUMNS_PKL)
unified_reg_model = safe_load(UNIFIED_REG_PKL)
unified_reg_columns = safe_load(UNIFIED_REG_COLS_PKL)

# Load district lookup table
def safe_load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

district_stats = safe_load_csv(DISTRICT_STATS_CSV)

def get_top_districts(disease, state, k=3):
    if district_stats is None:
        return []

    sub = district_stats[
        (district_stats["Disease"].str.lower() == str(disease).lower()) &
        (district_stats["state_ut"].str.lower() == str(state).lower())
    ]

    if sub.empty:
        return []

    return (
        sub.sort_values("Cases", ascending=False)
           .head(k)["district"]
           .tolist()
    )


# Pydantic schemas
class DiseaseInput(BaseModel):
    week_of_year: int
    state_ut: str
    temp_celsius: float
    preci: float

class CaseInput(DiseaseInput):
    Disease: str

# Helper to prepare input
def prepare_input(data_dict, model_columns):
    df = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df)

    if model_columns is None:
        return df_encoded

    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_final

# Endpoints
@app.post("/predict_disease")
def predict_disease(input_data: DiseaseInput):
    if disease_model is None or disease_model_columns is None:
        raise HTTPException(status_code=503, detail="Disease model not available.")

    try:
        input_df = prepare_input(input_data.dict(), disease_model_columns)
        pred = disease_model.predict(input_df)[0]

        try:
            proba = disease_model.predict_proba(input_df)
            confidence = float(np.max(proba))
        except:
            confidence = None

        top_risk = get_top_districts(str(pred), input_data.state_ut, k=3)

        return {
            "prediction": str(pred),
            "confidence": confidence,
            "top_districts": top_risk
        }

    except Exception as e:
        logging.exception("Error in predict_disease")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_cases")
def predict_cases(input_data: CaseInput):
    if unified_reg_model is None or unified_reg_columns is None:
        raise HTTPException(status_code=503, detail="Regression model not available.")

    try:
        input_df = prepare_input(input_data.dict(), unified_reg_columns)
        predicted_cases = float(unified_reg_model.predict(input_df)[0])

        return {
            "predicted_cases": round(predicted_cases, 2)
        }
    except Exception as e:
        logging.exception("Error in predict_cases")
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend
FRONTEND_DIR = os.path.abspath(os.path.join(BASE, "..", "frontend", "build"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend index.html not found")

# Health check
@app.get("/health", include_in_schema=False)
def health():
    return {
        "ok": True,
        "disease_model_loaded": disease_model is not None,
        "reg_model_loaded": unified_reg_model is not None
    }

# Route listing
logging.info("Registered routes:")
for r in app.routes:
    try:
        logging.info(
            " - %s %s (include_in_schema=%s)",
            ",".join(r.methods) if hasattr(r, "methods") else "-",
            r.path,
            getattr(r, "include_in_schema", True)
        )
    except:
        pass
