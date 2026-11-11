"""Carga de las tablas y de los modelos. Todo se guarda en memoria después de la primera vez."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from extractor_caracteristicas import SCENES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"


@st.cache_data(show_spinner="Cargando datos…")
def load_photos() -> pd.DataFrame:
    """Una fila por foto: puntuación, escenas, datos de la cámara y características."""
    photos = pd.read_csv(DATA / "fotos.csv", dtype={"image_id": str})
    photos["scene_count"] = (photos[SCENES] > 0).sum(axis=1)
    photos["fecha"] = pd.to_datetime(photos["exif_fecha_captura"], errors="coerce")
    return photos


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA / "predicciones_validacion.csv", dtype={"image_id": str})


@st.cache_data
def load_groups() -> pd.DataFrame:
    return pd.read_csv(DATA / "grupos_entrenamiento.csv", dtype={"image_id": str})


@st.cache_data
def load_model_summary() -> pd.DataFrame:
    return pd.read_csv(DATA / "resumen_modelos.csv")


@st.cache_data
def load_scene_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA / "metricas_escenas.csv")


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_csv(DATA / "importancia_caracteristicas.csv")


@st.cache_data
def load_training_curve() -> pd.DataFrame:
    return pd.read_csv(DATA / "historial_red_neuronal.csv")


@st.cache_data
def load_group_choice() -> pd.DataFrame:
    return pd.read_csv(DATA / "seleccion_grupos.csv")


@st.cache_resource(show_spinner="Cargando el modelo…")
def load_models() -> dict:
    """Modelos que se usan para evaluar fotos nuevas."""
    files = {
        "score": "modelo_regresion.joblib",
        "scenes": "modelo_escenas.joblib",
        "group_scaler": "transformador_grupos.joblib",
        "group": "grupos_kmeans.joblib",
        "map": "mapa_pca.joblib",
    }
    return {key: joblib.load(MODELS / name) for key, name in files.items()}


def tables_ready() -> bool:
    return (DATA / "fotos.csv").exists()
