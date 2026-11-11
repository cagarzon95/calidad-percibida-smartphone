"""Punto de entrada de la aplicación. Cada sección está en su propio archivo."""
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # cálculo de características
sys.path.insert(0, str(HERE))                  # secciones de la aplicación

import evaluar  # noqa: E402
import explorar  # noqa: E402
import grupos  # noqa: E402
import inicio  # noqa: E402
import modelo  # noqa: E402
from datos import MODELS, tables_ready  # noqa: E402

SECTIONS = {
    "Inicio": inicio.show,
    "Explorar las fotos": explorar.show,
    "Qué tan bien predice el modelo": modelo.show,
    "Grupos de fotos": grupos.show,
    "Evaluar tus fotos": evaluar.show,
}
NEEDS_TABLES = {"Explorar las fotos", "Qué tan bien predice el modelo", "Grupos de fotos"}

st.set_page_config(page_title="Calidad percibida de fotos de smartphone", layout="wide")
st.sidebar.title("Calidad percibida")
choice = st.sidebar.radio("Sección", list(SECTIONS))

if choice == "Evaluar tus fotos" and not (MODELS / "modelo_regresion.joblib").exists():
    st.error("No se encuentran los modelos de la aplicación. Revise la carpeta de modelos del proyecto.")
elif choice in NEEDS_TABLES and not tables_ready():
    st.error("Faltan los datos de la aplicación. Ejecute `src/preparar_datos.py` (ver el README).")
else:
    SECTIONS[choice]()
