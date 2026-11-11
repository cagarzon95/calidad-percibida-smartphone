"""Evaluar tus fotos: sube fotos JPG y recibe una puntuación estimada para cada una."""
import numpy as np
import pandas as pd
import streamlit as st

from datos import load_models
from extractor_caracteristicas import jpeg_bytes_to_features, PREDICTORS
from util import label_photo, thumbnail

MAX_FILE_MB = 15
MAX_BATCH_FILES = 20
MAX_BATCH_MB = 50


def estimate_one(name: str, content: bytes, model) -> dict:
    """Estimación de una foto, o el motivo por el que no se pudo procesar."""
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        return {"archivo": name, "error": f"supera {MAX_FILE_MB} MB"}
    if not content.startswith(b"\xff\xd8\xff"):
        return {"archivo": name, "error": "no es un JPEG válido"}
    try:
        features = jpeg_bytes_to_features(content, name)
    except Exception:  # archivo corrupto o formato no soportado
        return {"archivo": name, "error": "no se pudo leer la imagen"}

    vector = pd.DataFrame([[features[key] for key in PREDICTORS]], columns=PREDICTORS)
    if not np.isfinite(vector.to_numpy(dtype=float)).all():
        return {"archivo": name, "error": "imagen sin variación de luz suficiente"}

    score = float(model.predict(vector)[0])
    return {"archivo": name, "error": None, "score": score, "thumb": thumbnail(content),
            "outside": not 0 <= score <= 100}


def check_batch(uploads) -> str:
    """Devuelve un mensaje de error si el lote supera un límite, o cadena vacía si está bien."""
    if len(uploads) > MAX_BATCH_FILES:
        return f"Ha subido {len(uploads)} fotos. El máximo es {MAX_BATCH_FILES}. Quite algunas."
    total_mb = sum(file.size for file in uploads) / (1024 * 1024)
    if total_mb > MAX_BATCH_MB:
        return f"El total pesa {total_mb:.1f} MB y el máximo es {MAX_BATCH_MB} MB. Quite algunas fotos."
    return ""


def show_mosaic(results: list) -> None:
    st.subheader("Mosaico de fotos")
    columns = st.columns(3)
    for index, result in enumerate(results):
        lines = ["Puntuación estimada:", f"{round(result['score'])}"]
        with columns[index % 3]:
            st.image(label_photo(result["thumb"], lines), caption=result["archivo"], width="stretch")
            if result["outside"]:
                st.caption("Fuera de la escala 0–100")


def show_table(results: list) -> None:
    table = pd.DataFrame([{"Foto": r["archivo"], "Puntuación estimada": round(r["score"], 1)} for r in results])
    with st.expander("Ver la tabla de resultados"):
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Descargar la tabla (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="estimaciones_fotos.csv", mime="text/csv")


def show() -> None:
    st.title("Evaluar tus fotos")
    st.markdown(
        f"Suba una o varias fotos en formato JPG (hasta **{MAX_BATCH_FILES}** fotos, **{MAX_FILE_MB} MB** cada "
        f"una y **{MAX_BATCH_MB} MB** en total). La aplicación estima la puntuación de calidad de cada foto. "
        "**Las fotos no se guardan.**")

    uploads = st.file_uploader("Fotos JPG", type=["jpg", "jpeg"], accept_multiple_files=True)
    if not uploads:
        st.info("Todavía no hay fotos cargadas. La aplicación no incluye fotos de ejemplo.")
        return
    problem = check_batch(uploads)
    if problem:
        st.error(problem)
        return

    model = load_models()["score"]
    with st.spinner("Estimando las fotos…"):
        results = [estimate_one(file.name, file.getvalue(), model) for file in uploads]

    for result in results:
        if result["error"]:
            st.error(f"{result['archivo']}: no se procesó ({result['error']}).")
    ok = [result for result in results if not result["error"]]
    if not ok:
        return
    show_mosaic(ok)
    show_table(ok)
