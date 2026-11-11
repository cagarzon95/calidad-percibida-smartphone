"""Página de inicio: qué es, de dónde vienen los datos y una exploración rápida."""
import numpy as np
import plotly.express as px
import streamlit as st

from datos import ROOT, load_photos, load_predictions
from extractor_caracteristicas import SCENES
from textos import DEFAULT_SCORE_RANGE, SCENE_INFO, SCENE_LABEL
from util import fmt

EXAMPLES = ROOT / "dashboard" / "assets" / "escenas"


def example_path(scene: str):
    return EXAMPLES / (scene.lower().replace(" ", "_").replace("-", "_") + ".jpg")


def show_header() -> None:
    st.title("Calidad percibida de fotos de smartphone")
    st.markdown("A partir del conjunto de datos *Smartphone Photography Attribute and Quality* (SPAQ).")

    photos = load_photos()
    predictions = load_predictions()
    error = predictions.MOS - predictions.MOS_predicted
    explained = 1 - (error ** 2).sum() / ((predictions.MOS - predictions.MOS.mean()) ** 2).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fotografías", f"{len(photos):,}".replace(",", "."))
    col2.metric("Teléfonos distintos", "66")
    col3.metric("Puntuación media de calidad", fmt(photos.MOS.mean(), 1), help="Escala de 0 a 100")
    col4.metric("Error medio del modelo", f"{fmt(error.abs().mean(), 1)} puntos",
                help=f"Validación. Proporción explicada: {fmt(explained, 2)}")

    st.markdown(
        "**Pregunta de investigación.** ¿En qué medida las características de la imagen de una fotografía "
        "permiten predecir su calidad percibida, y cómo varía esa capacidad según el nivel de calidad, "
        "la escena y las propiedades visuales?")


def show_sources() -> None:
    with st.container(border=True):
        st.markdown("#### Fuente y créditos")
        st.markdown(
            "- **Conjunto de datos:** SPAQ, Fang, Zhu, Zeng, Ma y Wang (2020), *Perceptual Quality Assessment of "
            "Smartphone Photography*, IEEE/CVF CVPR 2020. DOI [10.1109/CVPR42600.2020.00373]"
            "(https://doi.org/10.1109/CVPR42600.2020.00373).\n"
            "- **Repositorio original:** [github.com/h4nwei/SPAQ](https://github.com/h4nwei/SPAQ). "
            "Licencia: MIT (ver [issue #29](https://github.com/h4nwei/SPAQ/issues/29)).\n"
            "- **Trabajo:** Carlos Mario Garzón, Ciencia de Datos, Maestría en Inteligencia Artificial, "
            "Universidad de La Salle.")


def show_dataset() -> None:
    st.subheader("El conjunto de datos")
    st.markdown(
        "**11.125 fotografías** tomadas con **66 teléfonos** distintos. Cada foto tiene una **puntuación de "
        "calidad** de 0 a 100, que es el promedio de las opiniones de personas en un laboratorio controlado. "
        "Además, cada foto tiene **14 características de la imagen** calculadas a partir de sus píxeles, "
        "**etiquetas de escena** y **datos técnicos de la cámara**.")


def show_guide() -> None:
    st.subheader("Qué puede hacer en esta aplicación")
    st.markdown(
        "- **Explorar las fotos:** filtre por puntuación, características, escenas o fecha de captura.\n"
        "- **Qué tan bien predice el modelo:** vea cuánto se equivoca el modelo y en qué casos.\n"
        "- **Grupos de fotos:** vea cómo se agrupan las fotos según sus características, sin mirar la puntuación.\n"
        "- **Evaluar tus fotos:** suba fotos JPG y reciba una puntuación estimada.")


def show_dictionary() -> None:
    with st.expander("Diccionario"):
        st.markdown(
            "- **Puntuación de calidad:** promedio de las opiniones de personas sobre una foto, de 0 a 100.\n"
            "- **Característica de la imagen:** número calculado a partir de los píxeles de la foto, como el brillo o la nitidez.\n"
            "- **Modelo predictivo (bosque aleatorio, Random Forest):** combina muchos árboles de decisión para estimar la puntuación.\n"
            "- **Error medio:** en promedio, cuántos puntos se equivoca el modelo.\n"
            "- **Proporción explicada:** qué parte de las diferencias entre fotos explica el modelo. 1 es perfecto; 0 significa que no explica nada.\n"
            "- **Agrupación:** separar fotos en grupos con características parecidas.\n"
            "- **Datos de la cámara:** información que la cámara guarda en cada foto, como ISO, exposición, apertura y fecha.\n"
            "- **Grupo de entrenamiento, validación y prueba:** partes en que se dividieron las fotos para entrenar y evaluar el modelo.")


def show_scenes() -> None:
    st.subheader("Explore las escenas")
    st.markdown("Elija una escena para ver su significado, una foto de ejemplo y cuántas fotos la tienen.")
    photos = load_photos()
    scene = st.radio("Escena", SCENES, format_func=lambda key: SCENE_LABEL[key], horizontal=True,
                     key="scene_pick")
    label, description = SCENE_INFO[scene]
    with_scene = photos[photos[scene] > 0]

    left, right = st.columns(2)
    if example_path(scene).exists():
        left.image(str(example_path(scene)), width="stretch")
        left.caption("Foto: conjunto SPAQ (Fang et al., 2020, CVPR). Reducida. "
                     "Licencia MIT del repositorio original (issue #29).")
    right.markdown(f"### {label}")
    right.write(description)
    count_col, median_col = right.columns(2)
    count_col.metric("Fotos con esta escena", f"{len(with_scene):,}".replace(",", "."),
                     help=f"de {len(photos):,} fotos".replace(",", "."))
    median_col.metric("Puntuación mediana", fmt(with_scene.MOS.median(), 1) if len(with_scene) else "—")
    right.caption("Una foto puede tener varias escenas a la vez.")


def show_score_explorer() -> None:
    st.subheader("Explore la puntuación de calidad")
    st.markdown("Mueva el rango para ver cuántas fotos caen dentro.")
    photos = load_photos()
    low, high = st.slider("Rango de puntuación", 0, 100, DEFAULT_SCORE_RANGE, key="score_range")
    inside = photos.MOS.between(low, high)
    share = fmt(100 * inside.mean(), 1)
    st.write(f"**{int(inside.sum()):,}** de {len(photos):,} fotos ({share} %) tienen una puntuación "
             f"entre {low} y {high}.".replace(",", "."))

    labelled = photos.assign(Rango=np.where(inside, "Dentro del rango", "Fuera del rango"))
    figure = px.histogram(labelled, x="MOS", color="Rango", nbins=40, height=320,
                          labels={"MOS": "Puntuación de calidad", "count": "Fotos"},
                          color_discrete_map={"Dentro del rango": "#F58518", "Fuera del rango": "#9ecae1"})
    figure.update_layout(legend_title_text="", bargap=0.05)
    st.plotly_chart(figure, width="stretch")


def show() -> None:
    show_header()
    show_sources()
    show_dataset()
    show_guide()
    show_dictionary()
    show_scenes()
    show_score_explorer()
