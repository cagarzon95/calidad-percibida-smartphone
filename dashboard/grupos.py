"""Grupos de fotos: cómo se agrupan las fotos según sus características, sin mirar la puntuación."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datos import load_group_choice, load_groups, load_photos
from extractor_caracteristicas import PREDICTORS
from textos import FEATURE_LABEL
from util import technical_names


def show_choice_of_groups() -> None:
    choice = load_group_choice()
    figure = go.Figure()
    figure.add_scatter(x=choice.K, y=choice.inertia, mode="lines+markers", name="Dispersión dentro de los grupos")
    figure.add_scatter(x=choice.K, y=choice.silhouette, mode="lines+markers",
                       name="Qué tan separados están", yaxis="y2")
    figure.update_layout(title="¿Cuántos grupos conviene? (se evaluaron de 2 a 8)", height=320,
                         xaxis_title="Número de grupos", yaxis=dict(title="Dispersión"),
                         yaxis2=dict(title="Separación", overlaying="y", side="right"))
    st.plotly_chart(figure, width="stretch")
    st.caption("Con 3 grupos la separación es la mayor de las opciones evaluadas, aunque moderada. "
               "Cuanto más alta, mejor separados están los grupos.")


def show_map(groups) -> None:
    method = st.radio("Forma de agrupar", ["Tres grupos", "Por densidad"], horizontal=True)
    column = "cluster_kmeans" if method == "Tres grupos" else "cluster_dbscan"
    groups = groups.assign(Grupo=groups[column].astype(str).replace({"-1": "Fotos aisladas"}))
    figure = px.scatter(groups, x="PCA1", y="PCA2", color="Grupo", opacity=0.5, height=480,
                        title="Mapa de las fotos de entrenamiento (dos dimensiones)",
                        labels={"PCA1": "Eje 1 del mapa", "PCA2": "Eje 2 del mapa"})
    st.plotly_chart(figure, width="stretch")
    if method == "Tres grupos":
        st.caption("El mapa resume muchas características en dos ejes. Es una vista parcial. "
                   "Los tres grupos tienen 1.313, 1.750 y 5.301 fotos de entrenamiento.")
    else:
        st.caption("Las fotos que no encajan en ninguna zona densa quedan como «aisladas». "
                   "Es un resultado del entrenamiento y no confirma que existan clases naturales.")


def show_profiles(photos, groups) -> None:
    joined = photos.merge(groups[["image_id", "cluster_kmeans"]], on="image_id", how="inner")
    medians = joined.groupby("cluster_kmeans")[PREDICTORS].median()
    scaled = (medians - medians.min()) / (medians.max() - medians.min())
    long = scaled.reset_index().melt(id_vars="cluster_kmeans", var_name="Característica", value_name="Posición")
    long["Característica"] = long["Característica"].map(FEATURE_LABEL)
    long["Grupo"] = "Grupo " + long["cluster_kmeans"].astype(str)
    figure = px.line_polar(long, r="Posición", theta="Característica", color="Grupo", line_close=True,
                           height=520, title="Perfil de cada grupo (0 = valor más bajo entre grupos, 1 = más alto)")
    st.plotly_chart(figure, width="stretch")
    st.caption("Mediana de cada característica dentro del grupo, escalada entre los tres grupos.")
    technical_names()

    st.subheader("Puntuación de calidad por grupo")
    scores = joined.groupby("cluster_kmeans").MOS.mean().reset_index()
    scores["Grupo"] = "Grupo " + scores["cluster_kmeans"].astype(str)
    figure = px.bar(scores, x="Grupo", y="MOS", color="Grupo", text="MOS", height=340,
                    title="Puntuación media de calidad por grupo", labels={"MOS": "Puntuación media"})
    figure.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    figure.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(figure, width="stretch")
    st.caption("La puntuación no se usó para formar los grupos; se muestra después para comparar.")


def show() -> None:
    st.title("Grupos de fotos")
    st.markdown("Aquí se agrupan las fotos según sus características, **sin mirar su puntuación**. "
                "Después se compara la puntuación de cada grupo, solo para entender qué tienen en común.")
    show_choice_of_groups()
    groups = load_groups()
    show_map(groups[groups.split == "train"].copy())
    show_profiles(load_photos(), groups[groups.split == "train"])
