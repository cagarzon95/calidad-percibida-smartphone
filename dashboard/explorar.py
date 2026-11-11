"""Explorar las fotos: filtros, indicadores y gráficas de la selección."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datos import load_photos
from extractor_caracteristicas import PREDICTORS, SCENES
from textos import EXIF_VARS, FEATURE_LABEL, PARTITION_NAMES, SCENE_LABEL
from util import fmt

NUMERIC_NAMES = {"MOS": "Puntuación de calidad", **{p: FEATURE_LABEL[p] for p in PREDICTORS}, **EXIF_VARS}
PARTITIONS = ["train", "validation", "test"]


def sidebar_filters(photos: pd.DataFrame):
    """Filtros de la izquierda. Devuelve la máscara de fotos seleccionadas."""
    with st.sidebar:
        st.subheader("Filtros")
        score = st.slider("Puntuación de calidad", 0.0, 100.0,
                          (float(photos.MOS.min()), float(photos.MOS.max())))
        variable = st.selectbox("Característica para filtrar", list(NUMERIC_NAMES),
                                format_func=lambda key: NUMERIC_NAMES[key])
        low, high = float(photos[variable].min()), float(photos[variable].max())
        value_range = st.slider(f"Rango de: {NUMERIC_NAMES[variable]}", low, high, (low, high))
        scenes = st.multiselect("Escenas", SCENES, default=[], format_func=lambda key: SCENE_LABEL[key])
        match_any = st.radio("Cuando elige varias escenas",
                             ["Basta con que aparezca una (O)", "Deben aparecer todas (Y)"]) == "Basta con que aparezca una (O)"
        groups = st.multiselect("Grupo de datos", PARTITIONS, default=PARTITIONS,
                                format_func=lambda key: PARTITION_NAMES[key])
        dates = photos.fecha.dropna()
        first, last = dates.min().date(), dates.max().date()
        date_range = st.slider("Fecha de captura", first, last, (first, last))
        include_undated = st.checkbox("Incluir fotos sin fecha válida", value=True)
        st.caption("Una escena cuenta como presente si al menos un evaluador la marcó.")

    in_dates = photos.fecha.between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    mask = (photos.MOS.between(*score) & photos[variable].between(*value_range)
            & photos.split.isin(groups) & (in_dates | (photos.fecha.isna() & include_undated)))
    if scenes:
        present = photos[scenes] > 0
        mask &= present.any(axis=1) if match_any else present.all(axis=1)
    return mask, variable


def show_indicators(photos: pd.DataFrame, selected: pd.DataFrame) -> None:
    total, shown = len(photos), len(selected)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fotos seleccionadas", f"{shown:,} de {total:,}".replace(",", "."))
    col2.metric("Puntuación media", fmt(selected.MOS.mean()) if shown else "—",
                delta=f"{fmt(selected.MOS.mean() - photos.MOS.mean())} frente al total" if shown else None)
    col3.metric("Puntuación mediana", fmt(selected.MOS.median()) if shown else "—")
    col4.metric("Parte del total", fmt(100 * shown / total, 1) + " %")


def show_quality(selected: pd.DataFrame) -> None:
    with st.expander("Calidad de los datos y relaciones entre características"):
        columns = ["MOS", *PREDICTORS, *EXIF_VARS]
        rows = []
        for column in columns:
            values = selected[column]
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            spread = q3 - q1
            outliers = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
            rows.append({"Variable": NUMERIC_NAMES[column], "Valores atípicos": outliers})
        figure = px.bar(pd.DataFrame(rows), x="Variable", y="Valores atípicos", color="Variable", height=380,
                        title="Valores atípicos por variable (selección actual)")
        figure.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(figure, width="stretch")
        st.caption("Valores atípicos: fuera del rango normal de la variable (método de los cuartiles).")

        if len(selected) >= 3:
            corr = selected[columns].corr(method="spearman").round(2)
            corr.index = corr.columns = [NUMERIC_NAMES[c] for c in columns]
            figure = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                               height=520, title="Relación entre variables (1 = se mueven juntas, −1 = en sentido contrario)")
            st.plotly_chart(figure, width="stretch")


def show_charts(photos: pd.DataFrame, selected: pd.DataFrame, variable: str) -> None:
    left, right = st.columns(2)
    histogram = go.Figure()
    histogram.add_histogram(x=photos.MOS, name="Todas las fotos", histnorm="probability", opacity=0.45)
    histogram.add_histogram(x=selected.MOS, name="Selección", histnorm="probability", opacity=0.75)
    histogram.update_layout(barmode="overlay", title="Cómo se reparten las puntuaciones",
                            xaxis_title="Puntuación de calidad", yaxis_title="Proporción de fotos", height=380)
    left.plotly_chart(histogram, width="stretch")

    sample = selected.sample(min(3000, len(selected)), random_state=7).copy()
    sample["Grupo de datos"] = sample["split"].map(PARTITION_NAMES)
    x_column = PREDICTORS[0] if variable == "MOS" else variable
    scatter = px.scatter(sample, x=x_column, y="MOS", color="Grupo de datos", opacity=0.4,
                         labels={x_column: NUMERIC_NAMES[x_column], "MOS": "Puntuación de calidad"},
                         title=f"{NUMERIC_NAMES[x_column]} frente a la puntuación (muestra de hasta 3.000 fotos)",
                         height=380)
    right.plotly_chart(scatter, width="stretch")

    counts = (selected[SCENES] > 0).sum().reset_index()
    counts.columns = ["escena", "n"]
    counts["Escena"] = counts["escena"].map(SCENE_LABEL)
    counts["%"] = 100 * counts["n"] / len(selected)
    bars = px.bar(counts, x="Escena", y="%", text="n", height=340,
                  title="Escenas presentes en la selección")
    bars.update_traces(texttemplate="%{text}", textposition="outside")
    bars.update_layout(xaxis_title=None, yaxis_title="% de las fotos seleccionadas")
    st.plotly_chart(bars, width="stretch")
    valid_dates = f"{int(photos.fecha.notna().sum()):,}".replace(",", ".")
    total = f"{len(photos):,}".replace(",", ".")
    st.info("Una foto puede tener varias escenas, por eso los porcentajes suman más de 100 %. "
            "Fecha de captura: la que la cámara guardó en cada foto. "
            f"Tienen fecha válida {valid_dates} de {total} fotos.")


def show() -> None:
    st.title("Explorar las fotos")
    st.markdown("Use los filtros de la izquierda. Los números se recalculan con la selección; el total aparece junto a cada cifra.")
    photos = load_photos()
    mask, variable = sidebar_filters(photos)
    selected = photos[mask]
    show_indicators(photos, selected)
    show_quality(selected)
    if len(selected) < 30:
        st.warning("Hay menos de 30 fotos seleccionadas: los promedios son poco estables.")
        return
    show_charts(photos, selected, variable)
