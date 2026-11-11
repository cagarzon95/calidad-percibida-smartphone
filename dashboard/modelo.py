"""Qué tan bien predice el modelo: errores, escenas e influencia de cada característica."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datos import load_importance, load_model_summary, load_predictions, load_scene_metrics, load_training_curve
from textos import FEATURE_LABEL, SCENE_LABEL
from util import fmt, technical_names



def regression_table() -> pd.DataFrame:
    """Resultados de los modelos de regresión, incluida la regresión lineal del análisis previo."""
    summary = load_model_summary()
    results = summary[summary.task == "regression"].copy()
    linear = pd.DataFrame([{"model": "Regresión lineal", "task": "regression", "split": "validation",
                            "MAE": 12.5055, "RMSE": 15.2313, "R2": 0.4529}])
    return pd.concat([results, linear], ignore_index=True)


def show_error_charts() -> None:
    results = regression_table()
    col1, col2, col3 = st.columns(3)
    for col, metric, title in [(col1, "MAE", "Error medio (puntos)"),
                               (col2, "RMSE", "Error típico (puntos)"),
                               (col3, "R2", "Proporción explicada")]:
        figure = px.bar(results, x="model", y=metric, color="model", text=metric, height=340, title=title)
        figure.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        col.plotly_chart(figure, width="stretch")
    st.caption("Validación, n = 1.365 fotos. La regresión lineal no tiene penalización. Ridge aplica una penalización "
               "que evita el sobreajuste; por eso sus resultados casi no cambian.")


def show_scene_charts() -> None:
    summary = load_model_summary()
    classes = summary[summary.task == "multilabel_classification"].melt(
        id_vars="model", value_vars=["F1_micro", "AUC_macro"], var_name="Medida", value_name="Valor")
    classes["Medida"] = classes["Medida"].map({"F1_micro": "Acierto global", "AUC_macro": "Capacidad de distinguir"})
    figure = px.bar(classes, x="model", y="Valor", color="Medida", barmode="group", height=340,
                    title="Clasificación de escenas (validación)", labels={"model": "Modelo"})
    st.plotly_chart(figure, width="stretch")

    scenes = load_scene_metrics().copy()
    scenes["Escena"] = scenes["scene"].map(SCENE_LABEL)
    figure = px.bar(scenes.sort_values("F1", ascending=False), x="Escena", y="F1", color="Escena", text="F1",
                    height=380, title="Acierto por escena (validación)", labels={"F1": "Acierto"})
    figure.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    figure.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(figure, width="stretch")
    st.caption("Las escenas con pocas fotos (Animal, con 18 positivos en prueba) dan resultados poco estables.")


def show_prediction_scatter() -> None:
    predictions = load_predictions()
    st.subheader("Predicción frente a la puntuación real (validación)")
    st.caption("Modelo de bosque aleatorio. Es el único modelo con predicción foto a foto en validación.")
    absolute_error = (predictions.MOS - predictions.MOS_predicted).abs().mean()
    st.write(f"Error medio absoluto en validación: **{fmt(absolute_error, 1)} puntos**.")

    left, right = st.columns(2)
    figure = px.scatter(predictions, x="MOS", y="MOS_predicted", opacity=0.4,
                        title="Puntuación real frente a la estimada", height=420,
                        labels={"MOS": "Puntuación real", "MOS_predicted": "Puntuación estimada"})
    figure.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Acierto perfecto",
                                line=dict(dash="dash")))
    left.plotly_chart(figure, width="stretch")

    bands = predictions.groupby("MOS_band", observed=True).agg(
        Fotos=("MOS", "size"), error=("absolute_error", "mean")).reset_index()
    bands = bands.rename(columns={"MOS_band": "Rango de puntuación real", "error": "Error medio (puntos)"})
    figure = px.bar(bands, x="Rango de puntuación real", y="Error medio (puntos)", text="Fotos", height=420,
                    title="Error medio según el rango de puntuación real")
    figure.update_traces(texttemplate="%{text} fotos", textposition="outside")
    right.plotly_chart(figure, width="stretch")
    st.info("Lectura: el modelo tiende a acercarse al centro de la escala. Las fotos muy buenas o muy malas "
            "tienen el mayor error. Una relación entre características y calidad no demuestra causa.")


def show_influence() -> None:
    st.subheader("Características que más influyen")
    importance = load_importance().copy()
    importance["Característica"] = importance["predictor"].map(FEATURE_LABEL).fillna(importance["predictor"])
    figure = px.bar(importance, x="permutation_RMSE_mean", y="Característica", orientation="h",
                    error_x="permutation_RMSE_sd", height=460,
                    title="Cuánto empeora el modelo si se quita cada característica",
                    labels={"permutation_RMSE_mean": "Empeoramiento del error", "Característica": ""})
    figure.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figure, width="stretch")
    st.caption("Medido desordenando cada característica y viendo cuánto sube el error.")
    technical_names()


def show_training_curve() -> None:
    with st.expander("Curva de aprendizaje de la red neuronal (avanzado)"):
        history = load_training_curve()
        figure = px.line(history, x="epoch", y=["train_mse", "stop_mse"], height=360,
                         title="Error por época de entrenamiento",
                         labels={"value": "Error cuadrático medio", "epoch": "Época", "variable": "Serie"})
        names = {"train_mse": "Entrenamiento", "stop_mse": "Parada temprana"}
        figure.for_each_trace(lambda trace: trace.update(name=names.get(trace.name, trace.name)))
        st.plotly_chart(figure, width="stretch")


def show() -> None:
    st.title("Qué tan bien predice el modelo")
    st.markdown(
        "Se entrenó un modelo para estimar la puntuación de calidad a partir de características de la foto. "
        "Aquí se muestra qué tan bien lo hace, con datos que no usó para aprender.")
    show_error_charts()
    show_scene_charts()
    show_prediction_scatter()
    show_influence()
    show_training_curve()
