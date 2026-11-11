"""Genera las tablas de la aplicación a partir del análisis original y de las anotaciones de SPAQ.

Uso:
    python src/preparar_datos.py --resultados <resultados> --annotations <SPAQ/Annotations> --out data/processed
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from extractor_caracteristicas import PREDICTORS, SCENES

EXIF_RENAME = {"Focal length": "exif_Focal length", "F-number": "exif_F-number",
               "Exposure time": "exif_Exposure time", "ISO": "exif_ISO"}
COPIES = {"resumen_modelos.csv": "resumen_modelos.csv",
          "metricas_clasificacion_por_escena.csv": "metricas_escenas.csv",
          "importancia_permutacion.csv": "importancia_caracteristicas.csv",
          "historial_mlp.csv": "historial_red_neuronal.csv",
          "seleccion_kmeans.csv": "seleccion_grupos.csv"}


def read_annotation(path, columns):
    """Lee una hoja de anotaciones y usa el nombre de archivo sin extensión como identificador."""
    frame = pd.read_excel(path)
    frame["image_id"] = frame["Image name"].map(lambda name: Path(str(name)).stem.zfill(5))
    return frame[["image_id", *columns]]


def capture_date(path):
    """Fecha y hora de captura guardadas por la cámara. Años anteriores a 2000 no son válidos."""
    with Image.open(path) as image:
        exif = image.getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    text = str(raw) if raw else ""
    if len(text) != 19 or not text[:4].isdigit() or int(text[:4]) < 2000:
        return None, None
    return text[:10].replace(":", "-"), text[11:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultados", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    images = args.images or args.annotations.parent / "TestImage"
    args.out.mkdir(parents=True, exist_ok=True)
    notes = args.annotations

    features = pd.read_csv(args.resultados / "descriptores_por_imagen.csv", dtype={"image_id": str})
    exif = read_annotation(notes / "EXIF_tags.xlsx", list(EXIF_RENAME)).rename(columns=EXIF_RENAME)
    photos = features[["image_id", "sha256", "split", *PREDICTORS]]
    for part in (read_annotation(notes / "MOS and Image attribute scores.xlsx", ["MOS"]),
                 read_annotation(notes / "Scene category labels.xlsx", SCENES), exif):
        photos = photos.merge(part, on="image_id", how="left")
    dates = [capture_date(images / f"{image_id}.jpg") for image_id in photos.image_id]
    photos["exif_fecha_captura"] = [day for day, _ in dates]
    photos["exif_hora_captura"] = [hour for _, hour in dates]
    assert len(photos) == 11125 and photos["MOS"].notna().all(), "tabla incompleta"
    photos.to_csv(args.out / "fotos.csv", index=False)

    predictions = pd.read_csv(args.resultados / "predicciones_regresion_validacion.csv", dtype={"image_id": str})
    predictions[["image_id", "MOS", "MOS_predicted", "residual", "absolute_error", "MOS_band",
                 "cluster_kmeans_posthoc"]].to_csv(args.out / "predicciones_validacion.csv", index=False)
    groups = pd.read_csv(args.resultados / "clusters_entrenamiento.csv", dtype={"image_id": str})
    groups[["image_id", "split", "MOS", "cluster_kmeans", "cluster_dbscan", "PCA1", "PCA2"]].to_csv(
        args.out / "grupos_entrenamiento.csv", index=False)
    for source, target in COPIES.items():
        shutil.copyfile(args.resultados / source, args.out / target)
    print("tablas escritas en", args.out.resolve())


if __name__ == "__main__":
    main()
