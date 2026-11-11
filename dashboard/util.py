"""Funciones pequeñas de apoyo: formato de números, miniaturas y etiquetas sobre las fotos."""
import io

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from extractor_caracteristicas import PREDICTORS
from textos import FEATURE_LABEL

BOLD_FONT = "C:/Windows/Fonts/arialbd.ttf"


def bold_font(size: int):
    """Fuente en negrita de Windows si existe; si no, la fuente que trae Pillow."""
    try:
        return ImageFont.truetype(BOLD_FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def fmt(value: float, digits: int = 2) -> str:
    """Número con coma decimal y punto de miles, como en español."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def thumbnail(content: bytes, size: int = 360) -> bytes:
    """Miniatura JPEG con la orientación corregida."""
    with Image.open(io.BytesIO(content)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((size, size))
        out = io.BytesIO()
        image.save(out, "JPEG", quality=85)
        return out.getvalue()


def label_photo(photo: bytes, lines: list) -> bytes:
    """Escribe texto grande y de color sobre la esquina superior de la foto."""
    with Image.open(io.BytesIO(photo)) as image:
        image = image.convert("RGB")
        width, _ = image.size
        small = bold_font(max(14, width // 20))
        big = bold_font(max(22, width // 9))
        fonts = [small] + [big] * (len(lines) - 1)
        draw = ImageDraw.Draw(image, "RGBA")
        pad = max(6, width // 40)
        heights = [draw.textbbox((0, 0), text, font=font)[3] for text, font in zip(lines, fonts)]
        widths = [draw.textbbox((0, 0), text, font=font)[2] for text, font in zip(lines, fonts)]
        box_height = sum(heights) + 2 * pad
        draw.rectangle((0, 0, min(width, max(widths) + 2 * pad), box_height), fill=(31, 119, 180, 235))
        y = pad
        for text, font, height in zip(lines, fonts, heights):
            draw.text((pad, y), text, font=font, fill=(255, 255, 255))
            y += height
        out = io.BytesIO()
        image.save(out, "JPEG", quality=88)
        return out.getvalue()


def feature_table(keys: list) -> pd.DataFrame:
    """Tabla con el nombre para el usuario y el nombre técnico de cada característica."""
    return pd.DataFrame({"Nombre en la aplicación": [FEATURE_LABEL[k] for k in keys], "Nombre técnico": keys})


def technical_names() -> None:
    """Desplegable con los nombres técnicos, para quien quiera verificarlos."""
    with st.expander("Ver nombres técnicos de las características"):
        st.dataframe(feature_table(PREDICTORS), hide_index=True, width="stretch")
