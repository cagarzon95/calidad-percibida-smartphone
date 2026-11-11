"""Nombres y textos que ve el usuario."""

# Características de la imagen: nombre para el usuario y explicación.
FEATURES = {
    "s512_luma_mean": ("Brillo medio", "Qué tan clara es la foto en promedio (0 = negro, 1 = blanco)."),
    "s512_luma_std": ("Variación del brillo", "Cuánto cambia el brillo dentro de la foto. Valores altos indican más contraste."),
    "heterogeneity_ratio": ("Desigualdad del brillo entre zonas", "Compara cuánto varía el brillo entre zonas de la foto con su variación total."),
    "luma_p95_minus_p05": ("Rango de brillo útil", "Diferencia entre las zonas muy claras y las muy oscuras."),
    "s512_gradient_mean": ("Nitidez de bordes", "Qué tan marcados son los contornos y los cambios bruscos de luz."),
    "s512_laplacian_var": ("Detalle fino", "Cantidad de detalle pequeño. Baja con fotos borrosas."),
    "s512_fft_high_fraction": ("Proporción de detalle fino", "Parte de la imagen formada por detalles finos o granulado."),
    "s512_entropy": ("Variedad de tonos de luz", "Cuántos niveles distintos de brillo aparecen en la foto."),
    "hsv_s_mean": ("Saturación de color", "Qué tan intensos son los colores en promedio."),
    "lab_chroma_mean": ("Intensidad de color", "Fuerza de los colores medida con un criterio parecido a la vista humana."),
    "color_entropy": ("Variedad de colores", "Cuántos colores distintos aparecen en la foto."),
    "neighbor_absdiff": ("Diferencia entre píxeles vecinos", "Cuánto cambia cada punto frente a sus vecinos. Se relaciona con el grano."),
    "lbp_entropy": ("Complejidad de texturas", "Variedad de patrones de textura en las superficies de la foto."),
    "tile_entropy_std": ("Variación de complejidad entre zonas", "Cuánto cambia la complejidad de una zona de la foto a otra."),
}
FEATURE_LABEL = {key: value[0] for key, value in FEATURES.items()}

EXIF_VARS = {
    "exif_ISO": "Sensibilidad ISO",
    "exif_Exposure time": "Tiempo de exposición (s)",
    "exif_F-number": "Apertura (número f)",
    "exif_Focal length": "Distancia focal (mm)",
}

SCENE_INFO = {
    "Animal": ("Animal", "Un animal es el sujeto principal, por ejemplo un perro o un pájaro."),
    "Cityscape": ("Ciudad", "Calles, edificios o paisajes urbanos, por ejemplo una avenida con edificios."),
    "Human": ("Personas", "Personas como sujeto principal, por ejemplo un retrato o un grupo."),
    "Indoor scene": ("Interior", "Espacios cerrados, por ejemplo una habitación o una cocina."),
    "Landscape": ("Paisaje natural", "Campo, montañas, mar o cielo abierto, por ejemplo un valle al atardecer."),
    "Night scene": ("Escena nocturna", "Foto tomada de noche o con poca luz, por ejemplo las luces de una calle."),
    "Plant": ("Plantas", "Plantas, flores u hojas como sujeto principal, por ejemplo una flor en primer plano."),
    "Still-life": ("Bodegón", "Objetos quietos dispuestos a propósito, por ejemplo frutas sobre una mesa."),
    "Others": ("Otros", "Ninguna de las anteriores, por ejemplo objetos sueltos o texturas."),
}
SCENE_LABEL = {key: value[0] for key, value in SCENE_INFO.items()}

PARTITION_NAMES = {"train": "Entrenamiento", "validation": "Validación", "test": "Prueba"}

# Puntos de partida para el rango de la puntuación en el inicio.
DEFAULT_SCORE_RANGE = (40, 70)
