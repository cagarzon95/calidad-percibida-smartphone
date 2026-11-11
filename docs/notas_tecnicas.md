# Notas técnicas

Este documento explica cómo se construyó la aplicación, cómo se comprobó y qué limitaciones tiene.

## 1. Datos de origen

- 11.125 fotografías JPEG del conjunto SPAQ, con puntuación de calidad (promedio de opiniones de personas, de 0 a 100).
- Tres tablas de anotaciones: puntuaciones, etiquetas de escena y datos de la cámara.
- Las fotografías completas no se incluyen en el repositorio.

## 2. Qué se calculó y de dónde sale

- **Características de la imagen (68):** brillo, contraste, nitidez, color, textura y detalle en tres tamaños de imagen. De ellas se eligieron 14 para el modelo.
- **Modelo de puntuación:** bosque aleatorio (Random Forest) entrenado con los datos de entrenamiento y validación. Los demás modelos (regresión lineal, Ridge y red neuronal) se usan solo para comparar.
- **Modelo de escenas:** clasificador por vecinos más cercanos, con una etiqueta por escena.
- **Agrupación:** tres grupos (K-Means) calculados solo con los datos de entrenamiento, y una agrupación por densidad (DBSCAN) que solo se muestra para las fotos del conjunto, no para fotos nuevas.
- **Mapa de dos dimensiones (PCA):** resume las características en dos ejes para visualizarlas.

Los modelos y resultados se calcularon en un análisis previo. Esta aplicación no los vuelve a entrenar; solo los carga y los muestra.

## 3. Cómo se comprobó

- Las cifras de la aplicación coinciden con las del análisis original: error medio del modelo de puntuación, 8,93 puntos en validación; proporción explicada, 0,68.
- Las características de tres fotografías del conjunto coinciden con las del cálculo original.
- La fecha de captura se leyó de los datos EXIF de cada foto (11.122 de 11.125 tienen fecha válida).
- Una foto nueva pasa por el mismo cálculo que las fotos del conjunto, incluida la corrección de orientación y el paso al espacio de color sRGB.

## 4. Decisiones de diseño

- Las fotos cargadas no se guardan. Solo se calcula su huella digital (SHA-256) para identificarlas.
- Se aceptan hasta 20 fotos por lote, de 15 MB cada una y 50 MB en total.
- La puntuación estimada se presenta sin rango de error. El error medio del modelo está en la sección de desempeño.

## 5. Limitaciones

- La puntuación es una opinión humana promedio, no una medida objetiva de la cámara.
- Las relaciones entre características y calidad no demuestran causa.
- El modelo estima el promedio de opiniones, no la opinión de un calificador concreto.
- Las etiquetas de escena vienen de votaciones y no son exactas.
- La fecha de captura la guarda la cámara, así que puede estar mal configurada en algunas fotos.

## 6. Cómo reproducir la preparación de datos

Requiere la carpeta de resultados del análisis original y la carpeta de anotaciones de SPAQ:

```bash
python src/preparar_datos.py --resultados "<carpeta de resultados originales>" --annotations "<carpeta SPAQ/Annotations>" --out data/processed
```
