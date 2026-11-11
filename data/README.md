# Datos

Este repositorio no incluye las fotografías ni las anotaciones originales de SPAQ. Incluye tablas derivadas y nueve fotos reducidas, una por escena (en `dashboard/assets/escenas/`), con atribución según el README principal.

## Fuente

- Repositorio original: https://github.com/h4nwei/SPAQ
- Artículo: Fang, Y., Zhu, H., Zeng, Y., Ma, K., & Wang, Z. (2020). *Perceptual Quality Assessment of Smartphone Photography*. CVPR 2020. https://doi.org/10.1109/CVPR42600.2020.00373
- Licencia declarada por el titular del repositorio (issue #29, 20 de septiembre de 2023): MIT. El repositorio no tiene archivo `LICENSE`.
- Justificación de la fuente: SPAQ es un conjunto de datos académico, presentado en una conferencia científica (CVPR 2020). Es de acceso público y se distribuye con atribución bajo licencia MIT, según declara su titular.

## Archivos incluidos

| Archivo | Contenido |
|---|---|
| `fotos.csv` | Una fila por foto: identificador, puntuación de calidad, escenas, datos de la cámara y 14 características de la imagen |
| `predicciones_validacion.csv` | Puntuación real y estimada de cada foto del grupo de validación |
| `grupos_entrenamiento.csv` | Grupo asignado y posición en el mapa de cada foto de entrenamiento |
| `resumen_modelos.csv` | Errores y aciertos de cada modelo |
| `metricas_escenas.csv` | Aciertos del clasificador de escenas |
| `importancia_caracteristicas.csv` | Cuánto empeora el modelo al quitar cada característica |
| `historial_red_neuronal.csv` | Error por época de la red neuronal |
| `seleccion_grupos.csv` | Medidas para elegir el número de grupos |

## Cómo se generan

Las tablas se generan a partir de los resultados del análisis original y de las anotaciones de SPAQ:

```bash
python src/preparar_datos.py --resultados "<carpeta de resultados originales>" --annotations "<carpeta SPAQ/Annotations>" --out data/processed
```

Sin estas tablas, la aplicación no puede mostrar las secciones de exploración, modelo y grupos. La sección de evaluación de fotos usa los modelos incluidos en `models/`.
