# Calidad percibida de fotos de smartphone

Aplicación que muestra cómo perciben las personas la calidad de una foto tomada con el teléfono, y qué tan bien un modelo puede estimar esa calidad a partir de las características de la imagen.

## Qué puede hacer

- **Explorar las fotos:** filtrar por puntuación, características, escenas o fecha de captura.
- **Ver qué tan bien predice el modelo:** comparar los modelos y ver en qué casos se equivocan.
- **Ver grupos de fotos:** agrupar las fotos según sus características, sin mirar la puntuación.
- **Evaluar tus fotos:** subir fotos JPG y recibir una puntuación estimada.

## Ver la aplicación en su computador

Necesita Python 3.12. Desde la carpeta del proyecto:

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

Después abra en el navegador la dirección que aparece en la terminal, normalmente `http://localhost:8501`.

## Qué hay en cada carpeta

- `dashboard/`: la aplicación. `app.py` es el punto de entrada; cada sección tiene su archivo (`inicio.py`, `explorar.py`, `modelo.py`, `grupos.py`, `evaluar.py`).
- `src/`: el cálculo de las características de cada imagen y la preparación de las tablas.
- `data/processed/`: tablas que usa la aplicación.
- `models/`: modelos ya entrenados.
- `docs/`: notas técnicas del proyecto.

## Datos y créditos

- Conjunto de datos: **SPAQ** (Smartphone Photography Attribute and Quality), de Fang, Zhu, Zeng, Ma y Wang (2020), publicado en CVPR 2020. Enlace al artículo: [doi.org/10.1109/CVPR42600.2020.00373](https://doi.org/10.1109/CVPR42600.2020.00373).
- Repositorio original de SPAQ: [github.com/h4nwei/SPAQ](https://github.com/h4nwei/SPAQ). Licencia MIT según el titular ([issue #29](https://github.com/h4nwei/SPAQ/issues/29)).
- Este repositorio no incluye las fotografías completas. Incluye tablas derivadas, modelos entrenados y nueve fotos reducidas como ejemplo de escena. Todo se distribuye con esa atribución.
- Código de la aplicación: licencia MIT (ver [LICENSE](LICENSE)).

## Para quien quiera revisar el trabajo

- La explicación del proceso, las pruebas y las decisiones técnicas está en [docs/notas_tecnicas.md](docs/notas_tecnicas.md).
- Los archivos de datos, y cómo se generan, están en [data/README.md](data/README.md).
- El inventario de archivos y sus huellas está en [lista_de_archivos.csv](lista_de_archivos.csv).
