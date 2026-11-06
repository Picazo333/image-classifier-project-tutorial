# Clasificador de imágenes: perros y gatos

Clasificador convolucional del proyecto 4Geeks `image-classifier-project-tutorial`.
Entrena EfficientNet-B0 desde pesos aleatorios (`weights=None`) con el zip
oficial de perros y gatos, usando generadores de Keras a 200×200.

La corrida guiada es el notebook ejecutado `src/explore.ipynb`.
Las métricas de esa corrida están en `reports/evaluation.txt` y
`reports/run_manifest.json`.

Modelo versionado en git normal, sin LFS: `models/dogs_vs_cats_best.keras`, 51 235 467 bytes (unos 51,2 MB), SHA-256 `056311485172c06ee84a3a00a24bb0b3ee57a00939af5e52c222943f98c8df95`.

## Resultado de la corrida ejecutada

Solo CPU, 8 épocas completas, sin recortar imágenes. `EarlyStopping(patience=2)`
paró en la última época prevista, después de dos épocas en las que subió
`val_loss`. El checkpoint conservado es el de menor `val_loss`: la época 6 de 8
(0,4656), no la última (0,5825).

En las 2 500 fotos de test, el checkpoint recargado da pérdida 0,4657 y
exactitud 0,7760 (1 940 aciertos, 560 errores). El 77,6% está por debajo del
80% del SVM de 2007 que menciona el enunciado: el entrenamiento fue desde cero
(`weights=None`), en CPU y con 8 épocas. La matriz de confusión, con filas de
etiqueta real cat, dog:

|  | pred cat | pred dog |
| --- | ---: | ---: |
| cat | 912 | 338 |
| dog | 222 | 1028 |

El detalle por clase está en `reports/evaluation.txt`. Las curvas, en
`reports/training_curves.png`.

## Cómo correrlo

Probado con Python 3.11.18, TensorFlow 2.20.0 y Keras 3.15.1. Sin GPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/dogs-vs-cats.zip \
  https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip
pytest -q
jupyter nbconvert --to notebook --execute src/explore.ipynb \
  --inplace --ExecutePreprocessor.timeout=21600 \
  --ExecutePreprocessor.kernel_name=python3
```

`python src/app.py` comprueba el zip, arma las carpetas por clase y, si ya
existe un checkpoint, imprime `reports/evaluation.json`. No entrena.

En la CPU de 4 núcleos usada aquí el notebook tarda del orden de 70–90 minutos:
8 épocas como máximo, unos 8,5 minutos de pasos de entrenamiento cada una,
más la validación. `EarlyStopping` puede cortar antes. El motivo está en
`MAX_EPOCHS_REASON` dentro de `src/pipeline.py`.

## Dataset

Hay que descargarlo. No está en git.

- URL: https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip
- Bytes: 581507449
- SHA-256: `8aabb6755161f0ec2a37f4fa9cff733c304824fe0a5cee652cf021e979d0937b`

La carpeta etiquetada `train/` trae 12 500 `cat.*.jpg` y 12 500 `dog.*.jpg`.
El censo de tamaños (`reports/image_size_survey.json`) midió 25 000 imágenes
RGB, 8 513 tamaños distintos, anchura 42–1050 y altura 32–768.
24 997 no eran ya de 200×200.

`sampleSubmission.csv` no trae etiquetas. El test de este trabajo es un corte
de las fotos etiquetadas, no ese archivo.

Corte por clase, semilla 42 en gatos y 43 en perros: 10 000 train, 1 250 val
y 1 250 test. La validación sirve para que `EarlyStopping` vigile `val_loss`.
Los archivos de `data/processed/` son hard links a `data/raw/`: las fotos no
se duplican.

## Modelo

```text
EfficientNetB0(include_top=False, weights=None, input_shape=(200, 200, 3))
GlobalAveragePooling2D
Dense(128, relu)
Dense(2, softmax)
```

El fragmento del enunciado usa `(224, 224, 3)`. El párrafo anterior exige
200×200, y ese es el tamaño usado.

Keras 3 no incluye `Model.fit_generator`, así que `ensure_fit_generator` instala ese método como una delegación a `Model.fit`, que entrena el generador y recibe los callbacks. El notebook llama a `model.fit_generator(...)`. Ahí entran `ModelCheckpoint` y `EarlyStopping`.

El checkpoint de mejor `val_loss` está en `models/dogs_vs_cats_best.keras` (51 235 467 bytes, menos de 100 MB, por eso se versiona). SHA-256: `056311485172c06ee84a3a00a24bb0b3ee57a00939af5e52c222943f98c8df95`. Keras 3.15 escribe `.keras`; el enunciado no nombra `.h5` ni pide Git LFS. El resto de `*.keras` sigue ignorado. Una recarga posterior de ese archivo, sin volver a entrenar, repitió la exactitud 0,7760 (`reports/reload_eval.json`). Los píxeles se dejan en 0–255 porque este EfficientNet reescala dentro de la red.

Optimizador: Adam, tasa 1e-4, `clipnorm=1`. El enunciado no fija el optimizador.

## Estructura

- `src/explore.ipynb` — pasos, entrenamiento, curvas, matriz de confusión y muestra de aciertos y errores.
- `src/pipeline.py` — zip, corte, generadores, modelo y el puente de `fit_generator`.
- `src/plots.py` — paleta y pila tipográfica de las figuras.
- `src/app.py` — prepara el corte e imprime métricas ya guardadas.
- `src/utils.py` — ayuda de base de datos de la plantilla. Este proyecto no la llama.
- `tests/test_load_pipeline.py` — forma 200×200, etiquetas, semilla y callbacks.
- `reports/` — curvas, matriz, evaluación y censo de tamaños.
- `docs/upstream/` — README y `learn.json` originales de la plantilla.
- `PROVENANCE.md` — commit de la plantilla y hash del dataset.
- `.devcontainer/` — Codespaces de la plantilla, sin cambios.

## Procedencia

La estructura, el devcontainer, `src/utils.py` y los README de
`docs/upstream/` vienen de
https://github.com/4GeeksAcademy/machine-learning-python-template
commit `b49eff6673b76ba5b0003103e32f62a2eb9904a4`.
Ese repositorio no publica un archivo LICENSE. No se añadió uno.
La nota de colaboradores de la plantilla se queda en `docs/upstream/`.
Este repositorio no añade colaboradores.

No se descargó ni se leyó `solution.ipynb` ni `solution.es.ipynb`.
Desde este workspace no se envía nada al LMS de 4Geeks.

## Figuras

Las gráficas usan la paleta académica: texto `#18202B` sobre `#F7F8FA`,
series `#204EA0` y `#B53A17`. Arial y Helvetica no están instaladas;
Matplotlib usa Arimo, la siguiente de la pila
(Arial, Helvetica, Arimo, Liberation Sans). No hay interfaz web, así que
el lockup 4Geeks × Divinivid no aplica.
