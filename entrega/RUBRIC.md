# Rúbrica — image-classifier-project-tutorial

Fuente de autoridad: README del enunciado en `4GeeksAcademy/image-classifier-project-tutorial` commit `aa69493bc3a3930e1656f089498ba70bb9786734`. El encargo de ejecución (generadores, 200×200, no subir imágenes ni pesos, no LMS, no merge, no leer `solution.ipynb`) manda cuando recorta un paso del enunciado; eso se marca BLOCKED u omitido, no PASS.

| ID | Required | Criterio |
| --- | --- | --- |
| R01 | sí | El repo parte de `4GeeksAcademy/machine-learning-python-template`. El árbol de ese commit está copiado, el commit de origen está documentado y no se inventa una licencia ni colaboradores. |
| R02 | sí | La solución no sale de `solution.ipynb` ni de `solution.es.ipynb` (no se descargan ni se leen). |
| R03 | sí | El zip se descarga de `https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip` y el archivo local coincide en tamaño con la respuesta HTTP. |
| R04 | sí | El zip se descomprime. Hay una carpeta `train` con JPEG de perros y gatos etiquetados por el nombre de archivo (`dog` / `cat`). El conteo real queda escrito. |
| R05 | sí | Las imágenes no están en git. El README y `.gitignore` explican la descarga. |
| R06 | sí | Las primeras nueve fotos de perros se muestran en una sola figura. |
| R07 | sí | Las primeras nueve fotos de gatos se muestran en una sola figura. |
| R08 | sí | Queda constatado que las fotos son en color y tienen tamaños distintos, antes de fijar el tamaño. |
| R09 | sí | El entrenamiento ve las imágenes a 200×200. |
| R10 | sí | Ruta de poca RAM: `ImageDataGenerator` y `flow_from_directory`, con el motivo escrito (RAM útil / instrucción / tamaño del tensor). |
| R10b | alternativa excluyente | Ruta de más de 12 GB: cargar las 25.000 fotos en memoria. No se implementa si se cumple R10. |
| R11 | sí | Hay directorios separados de train y de test, y dentro de cada uno un subdirectorio por clase. |
| R12 | sí | Existen los objetos `trdata` (train) y `tsdata` (test) creados con el generador. |
| R13 | sí | Las etiquetas salen de los directorios de clase, no de un CSV escrito a mano. |
| R14 | sí | Arquitectura: `EfficientNetB0(include_top=False, weights=None)` + `GlobalAveragePooling2D` + `Dense(128, relu)` + `Dense(2, softmax)`. El fragmento del enunciado usa `input_shape=(224, 224, 3)`; el texto obliga 200×200. Se usa `(200, 200, 3)` y el conflicto queda documentado. |
| R15 | sí | El modelo se compila, se entrena de verdad y se mide el rendimiento (pérdida y exactitud). |
| R16 | sí | Se importan `ModelCheckpoint` y `EarlyStopping`. |
| R17 | sí | Se crean ambos objetos y se pasan como callbacks a `fit_generator`. |
| R18 | sí | Se carga el mejor modelo guardado y se predicen las imágenes de test. |
| R19 | sí | El modelo se guarda en `models/` (carpeta de modelos de la plantilla). Los pesos no entran en git: el enunciado no pide Git LFS y el encargo prohíbe subirlos. |
| R20 | sí | `src/explore.ipynb` queda ejecutado, con salidas reales de esta corrida. |
| R21 | sí | README técnico y neutro: qué es, cómo correrlo, estructura, decisiones, descarga del dataset, sin autoría inventada. |
| R22 | sí (enunciado) / prohibido por el encargo | Subir el enlace del repo a 4Geeks.com. No se hace. |
| R23 | sí (encargo) | Rama de trabajo y pull request contra `main`, sin merge. |
| R24 | sí (fase 2, sin UI) | SIGNAL en README y gráficas: jerarquía, Arial/Helvetica o el sans disponible documentado, paleta y contraste AA. El lockup no aplica. |
| R25 | sí (fase 2) | `DESIGN-QA.md` con lo que no aplica justificado (no hay UI). |
| R26 | extra visible exigido por el encargo | Matriz de confusión en el notebook ejecutado. |
| R27 | extra visible exigido por el encargo | Cuadrícula de predicciones de muestra con aciertos y errores. Si un grupo no existe, no se inventa: el criterio no pasa. |
| R28 | extra técnico | Semilla fija y usada en el split, los generadores y TensorFlow. |
| R29 | extra técnico | Celda de evaluación que recarga el modelo desde disco y recalcula las métricas de test. |
| R30 | extra técnico | Test o script mínimo que verifica el pipeline de carga (forma 200×200, etiquetas, semilla del lote). |
| R31 | sí (entrega) | `TELEMETRY.json` con tiempos reales, iteraciones EVAL→REPAIR, y tokens en `null` más `tokens_unmeasurable_reason` y `tokens_proxy`. Ninguna cifra de tokens inventada. |
| R32 | no exigido por el enunciado | Aumento de datos. Lo menciona el texto corto de `learn.json` del tutorial, no los pasos del README. No se implementa y no cuenta como fallo. |

Estados finales: ver `RUBRIC_JUSTIFICATION.md` después de la evaluación. Un criterio no se marca PASS sin evidencia. UNVERIFIED no es PASS.
