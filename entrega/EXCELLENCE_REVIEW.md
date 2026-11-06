# Revisión de excelencia

Los extras no sustituyen al enunciado. La corrida de 8 épocas ya estaba medida cuando se cerró esta revisión; no se relanzó el entrenamiento.

## Extra visible — IMPLEMENTADO

Matriz de confusión y cuadrícula de aciertos y errores dentro del notebook ejecutado.

- Matriz: `[[912, 338], [222, 1028]]` en `reports/confusion_matrix.png`. Los números van en blanco o en tinta según el fondo de la celda, así que no dependen solo del color.
- Cuadrícula: 6 aciertos y 6 errores, `reports/sample_predictions.png`. Título con los totales reales (1 940 y 560).
- Evidencia: celdas del notebook `src/explore.ipynb` posteriores a la evaluación, y los PNG de `reports/`.

## Extra técnico — IMPLEMENTADO

Semilla fija, celda de evaluación que recarga el checkpoint, y tests del cargador.

- Semilla 42 documentada en `src/pipeline.py` y en `reports/run_manifest.json`.
- La celda de evaluación llama a `load_model` y vuelve a crear `tsdata`. La pérdida de test (0,4657) cuadra con el mejor `val_loss` (0,4656), no con la última época.
- `tests/test_load_pipeline.py`: 8 passed en `reports/pytest.txt`. El test no necesita las 25 000 fotos: fabrica JPEG pequeños y comprueba el lote 200×200, las etiquetas y que el callback de checkpoint escribe un archivo.

## Propuestas adicionales

### 1. Aumento de datos — DEFER

El texto corto del `learn.json` del tutorial menciona augmentation. Los pasos del README no. Añadirlo cambiaría el resultado ya medido y obligaría a otra corrida de más de una hora. No se hizo.

### 2. Pesos ImageNet — REJECT

El enunciado fija `weights=None`. Cargar ImageNet subiría la exactitud y incumpliría ese requisito.

### 3. Muestra de errores equilibrada entre clases — DEFER

La cuadrícula toma los primeros seis aciertos y los primeros seis errores del generador sin barajar. Como los gatos van primero, esas doce fotos son gatos. Es reproducible y cumple el extra (hay aciertos y hay errores). Una muestra estratificada se vería más variada y no se implementó para no cambiar la regla ya ejecutada.

## SIGNAL y lockup

No hay UI. El lockup 4Geeks × Divinivid no se colocó: el brief dice que sin UI no aplica, y el enunciado no pide una interfaz. Las figuras usan la paleta y, en lugar de Arial/Helvetica (no instaladas), Arimo. Detalle en `DESIGN-QA.md`.
