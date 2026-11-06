# Auditoría final

Fecha: 2026-11-06. Estado del trabajo: **PENDIENTE_REVISION_ORQUESTADOR**. No está marcado READY_TO_SUBMIT. No hubo merge ni envío al LMS.

## Recuento

33 filas en la rúbrica.

- PASS: 30 (R01–R21, R23–R31).
- BLOCKED: 1 (R22, subida a 4Geeks.com prohibida por el encargo).
- NO APLICA: 2 (R10b, ruta de más de 12 GB, alternativa excluyente; R32, augmentation no pedido por los pasos del README).
- FAIL: 0.

Todos los criterios required que se podían cumplir en esta VM están en PASS. R22 no se convierte en PASS.

## Lo que se comprobó en esta auditoría

- El notebook ejecutado no tiene celdas de código sin `execution_count` ni salidas `error`.
- `nbconvert` devolvió `EXIT:0` a las 11:36:51+00:00.
- `pytest`: 8 passed (`reports/pytest.txt`).
- El checkpoint de 51 235 467 bytes sigue ignorado por git y su SHA-256 coincide con `reports/evaluation.json`.
- La pérdida de test (0,4657) corresponde al mejor `val_loss` (0,4656 en la época 6), no a la última época (0,5825). El modelo cargado es el checkpoint, no los pesos ya sobreajustados.
- Hay aciertos (1 940) y errores (560). La cuadrícula no inventa ninguno de los dos.
- No se versionan el zip, los JPEG ni `*.keras`.
- El fragmento 224×224 no se usó. El entrenamiento vio 200×200.
- `weights=None`. No se cargó ImageNet.
- No se leyó `solution.ipynb`.

## Límites que quedan a la vista

- Sin GPU. Ocho épocas, no más. `EarlyStopping` agotó la paciencia en la última época prevista.
- Keras 3.15 no tiene `Model.fit_generator` nativo. La llamada del notebook es real y delega en `Model.fit`.
- Arial y Helvetica no están instaladas. Las figuras usan Arimo.
- No hay UI, así que no hay capturas responsive, video ni Lighthouse.
- Los pesos viven en la máquina donde se entrenó (`models/dogs_vs_cats_best.keras`) y no en el repositorio. Repetir el notebook los regenera.

## Reparaciones

Una iteración antes de la corrida larga: dos tests fallaban (nombre de clase de EfficientNet y el parser de fuentes de Matplotlib) y el título de la cuadrícula habría escrito «Errore». Se corrigieron y la corrida única del notebook terminó bien. No hizo falta un segundo entrenamiento.
