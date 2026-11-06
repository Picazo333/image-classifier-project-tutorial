# Auditoría final

Fecha: 2026-11-06. Estado del trabajo: **PENDIENTE_REVISION_ORQUESTADOR**. No está marcado READY_TO_SUBMIT. No hubo merge ni envío al LMS.

## Recuento

33 filas en la rúbrica.

- PASS: 30 (R01–R21, R23–R31).
- HITL_ENVIO_MIGUEL: 1 (R22). No cuenta contra el 100%.
- NO APLICA: 2 (R10b, ruta de más de 12 GB, alternativa excluyente; R32, augmentation no pedido por los pasos del README).
- BLOCKED: 0.
- FAIL: 0.

Denominador del 100%: 30 criterios (se excluyen R22 y los dos que no aplican). Esos 30 están en PASS: **30/30**.

## Lo que se comprobó en esta auditoría

- El notebook ejecutado no tiene celdas de código sin `execution_count` ni salidas `error`.
- `nbconvert` devolvió `EXIT:0` a las 11:36:51+00:00.
- `pytest`: 8 passed (`reports/pytest.txt`).
- El checkpoint de mejor `val_loss` pesa 51 235 467 bytes (menos de 100 MB) y se versiona. SHA-256 `056311485172c06ee84a3a00a24bb0b3ee57a00939af5e52c222943f98c8df95`.
- Recargado desde `models/dogs_vs_cats_best.keras`, sin reentrenar, el test repite exactitud 0,7760 y pérdida 0,4657174348831177 (`reports/reload_eval.json`). Esa pérdida es la del mejor `val_loss` (época 6, 0,4656), no la de la última época (0,5825).
- Hay aciertos (1 940) y errores (560). La cuadrícula no inventa ninguno de los dos.
- No se versionan el zip ni los JPEG. Otros `*.keras` siguen ignorados.
- El fragmento 224×224 no se usó. El entrenamiento vio 200×200.
- `weights=None`. No se cargó ImageNet.
- No se leyó `solution.ipynb`.

## Límites que quedan a la vista

- Sin GPU. Ocho épocas, no más. `EarlyStopping` agotó la paciencia en la última época prevista.
- Keras 3.15 no tiene `Model.fit_generator` nativo. La llamada del notebook es real y delega en `Model.fit`.
- Arial y Helvetica no están instaladas. Las figuras usan Arimo.
- No hay UI, así que no hay capturas responsive, video ni Lighthouse.
- El 77,6% de test queda por debajo del 80% del SVM de 2007 citado en el enunciado. El modelo salió de `weights=None`, en CPU y con 8 épocas. No se reentrenó para acercarse a esa cifra.

## Checklist de entrega

- [ ] R22 HITL_ENVIO_MIGUEL: Miguel sube el enlace de este repositorio a 4Geeks.com. Este trabajo no lo envía. No cuenta contra el 100% y no es un FAIL.

## Reparaciones

Una iteración antes de la corrida larga: dos tests fallaban (nombre de clase de EfficientNet y el parser de fuentes de Matplotlib) y el título de la cuadrícula habría escrito «Errore». Se corrigieron y la corrida única del notebook terminó bien. No hizo falta un segundo entrenamiento.
