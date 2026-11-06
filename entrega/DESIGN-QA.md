# DESIGN-QA

## Alcance

No hay interfaz web ni CLI interactiva con pantallas. El entregable visible es el notebook, el README y las figuras. El lockup 4Geeks × Divinivid no aplica. No se añadió una UI solo para mostrarlo.

El paquete `SIGNAL_PACKAGE.zip` y su `MANIFEST.json` no estaban entre los adjuntos de esta ejecución (sí el brief). No se copió ningún SVG o PNG de marca, así que no hay hash de asset que comparar.

## Qué se revisó

| Comprobación | Resultado | Motivo |
| --- | --- | --- |
| Capturas 390 / 768 / 1280 | NO APLICA | No hay UI. |
| Video de recorrido | NO APLICA | No hay UI. |
| Lighthouse (LCP, CLS, TBT/INP) | NO APLICA | No hay build de frontend. |
| Lockup en footer | NO APLICA | Sin UI. |
| `prefers-reduced-motion` | NO APLICA | Las figuras son PNG estáticos. No hay animación. |
| SEO, canonical, sitemap | NO APLICA | No es una landing pública. |
| Contraste de las figuras | HECHO | Medido en la corrida. |
| Tipografía | HECHO | Pila Arial, Helvetica, Arimo, Liberation Sans. Cara resuelta: Arimo. |
| Jerarquía del README | HECHO | Título, resultado medido, cómo correr, dataset, modelo, estructura, procedencia. |

## Contraste medido

Salida de `contrast_report()` guardada en `reports/run_manifest.json`:

| Par | Ratio | Umbral usado |
| --- | ---: | --- |
| `#18202B` sobre `#F7F8FA` | 15,44 | texto 4,5 |
| `#18202B` sobre `#FFFFFF` | 16,40 | texto 4,5 |
| `#505E70` sobre `#F7F8FA` | 6,22 | texto 4,5 |
| `#FFFFFF` sobre `#B53A17` | 5,86 | texto 4,5 |
| `#204EA0` sobre `#FFFFFF` | 7,90 | gráfico 3,0 |
| `#B53A17` sobre `#FFFFFF` | 5,86 | gráfico 3,0 |
| `#236344` sobre `#FFFFFF` | 7,14 | gráfico 3,0 |
| `#A32A26` sobre `#FFFFFF` | 7,20 | gráfico 3,0 |

Los tests de `tests/test_load_pipeline.py` fallan si un par de texto baja de 4,5 o un par gráfico baja de 3.

## Figuras

- Curvas: entrenamiento en línea continua `#204EA0`, validación en línea discontinua `#B53A17`, leyenda con texto. No se distingue solo por color. `reports/training_curves.png`.
- Matriz: cada celda lleva el conteo. El texto pasa a blanco cuando la celda es oscura. `reports/confusion_matrix.png`.
- Muestra: borde verde y la palabra «Acierto», borde granate y la palabra «Error». `reports/sample_predictions.png`.
- Nueve perros y nueve gatos: tamaño original escrito en cada título (`499×375 RGB`, etc.).

Fondo de figura `#F7F8FA`, ejes en blanco, títulos en negrita. No hay gradientes neón, ni tres tarjetas vacías, ni métricas inventadas: los números son los de esta corrida.

## README

Copy concreto (qué archivo correr, qué URL descargar, qué exactitud salió). Sin eslogan. La procedencia de la plantilla está separada de la autoría de este trabajo.
