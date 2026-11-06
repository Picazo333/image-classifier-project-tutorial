# Dogs vs cats image classifier

Convolutional classifier for the 4Geeks project `image-classifier-project-tutorial`.
It trains EfficientNet-B0 from random weights (`weights=None`) on the official
dogs-vs-cats zip, using Keras generators at 200×200.

The guided run is the executed notebook `src/explore.ipynb`.
Metrics from that run are in `reports/evaluation.txt` and `reports/run_manifest.json`.
This README does not restate a number that is not in those files.

## Run

Tested with Python 3.11.18, TensorFlow 2.20.0 and Keras 3.15.1. No GPU.

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

`python src/app.py` checks the zip, builds the class folders, and prints
`reports/evaluation.json` when a checkpoint already exists. It does not train.

The notebook takes on the order of 70–90 minutes on the 4-core CPU used here:
8 epochs maximum, about 8.5 minutes of training steps each, plus validation.
`EarlyStopping` can stop sooner. See `MAX_EPOCHS_REASON` in `src/pipeline.py`.

## Dataset

Download the zip yourself. It is not in git.

- URL: https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip
- Bytes: 581507449
- SHA-256: `8aabb6755161f0ec2a37f4fa9cff733c304824fe0a5cee652cf021e979d0937b`

After unzipping the labeled `train/` folder there are 12 500 `cat.*.jpg` and
12 500 `dog.*.jpg`. A size census (`reports/image_size_survey.json`) found
25 000 RGB images, 8 513 distinct sizes, width 42–1050 and height 32–768.
24 997 of them are not already 200×200.

The zip also contains `sampleSubmission.csv`, which has no labels. The test
set used here is a held-out slice of the labeled photos, not that file.

Split, seed 42 for cats and 43 for dogs, per class: 10 000 train, 1 250 val,
1 250 test. Validation exists so `EarlyStopping` can watch `val_loss`.
Files in `data/processed/` are hard links into `data/raw/`, so the photos
are not duplicated.

## Model

```text
EfficientNetB0(include_top=False, weights=None, input_shape=(200, 200, 3))
GlobalAveragePooling2D
Dense(128, relu)
Dense(2, softmax)
```

The assignment's snippet uses `(224, 224, 3)`. The paragraph above that
snippet requires 200×200, and that is the size used.

Keras 3.15 removed `Model.fit_generator`. The notebook still calls
`model.fit_generator(...)`. `ensure_fit_generator` forwards that call to
`Model.fit`, which is the current API for generators and callbacks.
`ModelCheckpoint` and `EarlyStopping` are both passed in.

The best checkpoint is `models/dogs_vs_cats_best.keras`. Git ignores `*.keras`
and everything under `models/` except `.gitkeep`. The assignment does not ask
for Git LFS. Pixel values stay in 0–255 because this EfficientNet rescales
inside the network.

Optimizer: Adam, learning rate 1e-4, `clipnorm=1`. The assignment does not
name an optimizer.

## Layout

- `src/explore.ipynb` — steps, training, curves, confusion matrix, sample grid.
- `src/pipeline.py` — zip check, split, generators, model, `fit_generator` shim.
- `src/plots.py` — figure colors and type stack.
- `src/app.py` — prepare the split and print saved metrics.
- `src/utils.py` — database helper kept from the template. This project does not call it.
- `tests/test_load_pipeline.py` — batch shape 200×200, labels, seed, callbacks.
- `reports/` — curves, confusion matrix, evaluation, size census.
- `docs/upstream/` — original template README files and `learn.json`.
- `PROVENANCE.md` — template commit and dataset hash.
- `.devcontainer/` — template Codespaces setup, unchanged.

## Provenance

Folder layout, devcontainer, `src/utils.py` and the upstream READMEs come from
https://github.com/4GeeksAcademy/machine-learning-python-template
commit `b49eff6673b76ba5b0003103e32f62a2eb9904a4`.
That repository publishes no LICENSE file. None was added.
The template's own contributor note stays in `docs/upstream/`. This repository
does not add collaborators.

`solution.ipynb` and `solution.es.ipynb` were not downloaded or read.
Nothing is submitted to the 4Geeks LMS from this workspace.

## Figures

Plots use the academic palette `#18202B` on `#F7F8FA`, with `#204EA0` and
`#B53A17` for the two series. Arial and Helvetica are not installed here;
Matplotlib uses Arimo, which is the next face in the stack
(Arial, Helvetica, Arimo, Liberation Sans). There is no web UI, so the
4Geeks × Divinivid lockup is not used.
