# Letters NN — Handwritten Letter Classifier (NumPy, from scratch)

A clean, professional implementation of a **single-hidden-layer neural network** (sigmoid hidden, softmax output) trained with **SGD** to classify handwritten letters from CSV datasets. No ML frameworks — just **NumPy**.

## Highlights
- Vectorized **forward** (linear → sigmoid → linear → softmax) & **backward** with cross-entropy
- Deterministic **SGD** (no shuffling) + two inits: `zero` and `random`
- **CLI tools** for training and prediction
- Saves **weights** to `.npz` and **metrics** to CSV
- Tests, **GitHub Actions CI**, pre-commit

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .[dev]
pre-commit install
```

### Train (CSV with first column = label, remaining 128 features)
```bash
python scripts/train.py   --train data/smallTrain.csv   --valid data/smallValidation.csv   --hidden 64 --epochs 15 --lr 0.1 --init random   --out runs/exp1
```

### Predict
```bash
python scripts/predict.py   --model runs/exp1/model.npz   --data data/smallValidation.csv   --out runs/exp1/valid_pred.labels
```

`predict.py` prints accuracy when labels are present in the CSV.

## Data format
- CSV rows: `[label, f1, f2, ..., f128]` (16×8 flattened)
- Labels 0..9 map to letters: `["a","e","g","i","l","n","o","r","t","u"]`

> Tip: Place datasets under `data/`. Add your own or use scripts to create toy data.

## Notes
- Model uses bias via augmented inputs (x0=1, z0=1). Weights: `alpha` (D×(M+1)) and `beta` (K×(D+1)).
- This implementation follows the classic intro ML neural-net spec for OCR-style letters. (See your course PDF.)

## License
MIT © 2025 Dhamodar Burla
