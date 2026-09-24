# Development

## Setup

```bash
pip install -r requirements-dev.txt -c constraints.txt    # tests, lint, ONNX, TensorBoard, Hub (exact versions)
pip install -r requirements-demo.txt -c constraints.txt   # if you work on app.py
```

## Tests and lint

```bash
ruff check .        # lint; `ruff check . --fix` fixes import order and similar issues
pytest              # about 2 minutes on a CPU
```

The tests never download anything. `tests/conftest.py`:

- swaps pretrained weights for random ones;
- builds a tiny dataset of random images with three classes (`1`, `10`, `2`) in `train/valid/test`;
- trains one small model that many tests share.

The tests check that everything runs and behaves correctly, not how accurate the model is.

| File | Covers |
| --- | --- |
| `tests/test_smoke.py` | Every architecture builds and freezes correctly; train → predict end to end; folders, plots, odd image formats; resume; seeds; early stopping |
| `tests/test_features.py` | Hidden layers, old checkpoints, label smoothing, fine-tuning, history, evaluation, export, demo |
| `tests/test_extras.py` | Dataset download, TTA, evaluation charts, TensorBoard, Hub publishing, `--hub_repo`, the Colab notebook's options |
| `tests/test_review_fixes.py` | Regression tests for bugs found in code review (frozen BatchNorm, resume edge cases, stale checkpoints, …) |

## Continuous integration

`.github/workflows/tests.yml` runs on every pull request and every push to `main`:

- **test**: installs the pinned versions (CPU PyTorch), runs `ruff check .` and `pytest`.
- **docker**: builds the `Dockerfile`, starts the demo with an untrained checkpoint, and checks the page is served.

## Dependencies

- `requirements*.txt` give version *ranges*, capped below the next major version, so users (and Colab, with its
  preinstalled PyTorch) aren't forced onto exact versions.
- `constraints.txt` pins the exact versions CI and Docker use. To upgrade a dependency:
  1. Change it in `constraints.txt`, and in the `torch==` / `torchvision==` lines of the workflow and
     `Dockerfile` for PyTorch.
  2. Open a pull request and let CI confirm it.

## Adding an architecture

1. Add an entry to `ARCHS` in `model_utils.py`:
   ```python
   "mobilenet_v3_large": {"in_features": 960, "head": "classifier",
                          "last_block": ["features.16"]},
   ```
   - `in_features`: the number of inputs to the layer you replace.
   - `head`: the attribute path of the layer to replace (dotted paths like `classifier.2` work).
   - `last_block`: the modules to unfreeze for fine-tuning.
2. Check it: `pytest tests/test_smoke.py -k build_model` builds the network and checks the output shape, and that
   only the head is trainable.
3. Add it to the table in [How It Works](How-It-Works.md#transfer-learning) and to the README.

(The MobileNet entry is an example only; check a network's layers with `print(model)` before adding it.)

## Conventions

- Scripts keep their argument parsing in `get_args(argv=None)` and their work in `main(argv=None)`, so tests
  can call `main([...])` directly.
- Shared logic belongs in `model_utils.py`, not duplicated across scripts.
- Every bug fix comes with a test that fails without the fix.
