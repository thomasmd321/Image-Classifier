# Project History

## Origin

The project began in 2020 as the final project of Udacity's *AI Programming with Python* Nanodegree: build a
flower classifier in a notebook, then turn it into two command-line apps, `train.py` and `predict.py`. That
original submission is preserved unchanged on the
[`original-code`](https://github.com/thomasmd321/Image-Classifier/tree/original-code) branch.

## Improvements

| Pull request | What changed |
| --- | --- |
| [#1](https://github.com/thomasmd321/Image-Classifier/pull/1) Bug fixes and refactor | Fixed a crash when training on the CPU. Only one layer had been frozen instead of the whole pretrained network. Fixed dropout left off after validation, a wrong validation loss, `--save_dir` being ignored, GPU checkpoints failing to load on a CPU, and random predictions (no `eval()`). Moved shared code to `model_utils.py`, made `predict.py` match the rubric's command line, and completed the notebook. |
| [#2](https://github.com/thomasmd321/Image-Classifier/pull/2) Training features, tests, CI | Best-checkpoint saving, early stopping, learning-rate schedule, resume, seeds, faster data loading, `vgg16`/`resnet50`, predicting folders and plots, `requirements.txt`, the first tests and GitHub Actions. |
| [#3](https://github.com/thomasmd321/Image-Classifier/pull/3) Accuracy and tooling | Fine-tuning phase, configurable hidden layers, color augmentation, label smoothing, mixed precision, training history and learning curves, `evaluate.py`, `export.py`, the Gradio demo, lint in CI, and a synced notebook. |
| [#4](https://github.com/thomasmd321/Image-Classifier/pull/4) Getting and sharing results | Colab notebook, `download_data.py`, `efficientnet_b0`/`convnext_tiny`, test-time augmentation, confusion matrix and mistakes gallery, TensorBoard, Hugging Face publishing, Dockerfile. |
| [#5](https://github.com/thomasmd321/Image-Classifier/pull/5) Review fixes and pinning | A code review found 10 issues, all fixed with regression tests. The most important: frozen layers are now kept in eval mode, so pretrained BatchNorm statistics no longer drift. Also fixed resume edge cases and stale checkpoints, and pinned dependency versions (`constraints.txt`). |

## Still to do

- A full training run on a GPU, to record real accuracy numbers in the README and on [Home](Home.md).
- Commit the course notebook with its outputs for submission.
- Optionally, publish the trained model and web demo on Hugging Face.
