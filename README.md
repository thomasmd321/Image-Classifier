# AI Programming with Python Project

[![tests](https://github.com/thomasmd321/Image-Classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/thomasmd321/Image-Classifier/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thomasmd321/Image-Classifier/blob/main/colab_train.ipynb)

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Files

| File | Purpose |
| --- | --- |
| `colab_train.ipynb` | Trains and evaluates the model on a free Google Colab GPU in one click |
| `download_data.py` | Downloads the flowers dataset into `flowers/` |
| `Image Classifier Project.ipynb` | Notebook version of the project: load data, train (with fine-tuning), plot learning curves, test, save/load a checkpoint, predict and plot |
| `train.py` | Command line app that trains a classifier on a dataset and saves checkpoints and a training history |
| `predict.py` | Command line app that predicts the class of an image (or a folder of images) from a checkpoint |
| `evaluate.py` | Per-class accuracy and most-confused flower pairs for a checkpoint |
| `export.py` | Exports a checkpoint to TorchScript and/or ONNX so it runs without this code |
| `app.py` | Web demo: upload a flower photo and see the top 5 predictions |
| `publish_to_hub.py` | Publishes a trained model to the Hugging Face Hub with a generated model card |
| `Dockerfile` | Container image for the web demo |
| `model_utils.py` | Shared code: model building, checkpoint save/load, image preprocessing, prediction |
| `workspace_utils.py` | Keeps the Udacity workspace alive during long runs (`train.py --keep_alive`) |
| `cat_to_name.json` | Maps category labels to flower names |
| `tests/` | Smoke tests (random weights, tiny synthetic dataset) run by GitHub Actions with a `ruff` lint check |

## Requirements

Python 3.7+ and PyTorch 1.10+. Install the dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` allows a range of versions (so, for example, Colab keeps its preinstalled PyTorch).
`constraints.txt` lists the exact versions the project is tested with; add `-c constraints.txt` to install
exactly those, as CI and the Docker image do. When updating a dependency, change it there and let CI confirm it.

Download the [flowers dataset](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)
(345 MB; 6,552 training, 818 validation and 819 test images of 102 species) into `flowers/`:

```bash
python download_data.py
```

To train on your own images, use a folder with `train/`, `valid/` and `test/` sub-folders, each with one
sub-folder per class.

## No GPU? Use Google Colab

[Open `colab_train.ipynb` in Colab](https://colab.research.google.com/github/thomasmd321/Image-Classifier/blob/main/colab_train.ipynb),
choose *Runtime -> Change runtime type -> GPU*, then *Runtime -> Run all*. It downloads the data, trains with
fine-tuning, shows the learning curves and evaluation charts, and downloads a zip with the checkpoint and reports.
Turn on its Google Drive option to keep checkpoints across disconnects; training resumes where it stopped.

## Training

```bash
python train.py flowers
python train.py flowers --arch densenet121 --epochs 20 --finetune_epochs 5 --gpu --seed 42 --save_dir checkpoints
```

Training runs in two phases:

1. **Classifier** (`--epochs`): the pretrained feature extractor is frozen and only the new classifier is trained.
2. **Fine-tuning** (`--finetune_epochs`, off by default): starting from the best classifier, the last block of the
   feature extractor is unfrozen too and trained at a lower learning rate. This usually adds several points of
   accuracy.

| Option | Default | Description |
| --- | --- | --- |
| `--save_dir` | `.` | Where checkpoints and the training history are written |
| `--arch` | `densenet121` | `densenet121`, `efficientnet_b0`, `convnext_tiny`, `resnet50`, `vgg16` or `alexnet` (see below) |
| `--hidden_units` | `512 90 80` | Size of each hidden layer of the classifier, e.g. `--hidden_units 512` or `--hidden_units 1024 256` |
| `--learning_rate` | `0.001` | Adam learning rate for the classifier phase; reduced 10x when validation loss stops improving for 2 epochs |
| `--epochs` | `20` | Maximum epochs for the classifier phase |
| `--finetune_epochs` | `0` | Epochs to fine-tune the last feature-extractor block afterwards (`0` = off) |
| `--finetune_lr` | `0.0001` | Learning rate for the fine-tuning phase |
| `--patience` | `5` | Stop a phase early after this many epochs without a validation accuracy improvement (`0` disables) |
| `--dropout` | `0.5` | Classifier dropout |
| `--label_smoothing` | `0.1` | Label smoothing on the training loss (`0` disables) |
| `--batch_size` | `20` | Images per batch |
| `--num_workers` | `4` | Data loading processes |
| `--print_every` | `50` | Print the training loss every N batches |
| `--seed` | none | Random seed for reproducible runs |
| `--resume` | none | Continue an interrupted run from `last_checkpoint.pt` (works in either phase) |
| `--gpu` | off | Train on the GPU (uses mixed precision automatically) |
| `--no_amp` | off | Turn off mixed precision on the GPU |
| `--tensorboard` | off | Also log metrics for TensorBoard to `<save_dir>/runs` (`pip install tensorboard`) |
| `--keep_alive` | off | Keep the Udacity workspace awake during long runs |

**Choosing an architecture.** `densenet121` is the project's original choice. `efficientnet_b0` is usually as
accurate or better while being smaller and faster, and `convnext_tiny` is usually the most accurate but is slower.
`resnet50`, `vgg16` and `alexnet` are older networks, kept for comparison. Compare them with the same `--seed`
and fill in the results table below.

Training images are augmented with random rotation, crops, flips and small color changes. Training prints the
training loss, then the validation loss and accuracy after every epoch, and writes to `--save_dir`:

- `check_point.pt`: the model with the best validation accuracy so far. This is the one to use for prediction,
  and the one the final test accuracy is measured on.
- `last_checkpoint.pt`: the latest epoch, including the optimizer and learning-rate scheduler state, for `--resume`.
- `history.csv` and `history.png`: loss, validation accuracy and learning rate for every epoch, and a chart of
  the learning curves.

To follow training live in TensorBoard, add `--tensorboard` and run `tensorboard --logdir <save_dir>/runs`.

To resume after an interruption, rerun with the same data and `--resume`; the architecture and hidden units come
from the checkpoint:

```bash
python train.py flowers --gpu --save_dir checkpoints --finetune_epochs 5 --resume checkpoints/last_checkpoint.pt
```

## Prediction

```bash
python predict.py flowers/test/10/image_07104.jpg check_point.pt
python predict.py flowers/test/10/image_07104.jpg check_point.pt --top_k 5 --category_names cat_to_name.json --gpu
python predict.py flowers/test/10 check_point.pt --category_names cat_to_name.json --plot_dir plots
```

`image_path` can be a single image or a folder of images. `--plot_dir` saves each image with a bar chart of its
top predictions. `--tta` (test-time augmentation) averages the predictions for the image and its mirror image,
which is usually slightly more accurate. A checkpoint trained on a GPU can be used for prediction on a CPU-only machine.

## Evaluation report

```bash
python evaluate.py flowers check_point.pt --category_names cat_to_name.json --gpu
```

Prints the overall accuracy, the classes the model gets wrong most often and the most common mistakes
(true flower -> predicted flower), and writes to `--output_dir`:

- `per_class_accuracy.csv` and `confused_pairs.csv`
- `confusion_matrix.png`: which flowers get mistaken for which (correct predictions are left out so the
  mistakes stand out)
- `misclassified.png`: the model's most confident mistakes, with the true and predicted names (`--gallery` sets
  how many)

Use `--tta` for test-time augmentation and `--split valid` to evaluate the validation images instead.

## Web demo

```bash
pip install -r requirements-demo.txt
python app.py --checkpoint check_point.pt
```

Open the printed local address, upload a flower photo and the demo shows the top 5 predictions. `--share` creates a
temporary public link. To host it for free on [Hugging Face Spaces](https://huggingface.co/spaces), publish the
model (see below), create a Gradio Space and upload `app.py`, `model_utils.py`, `cat_to_name.json` and
`requirements-demo.txt` (renamed to `requirements.txt`, with the `-r` line replaced by the contents of this
project's `requirements.txt`), then set the Space variable `HUB_REPO` to your model repository.

## Docker

The `Dockerfile` packages the web demo (CPU only):

```bash
docker build -t flower-classifier .
docker run -p 7860:7860 -v "$PWD/check_point.pt:/models/check_point.pt:ro" flower-classifier
# or, with a model published to the Hugging Face Hub:
docker run -p 7860:7860 -e HUB_REPO=your-name/flower-classifier flower-classifier
```

Then open <http://localhost:7860>.

## Publish to the Hugging Face Hub

```bash
pip install huggingface_hub && huggingface-cli login
python evaluate.py flowers checkpoints/check_point.pt --category_names cat_to_name.json --output_dir report
python publish_to_hub.py checkpoints/check_point.pt --repo_id your-name/flower-classifier --report_dir report
```

This uploads the checkpoint, the class labels, the learning curves and evaluation charts, and a generated model
card with the accuracy, usage example and limitations. Add `--dry_run preview` to write the files locally and
review the model card first. The web demo can then load the model directly:
`python app.py --hub_repo your-name/flower-classifier`.

## Export

```bash
python export.py check_point.pt --format both --category_names cat_to_name.json --output_dir exported
```

Writes `flower_classifier.torchscript.pt` (load with `torch.jit.load`), `flower_classifier.onnx` (run with
[ONNX Runtime](https://onnxruntime.ai/) from Python, C#, Java, JavaScript, mobile, ...) and
`flower_classifier.labels.json` with the class labels, flower names and the preprocessing to apply (resize to 256,
center-crop 224, normalize). The exported models take a batch of preprocessed images and return class
probabilities. ONNX export needs `pip install onnx`. Recent PyTorch versions mark TorchScript as deprecated, but
it still works; ONNX is the more portable choice.

## Results

_Fill in after a full training run on the flowers dataset (numbers from the training output, `history.csv` and
`evaluate.py`):_

| Architecture | Classifier epochs | Fine-tuning epochs | Best validation accuracy | Test accuracy | Test accuracy (TTA) | Training time (GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| densenet121 | | | | | | |
| efficientnet_b0 | | | | | | |
| convnext_tiny | | | | | | |

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests use random weights and a tiny generated dataset, so they run in under two minutes on a CPU without
downloading anything. GitHub Actions also builds the Docker image and checks that the demo starts in it. They check that the scripts run end to end, not model accuracy.
