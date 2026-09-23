# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Files

| File | Purpose |
| --- | --- |
| `Image Classifier Project.ipynb` | Notebook version of the project: load data, train, test, save/load a checkpoint, predict and plot |
| `train.py` | Command line app that trains a classifier on a dataset and saves a checkpoint |
| `predict.py` | Command line app that predicts the class of an image from a saved checkpoint |
| `model_utils.py` | Shared code: model building, checkpoint save/load, image preprocessing, prediction |
| `workspace_utils.py` | Keeps the Udacity workspace alive during long runs (`train.py --keep_alive`) |
| `cat_to_name.json` | Maps category labels to flower names |
| `tests/` | Quick smoke tests (random weights, tiny synthetic dataset) run by GitHub Actions |

## Requirements

Python 3.7+. Install the dependencies with:

```bash
pip install -r requirements.txt
```

The data directory must contain `train/`, `valid/` and `test/` folders, each with one sub-folder per class
(the [flowers dataset](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) uses this layout).

## Training

```bash
python train.py flowers
python train.py flowers --arch densenet121 --learning_rate 0.001 --hidden_units 512 --epochs 20 --gpu --save_dir checkpoints
```

| Option | Default | Description |
| --- | --- | --- |
| `--save_dir` | `.` | Where checkpoints are written |
| `--arch` | `densenet121` | `densenet121`, `alexnet`, `vgg16` or `resnet50` |
| `--learning_rate` | `0.001` | Adam learning rate; reduced 10x when validation loss stops improving for 2 epochs |
| `--hidden_units` | `512` | Size of the first hidden layer of the classifier |
| `--epochs` | `20` | Maximum number of epochs |
| `--patience` | `5` | Stop early after this many epochs without a validation accuracy improvement (`0` disables) |
| `--dropout` | `0.5` | Classifier dropout |
| `--batch_size` | `20` | Images per batch |
| `--num_workers` | `4` | Data loading processes |
| `--print_every` | `50` | Print the training loss every N batches |
| `--seed` | none | Random seed for reproducible runs |
| `--resume` | none | Continue an interrupted run from `last_checkpoint.pt` |
| `--gpu` | off | Train on the GPU |
| `--keep_alive` | off | Keep the Udacity workspace awake during long runs |

Training prints the training loss, then the validation loss and accuracy after every epoch. Two checkpoints are
written to `--save_dir`:

- `check_point.pt`: the model with the best validation accuracy so far. This is the one to use for prediction,
  and the one the final test accuracy is measured on.
- `last_checkpoint.pt`: the latest epoch, including the optimizer and learning-rate scheduler state, for `--resume`.

To resume after an interruption, rerun with the same data and `--resume`; the architecture and hidden units come
from the checkpoint:

```bash
python train.py flowers --gpu --save_dir checkpoints --resume checkpoints/last_checkpoint.pt
```

## Prediction

```bash
python predict.py flowers/test/10/image_07104.jpg check_point.pt
python predict.py flowers/test/10/image_07104.jpg check_point.pt --top_k 5 --category_names cat_to_name.json --gpu
python predict.py flowers/test/10 check_point.pt --category_names cat_to_name.json --plot_dir plots
```

`image_path` can be a single image or a folder of images. `--plot_dir` saves each image with a bar chart of its
top predictions. A checkpoint trained on a GPU can be used for prediction on a CPU-only machine.

## Results

_Fill in after a full training run on the flowers dataset:_

| Architecture | Epochs | Best validation accuracy | Test accuracy | Training time (GPU) |
| --- | --- | --- | --- | --- |
| densenet121 | | | | |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use random weights and a tiny generated dataset, so they run in under a minute on a CPU without
downloading anything. They check that the scripts run end to end, not model accuracy.
