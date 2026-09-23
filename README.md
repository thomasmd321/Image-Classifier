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

## Requirements

Python 3.7+, `torch`, `torchvision`, `Pillow`, plus `numpy` and `matplotlib` for the notebook.

The data directory must contain `train/`, `valid/` and `test/` folders, each with one sub-folder per class
(the [flowers dataset](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) uses this layout).

## Training

```bash
python train.py flowers
python train.py flowers --arch densenet121 --learning_rate 0.001 --hidden_units 512 --epochs 20 --gpu --save_dir checkpoints
```

Options: `--save_dir`, `--arch` (`densenet121` or `alexnet`), `--learning_rate`, `--hidden_units`, `--epochs`,
`--dropout`, `--batch_size`, `--gpu`, `--keep_alive`. Training prints the training loss, validation loss and
validation accuracy, then the test accuracy, and saves `check_point.pt` into `--save_dir`.

## Prediction

```bash
python predict.py flowers/test/10/image_07104.jpg check_point.pt
python predict.py flowers/test/10/image_07104.jpg check_point.pt --top_k 5 --category_names cat_to_name.json --gpu
```

A checkpoint trained on a GPU can be used for prediction on a CPU-only machine.
