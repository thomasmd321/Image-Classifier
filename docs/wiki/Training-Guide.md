# Training Guide

`train.py` trains a flower classifier on top of a network pretrained on ImageNet, a technique called
*transfer learning*. This page explains how a run works, what every option does, and how to choose settings.

## How a training run works

1. **Load the data.** `train/`, `valid/` and `test/` are read with `torchvision.datasets.ImageFolder`. Training
   images get random augmentation: rotation up to 30°, random crop and zoom, horizontal flip, and small brightness,
   contrast and saturation changes. Validation and test images are only resized (shortest side 256) and
   center-cropped to 224×224.
2. **Phase 1: classifier** (`--epochs`). The pretrained network is frozen and a new classifier is trained on its
   features. This is fast, and it learns most of what the model will know.
3. **Phase 2: fine-tuning** (`--finetune_epochs`, off by default). Starting from the best phase-1 model, the *last
   block* of the pretrained network is unfrozen too and trained at a much lower learning rate, adapting its
   high-level features to flowers.
4. **After every epoch**, the model is scored on the validation images:
   - If the accuracy is the best so far, the model is saved to `check_point.pt`.
   - The learning rate is cut 10× if the validation loss hasn't improved for 2 epochs.
   - The phase stops early if accuracy hasn't improved for `--patience` epochs.
5. **At the end**, the best model is scored on the test images, which training never looked at, and the learning
   curves are saved.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `data_dir` | (required) | Folder with `train/`, `valid/` and `test/` |
| `--save_dir` | `.` | Where checkpoints and history are written |
| `--arch` | `densenet121` | Pretrained network; see [Choosing an architecture](#choosing-an-architecture) |
| `--hidden_units` | `512 90 80` | Size of each hidden layer of the new classifier, e.g. `--hidden_units 512` or `--hidden_units 1024 256` |
| `--dropout` | `0.5` | Dropout before the classifier's first layer |
| `--learning_rate` | `0.001` | Adam learning rate in phase 1 |
| `--epochs` | `20` | Maximum epochs in phase 1 |
| `--finetune_epochs` | `0` | Epochs in phase 2 (`0` = no fine-tuning) |
| `--finetune_lr` | `0.0001` | Learning rate in phase 2 |
| `--patience` | `5` | End a phase after this many epochs without a validation accuracy improvement (`0` = never stop early) |
| `--label_smoothing` | `0.1` | Softens the training targets so the model is less over-confident (`0` = off) |
| `--batch_size` | `20` | Images per training step; lower it if you run out of GPU memory |
| `--num_workers` | `4` | Processes loading images in parallel (Colab: use 2) |
| `--print_every` | `50` | Print the training loss every N batches |
| `--seed` | none | Makes runs reproducible; use the same seed to compare settings fairly |
| `--gpu` | off | Train on the GPU (exits if no GPU is found) |
| `--no_amp` | off | Turn off mixed precision on the GPU |
| `--resume` | none | Continue from a `last_checkpoint.pt` |
| `--tensorboard` | off | Log metrics for TensorBoard |
| `--keep_alive` | off | Keep a Udacity workspace awake (only works there) |

## Choosing an architecture

| `--arch` | Notes |
| --- | --- |
| `densenet121` | The project's original choice. A solid, well-understood baseline. |
| `efficientnet_b0` | Usually as accurate or better, and smaller and faster. A good first alternative. |
| `convnext_tiny` | A modern network; usually the most accurate here, but slower and needs more GPU memory. |
| `resnet50` | Classic baseline. |
| `vgg16`, `alexnet` | Older and much less accurate per unit of compute; kept for comparison. |

These are general expectations, not measurements from this project. Compare them yourself by training each with
the same `--seed` and recording the results (see [Evaluation and Results](Evaluation-and-Results.md#recording-results)).

## Recommended settings

- **A good default:** `--epochs 20 --finetune_epochs 5 --seed 42`. Fine-tuning usually adds several points of
  accuracy over training the classifier alone.
- **Out of GPU memory:** lower `--batch_size` (e.g. 16 or 8).
- **Validation accuracy is much lower than training accuracy (overfitting):** raise `--dropout`, keep label
  smoothing on, or use fewer hidden units (e.g. `--hidden_units 256`).
- **Accuracy still rising when a phase ends:** raise `--epochs` or `--patience`.
- **Change one thing at a time** with a fixed `--seed`, so differences come from your change, not chance.

## Output files

Everything goes into `--save_dir`:

| File | Contents |
| --- | --- |
| `check_point.pt` | The model with the best validation accuracy. Use this one for prediction, evaluation, the demo and publishing. |
| `last_checkpoint.pt` | The latest epoch, plus the optimizer and learning-rate scheduler state, for `--resume`. |
| `history.csv` | One row per epoch: phase, training loss, validation loss and accuracy, learning rate, seconds. |
| `history.png` | Learning curves (loss and validation accuracy); a dashed line marks where fine-tuning starts. |
| `runs/` | TensorBoard logs (with `--tensorboard`). |

Starting a new run (without `--resume`) in the same `--save_dir` replaces the previous run's checkpoints and
history, and prints a message when it does.

## Resuming an interrupted run

```bash
python train.py flowers --gpu --epochs 20 --finetune_epochs 5 --save_dir checkpoints \
    --resume checkpoints/last_checkpoint.pt
```

- The architecture, hidden layers and dropout come from the checkpoint; the other options come from the command
  line, so pass the same `--epochs` / `--finetune_epochs` as before.
- Training continues in whichever phase it was in. If a phase had already stopped early, it moves on to the next one.
- The Colab notebook resumes automatically when its Google Drive option is on.

## Watching training live

```bash
python train.py flowers --gpu --tensorboard --save_dir checkpoints
tensorboard --logdir checkpoints/runs        # then open http://localhost:6006
```

Or just open `history.png` after the run.

## Mixed precision

On a GPU, the forward pass runs in 16-bit floats where it is safe to, which is usually 1.5–2× faster. Loss scaling
keeps small gradients from being lost. It is on by default with `--gpu`; `--no_amp` turns it off if you suspect
it is causing a problem.
