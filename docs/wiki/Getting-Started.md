# Getting Started

This page takes you from a fresh clone to a trained model and a first prediction.

## 1. Choose where to run

| Option | Good for | What you need |
| --- | --- | --- |
| **Google Colab** (recommended if you have no GPU) | A full training run for free | A Google account |
| **Your own machine with an NVIDIA GPU** | Repeated experiments | CUDA-capable GPU, Python 3.7+ |
| **CPU only** | Prediction, the web demo, the tests | Python 3.7+; training works but is very slow |

### Option A: Google Colab

1. Open [`colab_train.ipynb` in Colab](https://colab.research.google.com/github/thomasmd321/Image-Classifier/blob/main/colab_train.ipynb).
2. *Runtime → Change runtime type → Hardware accelerator: GPU (T4)*.
3. Optional: set `USE_DRIVE = True` in the second cell to keep checkpoints on Google Drive. If Colab disconnects,
   *Run all* again and training resumes where it stopped.
4. *Runtime → Run all*. It takes roughly 30–60 minutes and ends by downloading a zip with the checkpoint,
   learning curves and evaluation report.

That's it; skip to [what to do next](#what-next).

### Option B: Your own machine

```bash
git clone https://github.com/thomasmd321/Image-Classifier.git
cd Image-Classifier
pip install -r requirements.txt            # add -c constraints.txt for the exact tested versions
python download_data.py                    # 345 MB, unpacks into flowers/train, valid, test
```

## 2. Train

```bash
python train.py flowers --gpu --epochs 20 --finetune_epochs 5 --seed 42 --save_dir checkpoints
```

- Phase 1 trains a new classifier on top of a pretrained DenseNet-121 (up to 20 epochs, stopping early when
  validation accuracy stalls).
- Phase 2 fine-tunes the last block of DenseNet as well (5 epochs, lower learning rate).
- The best model is saved to `checkpoints/check_point.pt`, and its test accuracy is printed at the end.

The [Training Guide](Training-Guide.md) explains every option.

## 3. Predict

```bash
python predict.py flowers/test/10/image_07104.jpg checkpoints/check_point.pt \
    --category_names cat_to_name.json --top_k 5 --gpu
```

```
Top 5 predictions for flowers/test/10/image_07104.jpg:
 1. globe thistle                   97.10 %
 2. ...
```

_(The numbers are an illustration; yours will depend on the trained model.)_

## What next

- See where the model makes mistakes: [Evaluation and Results](Evaluation-and-Results.md).
- Try it in the browser: `pip install -r requirements-demo.txt && python app.py --checkpoint checkpoints/check_point.pt`
  (see [Prediction and Demo](Prediction-and-Demo.md)).
- Compare architectures with `--arch efficientnet_b0` or `--arch convnext_tiny`.

## Using your own images

Any folder laid out like this works in place of `flowers`:

```
my_data/
  train/<class name>/*.jpg
  valid/<class name>/*.jpg
  test/<class name>/*.jpg
```

Class folder names become the labels. To show friendly names, pass a JSON file mapping folder names to names with
`--category_names` (like `cat_to_name.json`).
