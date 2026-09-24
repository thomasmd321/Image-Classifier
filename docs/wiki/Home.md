# Flower Image Classifier – Wiki

A PyTorch image classifier that recognizes **102 species of flowers**. It started as the final project of
Udacity's *AI Programming with Python* Nanodegree and grew into a small, tested toolkit. It can train on a GPU
(or free on Google Colab), evaluate where the model goes wrong, predict from the command line or a web demo,
and export or publish the trained model.

## Where to start

| I want to… | Read |
| --- | --- |
| Train a model and make my first prediction | [Getting Started](Getting-Started.md) |
| Understand every training option and pick good settings | [Training Guide](Training-Guide.md) |
| Measure accuracy and see which flowers get confused | [Evaluation and Results](Evaluation-and-Results.md) |
| Classify my own photos, or run the web demo | [Prediction and Demo](Prediction-and-Demo.md) |
| Use the model outside this project, or share it | [Exporting and Publishing](Exporting-and-Publishing.md) |
| Understand how the model and checkpoints work | [How It Works](How-It-Works.md) |
| Change the code, run the tests, or add an architecture | [Development](Development.md) |
| Fix an error message | [Troubleshooting and FAQ](Troubleshooting-and-FAQ.md) |
| See how the project got here | [Project History](Project-History.md) |

## The workflow at a glance

```
download_data.py ─► train.py ─► check_point.pt ─┬─► evaluate.py   (accuracy, charts)
                     │                          ├─► predict.py    (command line)
                     ├─► history.csv/.png       ├─► app.py        (web demo, Docker)
                     └─► last_checkpoint.pt     ├─► export.py     (TorchScript / ONNX)
                         (for --resume)         └─► publish_to_hub.py (Hugging Face)
```

No GPU? [`colab_train.ipynb`](https://colab.research.google.com/github/thomasmd321/Image-Classifier/blob/main/colab_train.ipynb)
runs the whole workflow on a free Colab GPU.

## Results

_To be filled in after a full training run. See [Evaluation and Results](Evaluation-and-Results.md#recording-results)._
