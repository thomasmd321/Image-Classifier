# Evaluation and Results

Training prints a single test accuracy. `evaluate.py` goes further: it shows *which* flowers the model gets
wrong and what it mistakes them for.

## Running an evaluation

```bash
python evaluate.py flowers checkpoints/check_point.pt --category_names cat_to_name.json --gpu --output_dir report
```

| Option | Default | What it does |
| --- | --- | --- |
| `--split` | `test` | Which images to evaluate: `test`, `valid` or `train` |
| `--tta` | off | Test-time augmentation (see below) |
| `--category_names` | none | JSON mapping class folders to flower names |
| `--output_dir` | `.` | Where the report files go |
| `--top` | `10` | How many worst classes / confused pairs to print |
| `--gallery` | `16` | How many images to show in `misclassified.png` (`0` = none) |
| `--batch_size`, `--num_workers`, `--gpu` | | As in training |

## The report

| File | What it shows |
| --- | --- |
| `summary.json` | The split evaluated, overall accuracy, whether TTA was used, image and class counts |
| `per_class_accuracy.csv` | Accuracy for every flower, worst first |
| `confused_pairs.csv` | Every kind of mistake (true flower → predicted flower), most common first |
| `confusion_matrix.png` | A heatmap of the mistakes |
| `misclassified.png` | The model's most confident wrong answers, with photos |

### Reading the confusion matrix

Rows are the true flower and columns are the predicted flower. Correct predictions (the diagonal) are left out, so
only mistakes are drawn; darker cells mean more images. Look for:

- **Isolated dark cells:** two species the model confuses with each other; check whether they genuinely look alike.
- **A dark row:** a flower the model struggles with in general (few training images, or very varied photos).
- **A dark column:** a "catch-all" species the model guesses too often.

### Reading the mistakes gallery

These are the mistakes the model was *most sure* about. They are the most informative images in the project.
Common causes:

- Several species in one photo.
- An unusual angle, or only leaves or buds visible.
- A photo that is actually labelled wrongly in the dataset.
- Two species that really are nearly identical.

## Test-time augmentation (TTA)

With `--tta`, each image is classified twice, as-is and mirrored left-to-right, and the two sets of probabilities
are averaged. It costs twice the prediction time and usually adds a little accuracy, often around 0.5–1 point.
`predict.py --tta` does the same for single images.

## Which accuracy to report

- **Test accuracy** (from `train.py`'s last line, or `evaluate.py --split test`) is the honest number: those
  images were never used to make any training decision.
- **Validation accuracy** is used to pick the best epoch, so it is slightly optimistic. Report it only as a
  secondary number.

## Recording results

Fill in the table in the project README after each full run:

| Architecture | Classifier epochs | Fine-tuning epochs | Best validation accuracy | Test accuracy | Test accuracy (TTA) | Training time (GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| densenet121 | | | | | | |

- **Epochs:** the number of rows per phase in `history.csv`.
- **Best validation accuracy:** the highest `valid_accuracy` in `history.csv`.
- **Test accuracy:** the last line of the training output, or `evaluate.py`'s `summary.json`.
- **TTA column:** `evaluate.py --tta`.
