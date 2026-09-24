# Exporting and Publishing

## Exporting the model: `export.py`

Exported models run without this project's code, e.g. in another Python program, a mobile app, or a web
service written in another language.

```bash
python export.py check_point.pt --format both --category_names cat_to_name.json --output_dir exported
```

| Output | Use it with |
| --- | --- |
| `flower_classifier.torchscript.pt` | `torch.jit.load(...)` in Python or C++ (LibTorch). The class labels are embedded as `class_to_idx.json`. |
| `flower_classifier.onnx` | [ONNX Runtime](https://onnxruntime.ai/) from Python, C#, Java, JavaScript, iOS, Android, … |
| `flower_classifier.labels.json` | Class labels, flower names (with `--category_names`) and the preprocessing to apply |

- The exported models take a batch of preprocessed images (N×3×224×224) and return **class probabilities**, not
  log-probabilities.
- The preprocessing, also recorded in the labels file:
  1. Resize so the shortest side is 256 pixels.
  2. Center-crop 224×224.
  3. Scale pixel values to 0–1.
  4. Normalize with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- ONNX export needs `pip install onnx`.
- Recent PyTorch versions mark TorchScript as deprecated. It still works, but ONNX is the more future-proof choice.

Example with ONNX Runtime:

```python
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

labels = json.load(open('exported/flower_classifier.labels.json'))
session = ort.InferenceSession('exported/flower_classifier.onnx')

image = Image.open('photo.jpg').convert('RGB')
scale = 256 / min(image.size)
image = image.resize((round(image.width * scale), round(image.height * scale)))
left, top = (image.width - 224) // 2, (image.height - 224) // 2
image = image.crop((left, top, left + 224, top + 224))
x = (np.asarray(image, dtype=np.float32) / 255 - labels['input']['mean']) / labels['input']['std']
x = x.transpose(2, 0, 1)[None].astype(np.float32)

(probs,) = session.run(None, {'image': x})
best = int(probs[0].argmax())
print(labels.get('names', labels['labels'])[best], float(probs[0, best]))
```

## Publishing to the Hugging Face Hub

`publish_to_hub.py` uploads a trained model with a generated **model card** (the repository's README on the Hub).

```bash
pip install huggingface_hub
huggingface-cli login                                  # once; paste a token with write access
python evaluate.py flowers checkpoints/check_point.pt --category_names cat_to_name.json --output_dir report
python publish_to_hub.py checkpoints/check_point.pt --repo_id your-name/flower-classifier --report_dir report
```

What gets uploaded:

- `check_point.pt` and `labels.json`.
- `history.png` / `history.csv`, if they are next to the checkpoint (or set `--history_dir`).
- The evaluation report and charts from `--report_dir`.
- A model card with:
  - Hub metadata and the accuracy, labelled with the split `evaluate.py` actually used (e.g. "Test accuracy (TTA)").
  - A results table, the charts and a usage example.
  - The model's limitations.

| Option | What it does |
| --- | --- |
| `--repo_id` | Target repository, `your-name/model-name` |
| `--report_dir` | Folder with `evaluate.py`'s output |
| `--history_dir` | Folder with `history.png/csv` (default: the checkpoint's folder) |
| `--category_names` | Flower names for `labels.json` (default `cat_to_name.json`) |
| `--private` | Create a private repository |
| `--dry_run folder` | Write everything to `folder` instead of uploading, so you can review the model card first |

**Preview before publishing.** Run with `--dry_run preview` and read `preview/README.md` first.

**Using a published model:**

```python
from huggingface_hub import hf_hub_download
from model_utils import load_checkpoint, predict

model = load_checkpoint(hf_hub_download('your-name/flower-classifier', 'check_point.pt'))
print(predict('photo.jpg', model, topk=5))
```

or `python app.py --hub_repo your-name/flower-classifier`.
