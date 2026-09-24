# Prediction and Demo

Three ways to classify photos with a trained `check_point.pt`: the command line, a web page, or a Docker
container.

## Command line: `predict.py`

```bash
python predict.py path/to/photo.jpg check_point.pt --category_names cat_to_name.json --top_k 5
```

| Option | Default | What it does |
| --- | --- | --- |
| `image_path` | (required) | One image, or a folder of images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tif`, `.tiff`, `.webp`) |
| `checkpoint` | (required) | A checkpoint from `train.py` |
| `--top_k` | `3` | How many predictions to show (capped at the number of classes) |
| `--category_names` | none | JSON mapping class folders to flower names; without it you see folder names like `10` |
| `--plot_dir` | none | Also save each image with a bar chart of its predictions, as `<plot_dir>/<image name>.png` |
| `--tta` | off | Average the predictions for the image and its mirror image (slightly more accurate) |
| `--gpu` | off | Use the GPU |

- A checkpoint trained on a GPU works on a CPU-only machine.
- Images with transparency (PNG) or in greyscale are converted to RGB automatically.

## Web demo: `app.py`

```bash
pip install -r requirements-demo.txt
python app.py --checkpoint check_point.pt
```

Open the address it prints (normally <http://127.0.0.1:7860>), upload a flower photo and click **Submit** to see
the top 5 predictions as bars.

| Option | Default | What it does |
| --- | --- | --- |
| `--checkpoint` | `$CHECKPOINT` or `check_point.pt` | The model to serve |
| `--hub_repo` | `$HUB_REPO` | Download the model from a Hugging Face repository instead (see [Exporting and Publishing](Exporting-and-Publishing.md)) |
| `--category_names` | `cat_to_name.json` | Flower names |
| `--top_k` | `5` | Predictions shown |
| `--share` | off | Create a temporary public link (valid for a few days) |
| `--port` | `7860` | Port to serve on |
| `--gpu` | off | Use the GPU |

If a `flowers/test` folder is next to the app, the first image of four classes appears as one-click examples.

## Docker

The `Dockerfile` packages the web demo with a CPU-only PyTorch:

```bash
docker build -t flower-classifier .

# with a checkpoint on your machine
docker run -p 7860:7860 -v "$PWD/check_point.pt:/models/check_point.pt:ro" flower-classifier

# or with a model published to the Hugging Face Hub
docker run -p 7860:7860 -e HUB_REPO=your-name/flower-classifier flower-classifier
```

Then open <http://localhost:7860>. The image runs as an unprivileged user and installs the exact dependency
versions from `constraints.txt`. CI builds it and checks that the demo starts on every pull request.

## Free hosting on Hugging Face Spaces

1. Publish the model (see [Exporting and Publishing](Exporting-and-Publishing.md#publishing-to-the-hugging-face-hub)).
2. Create a new **Gradio** Space on [huggingface.co/spaces](https://huggingface.co/spaces).
3. Upload `app.py`, `model_utils.py` and `cat_to_name.json`.
4. Add a `requirements.txt` containing the lines of this project's `requirements.txt` plus
   `gradio>=5,<7` and `huggingface_hub>=0.20,<2`.
5. In the Space settings, add the variable `HUB_REPO` = `your-name/flower-classifier`.

The Space builds and serves the demo at a public URL you can put on your CV.
