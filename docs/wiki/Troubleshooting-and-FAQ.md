# Troubleshooting and FAQ

## Error messages

| Message | Cause and fix |
| --- | --- |
| `CUDA is not available.  Run without --gpu to use the CPU.` | No usable NVIDIA GPU. Drop `--gpu`, or on Colab choose *Runtime → Change runtime type → GPU*. |
| `CUDA out of memory` | Lower `--batch_size` (16, 8…). `convnext_tiny` and `vgg16` need the most memory. |
| `The classes in … do not match the checkpoint.` | You are resuming or evaluating with a dataset whose class folders differ from the one the model was trained on. Use the same data folder. |
| `No checkpoint was saved; are the epochs larger than the resumed epoch?` | Nothing was trained: `--epochs 0` with no fine-tuning, or `--resume` from a checkpoint already at or past `--epochs` + `--finetune_epochs`. Raise the epochs. |
| `Starting a fresh run: replacing the previous …check_point.pt` | Not an error. A new run without `--resume` replaces the old run's files in `--save_dir`. Use a different `--save_dir` to keep them. |
| `Checkpoint check_point.pt not found` (app.py) | Pass `--checkpoint path/to/check_point.pt`, set `CHECKPOINT`, or use `--hub_repo`. |
| `No images found in …` (predict.py) | The folder has no files with an image extension. |
| `invalid choice: '…'` for `--arch` | Use one of `densenet121`, `efficientnet_b0`, `convnext_tiny`, `resnet50`, `vgg16`, `alexnet`. |
| `… needs the onnx / tensorboard / huggingface_hub package` | Install the optional package named in the message, or `pip install -r requirements-dev.txt`. |
| `URLError` / SSL errors when training starts | The pretrained weights are downloaded from `download.pytorch.org` on first use and cached. Check your internet connection or proxy. |
| `Refusing to extract …` (download_data.py) | The archive contains unsafe paths or links and was rejected. Use the default URL or a trusted archive. |

## Questions

**Training is very slow.**
Make sure you pass `--gpu` and that it prints `Using GPU`. On a CPU a full run takes many hours; use Colab instead.
On a GPU, try `--num_workers 4` (Colab: 2), since loading images is often the bottleneck.

**Accuracy stays near 1% (random guessing).**
The model isn't learning. Check:
- Training images are in `train/<class>/`.
- `--learning_rate` isn't far too high (keep `0.001`).
- The pretrained weights actually loaded; the first run downloads them.

**Validation accuracy is much lower than training accuracy.**
Overfitting. See the tips in the [Training Guide](Training-Guide.md#recommended-settings).

**Colab disconnected in the middle of training.**
With `USE_DRIVE = True`, just run the notebook again and it resumes. Without it the files are gone, so turn it on
before long runs.

**Can I stop training and continue later?**
Yes: `--resume <save_dir>/last_checkpoint.pt` with the same epoch options. See
[Resuming an interrupted run](Training-Guide.md#resuming-an-interrupted-run).

**Which file do I use for predictions: `check_point.pt` or `last_checkpoint.pt`?**
`check_point.pt`. It is the best model by validation accuracy; `last_checkpoint.pt` is only for resuming.

**Will a checkpoint trained on a GPU work on my laptop?**
Yes. Checkpoints are saved with CPU tensors and loaded onto whatever device you choose.

**Do old checkpoints from the original project still load?**
Yes. Checkpoints that stored only `hidden_layer1` load with the original `[hidden_layer1, 90, 80]` classifier.

**The model is confident but wrong on my photo.**
It only knows the 102 species it was trained on, and assigns every image to one of them, even a dog or a car. It
also struggles with several flowers in one photo, unusual angles, and drawings.

**`--keep_alive` fails.**
It only works inside a Udacity workspace. Leave it off everywhere else.
