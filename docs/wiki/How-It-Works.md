# How It Works

## Transfer learning

Training an image classifier from scratch needs millions of images. Instead, this project starts from a network
already trained on ImageNet, 1.2 million photos of 1,000 everyday categories. Its early layers detect edges and
textures and its later layers detect shapes and parts, which are useful for flowers too.

The network's original 1,000-way output layer is replaced with a new **classifier** for the 102 flower species:

```
image (3×224×224)
   │
   ▼
pretrained feature extractor ── frozen in phase 1; last block trainable in phase 2
   │   e.g. DenseNet-121 → 1,024 features
   ▼
new classifier (always trainable)
   Dropout(0.5)
   Linear(1024 → 512) + ReLU      ┐
   Linear(512 → 90)   + ReLU      ├ --hidden_units 512 90 80
   Linear(90 → 80)    + ReLU      ┘
   Linear(80 → 102)
   LogSoftmax  → log-probabilities for the 102 species
```

The classifier replaces the network's `classifier` attribute (`fc` for ResNet-50). For ConvNeXt, only the final
Linear layer inside `classifier` is replaced, because its pretrained LayerNorm and Flatten must be kept.

| Architecture | Features into the classifier | Unfrozen when fine-tuning |
| --- | --- | --- |
| densenet121 | 1,024 | `features.denseblock4`, `features.norm5` |
| efficientnet_b0 | 1,280 | `features.7`, `features.8` |
| convnext_tiny | 768 | `features.7`, `classifier.0` (LayerNorm) |
| resnet50 | 2,048 | `layer4` |
| vgg16 | 25,088 | `features.28` (last convolution) |
| alexnet | 9,216 | `features.10` (last convolution) |

## Why frozen layers stay in eval mode

PyTorch layers such as BatchNorm and Dropout behave differently in *train* and *eval* mode. BatchNorm in train mode
keeps updating its running statistics, even when its weights are frozen. Stochastic depth, used by EfficientNet
and ConvNeXt, randomly skips layers in train mode.

So during training only the parts being trained are put in train mode: the classifier, plus the unfrozen last
block when fine-tuning. The frozen feature extractor stays in eval mode and keeps behaving exactly as it was
pretrained (`model_utils.set_train_mode`). A regression test checks that BatchNorm statistics outside the trained
parts never change.

## Loss and label smoothing

The model outputs log-probabilities (LogSoftmax), so the natural loss is `NLLLoss`. Training uses
`CrossEntropyLoss(label_smoothing=0.1)` instead. CrossEntropyLoss applies log-softmax again, which leaves
log-probabilities unchanged, so without smoothing the two are identical (a test checks this). With smoothing, the
target for the correct class is slightly below 100%, which discourages over-confident predictions. Validation and
test losses use plain `NLLLoss`.

## Preprocessing

| Step | Training | Validation / test / prediction |
| --- | --- | --- |
| Geometry | Random rotation ±30°, random resized crop 224, random horizontal flip | Resize shortest side to 256, center crop 224 |
| Color | Random brightness/contrast/saturation ±20% | none |
| Tensor | Scale to 0–1, normalize with ImageNet mean/std | same |

## Checkpoint format

Checkpoints are dictionaries saved with `torch.save`:

| Key | Meaning |
| --- | --- |
| `structure` | Architecture name, e.g. `densenet121` |
| `hidden_layers` | Classifier hidden-layer sizes, e.g. `[512, 90, 80]` |
| `num_classes`, `dropout` | Classifier shape |
| `class_to_idx` | Class folder name → output index (from `ImageFolder`) |
| `state_dict` | All weights, on the CPU so any machine can load them |
| `epoch`, `phase`, `best_accuracy` | Training progress |
| `phase_complete` | `True` if the phase stopped early (a resume moves on to the next phase) |
| `optimizer_state`, `scheduler_state` | Only in `last_checkpoint.pt`, for resuming |

- `model_utils.load_checkpoint` rebuilds the model with pretrained-weight downloads skipped (the saved weights
  replace them anyway).
- The model is returned in eval mode, on the requested device.
- Checkpoints from the original version of the project, which stored only `hidden_layer1`, still load; the
  hidden layers are assumed to be `[hidden_layer1, 90, 80]`.

## Code map

| File | Responsibility |
| --- | --- |
| `model_utils.py` | Everything shared: architectures, `build_model`, train/eval modes, checkpoints, preprocessing, `predict`, plots |
| `train.py` | Data loading, the two-phase training loop, early stopping, LR schedule, history, resume |
| `evaluate.py` | Predictions over a split, per-class report, confusion matrix, mistakes gallery |
| `predict.py` | Command-line prediction for images or folders |
| `app.py` | Gradio web demo |
| `export.py` | TorchScript / ONNX export |
| `publish_to_hub.py` | Hugging Face Hub upload and model card |
| `download_data.py` | Dataset download and safe extraction |
