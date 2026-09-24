#########################################################################################################
# Shared helpers for train.py and predict.py
# Builds the model, saves/loads checkpoints, preprocesses images and runs inference
#########################################################################################################
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

##########################
# Possible models to use
##########################
# For each supported architecture:
#   in_features - number of inputs its classifier receives
#   head        - attribute that holds the classifier (ResNets call it "fc")
#   last_block  - modules unfrozen for fine-tuning (the last block of the feature extractor)
ARCHS = {"densenet121": {"in_features": 1024, "head": "classifier",
                         "last_block": ["features.denseblock4", "features.norm5"]},
         "alexnet": {"in_features": 9216, "head": "classifier", "last_block": ["features.10"]},
         "vgg16": {"in_features": 25088, "head": "classifier", "last_block": ["features.28"]},
         "resnet50": {"in_features": 2048, "head": "fc", "last_block": ["layer4"]}}

# The classifier layout used by the original project: 512 -> 90 -> 80 -> classes
DEFAULT_HIDDEN_UNITS = [512, 90, 80]

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]



def train_transforms():
    ''' Random augmentation used while training. '''
    return transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                               transforms.ToTensor(),
                               transforms.Normalize(NORM_MEAN, NORM_STD)])


def eval_transforms():
    ''' Deterministic resize + center crop used for validation, testing and prediction. '''
    return transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(NORM_MEAN, NORM_STD)])


CPU = torch.device('cpu')

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp')


def get_device(use_gpu):
    ''' Returns the torch device to use, exiting if a GPU was requested but is not available.
    '''
    if use_gpu:
        if not torch.cuda.is_available():
            raise SystemExit('CUDA is not available.  Run without --gpu to use the CPU.')
        print('CUDA is available!  Using GPU ...')
        return torch.device('cuda')
    print('Using CPU ...')
    return torch.device('cpu')


def set_seed(seed):
    ''' Seeds every random number generator used during training so runs can be reproduced.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_pretrained(arch, pretrained=True):
    ''' Loads a torchvision model, supporting both the new (weights=) and old (pretrained=) APIs.
    '''
    constructor = getattr(models, arch)
    try:
        return constructor(weights='DEFAULT' if pretrained else None)
    except TypeError:
        # Older torchvision versions (< 0.13) only understand pretrained=
        return constructor(pretrained=pretrained)


def get_classifier(model):
    ''' Returns the trainable classifier head of a model built by build_model.
    '''
    return getattr(model, ARCHS[model.arch]['head'])


def unfreeze_last_block(model):
    ''' Makes the last block of the feature extractor trainable for fine-tuning.
        Returns the newly trainable parameters.
    '''
    params = []
    for name in ARCHS[model.arch]['last_block']:
        for param in model.get_submodule(name).parameters():
            param.requires_grad = True
            params.append(param)
    return params


def _hidden_list(hidden_units):
    if isinstance(hidden_units, int):
        return [hidden_units]
    hidden_units = list(hidden_units)
    if not hidden_units:
        raise SystemExit('The classifier needs at least one hidden layer.')
    return hidden_units


def build_classifier(in_features, hidden_units, num_classes, dropout=0.5):
    ''' Feed-forward classifier: dropout, then one Linear + ReLU per hidden layer, then the output layer.
        The layer names match the original project, so [512, 90, 80] loads old checkpoints.
    '''
    sizes = [in_features] + _hidden_list(hidden_units)
    layers = [('dropout', nn.Dropout(dropout))]
    for i in range(1, len(sizes)):
        name = 'inputs' if i == 1 else 'hidden_layer{}'.format(i - 1)
        layers += [(name, nn.Linear(sizes[i - 1], sizes[i])), ('relu{}'.format(i), nn.ReLU())]
    layers += [('hidden_layer{}'.format(len(sizes) - 1), nn.Linear(sizes[-1], num_classes)),
               ('output', nn.LogSoftmax(dim=1))]
    return nn.Sequential(OrderedDict(layers))


def build_model(arch='densenet121', hidden_units=DEFAULT_HIDDEN_UNITS, num_classes=102, dropout=0.5,
                pretrained=True):
    ''' Builds a pretrained feature extractor with a new, trainable feed-forward classifier.
        hidden_units is one size or a list of sizes, one per hidden layer.
    '''
    if arch not in ARCHS:
        raise SystemExit("Im sorry but {} is not a valid model. Did you mean one of {}?".format(
            arch, ', '.join(ARCHS)))

    model = _load_pretrained(arch, pretrained)

    # Freeze the pretrained feature parameters so only the new classifier is trained
    for param in model.parameters():
        param.requires_grad = False

    classifier = build_classifier(ARCHS[arch]['in_features'], hidden_units, num_classes, dropout)
    setattr(model, ARCHS[arch]['head'], classifier)
    model.arch = arch
    return model


################################
# Save / load the checkpoint
################################
def save_checkpoint(model, hidden_units, num_classes, dropout, class_to_idx,
                    save_dir='.', file_name='check_point.pt',
                    optimizer=None, scheduler=None, epoch=None, best_accuracy=None, phase='head'):
    ''' Saves everything needed to rebuild the model into save_dir/file_name.
        The optimizer/scheduler state and epoch are included so training can be resumed.
    '''
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    # Copy the weights to the CPU so the checkpoint loads on machines without a GPU
    state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
    checkpoint = {'structure': model.arch,
                  'hidden_layers': _hidden_list(hidden_units),
                  'num_classes': num_classes,
                  'dropout': dropout,
                  'state_dict': state_dict,
                  'class_to_idx': class_to_idx,
                  'epoch': epoch,
                  'best_accuracy': best_accuracy,
                  'phase': phase}
    if optimizer is not None:
        checkpoint['optimizer_state'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    torch.save(checkpoint, path)
    return path


def checkpoint_hidden_units(checkpoint):
    ''' Hidden layer sizes stored in a checkpoint; older checkpoints only stored the first size. '''
    if 'hidden_layers' in checkpoint:
        return checkpoint['hidden_layers']
    return [checkpoint['hidden_layer1'], 90, 80]


def load_checkpoint(path, device=CPU, with_checkpoint=False):
    ''' Loads a checkpoint and rebuilds the model in eval mode on the given device.
        With with_checkpoint=True, also returns the raw checkpoint dict (for resuming training).
    '''
    checkpoint = torch.load(path, map_location=device)
    # The pretrained weights are overwritten by the state_dict, so skip downloading them
    model = build_model(checkpoint['structure'],
                        checkpoint_hidden_units(checkpoint),
                        checkpoint.get('num_classes', 102),
                        checkpoint.get('dropout', 0.5),
                        pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    model.eval()
    if with_checkpoint:
        return model, checkpoint
    return model


#######################
# Image Preprocessing
#######################
def process_image(image_path):
    ''' Scales, crops, and normalizes an image file for a PyTorch model,
        returns a tensor of shape (3, 224, 224)
    '''
    return eval_transforms()(Image.open(image_path).convert('RGB'))


def find_images(path):
    ''' Returns [path] for a single image, or the sorted image files inside a directory.
    '''
    if os.path.isdir(path):
        return sorted(os.path.join(path, name) for name in os.listdir(path)
                      if name.lower().endswith(IMAGE_EXTENSIONS))
    return [path]


####################
# Class Prediction
####################
def predict(image_path, model, topk=5, device=CPU):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        Returns the top K probabilities and their class labels.
    '''
    model.to(device)
    model.eval()
    img_torch = process_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_torch)

    # The model outputs log-probabilities, so exp() gives the probabilities
    topk = min(topk, output.shape[1])
    probs, indices = torch.exp(output).topk(topk)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[index] for index in indices[0].tolist()]
    return probs[0].tolist(), classes


def plot_prediction(image_path, names, probs, out_path, title=None):
    ''' Saves a figure of the image above a bar chart of the top K predictions.
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Undo the normalization so the cropped image displays with its original colors
    image = process_image(image_path).numpy().transpose((1, 2, 0))
    image = np.clip(np.array(NORM_STD) * image + np.array(NORM_MEAN), 0, 1)

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), nrows=2)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(title or os.path.basename(image_path))

    positions = np.arange(len(names))
    ax2.barh(positions, probs)
    ax2.set_yticks(positions)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
