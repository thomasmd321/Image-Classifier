#########################################################################################################
# Shared helpers for train.py and predict.py
# Builds the model, saves/loads checkpoints, preprocesses images and runs inference
#########################################################################################################
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

##########################
# Possible models to use
##########################
# Maps each supported architecture to the number of input features its classifier receives
# and the name of the attribute that holds the classifier (ResNets call it "fc")
ARCHS = {"densenet121": {"in_features": 1024, "head": "classifier"},
         "alexnet": {"in_features": 9216, "head": "classifier"},
         "vgg16": {"in_features": 25088, "head": "classifier"},
         "resnet50": {"in_features": 2048, "head": "fc"}}

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

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


def build_model(arch='densenet121', hidden_units=512, num_classes=102, dropout=0.5, pretrained=True):
    ''' Builds a pretrained feature extractor with a new, trainable feed-forward classifier.
    '''
    if arch not in ARCHS:
        raise SystemExit("Im sorry but {} is not a valid model. Did you mean one of {}?".format(
            arch, ', '.join(ARCHS)))

    model = _load_pretrained(arch, pretrained)

    # Freeze the pretrained feature parameters so only the new classifier is trained
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('inputs', nn.Linear(ARCHS[arch]['in_features'], hidden_units)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(80, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    setattr(model, ARCHS[arch]['head'], classifier)
    model.arch = arch
    return model


################################
# Save / load the checkpoint
################################
def save_checkpoint(model, hidden_units, num_classes, dropout, class_to_idx,
                    save_dir='.', file_name='check_point.pt',
                    optimizer=None, scheduler=None, epoch=None, best_accuracy=None):
    ''' Saves everything needed to rebuild the model into save_dir/file_name.
        The optimizer/scheduler state and epoch are included so training can be resumed.
    '''
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    # Copy the weights to the CPU so the checkpoint loads on machines without a GPU
    state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
    checkpoint = {'structure': model.arch,
                  'hidden_layer1': hidden_units,
                  'num_classes': num_classes,
                  'dropout': dropout,
                  'state_dict': state_dict,
                  'class_to_idx': class_to_idx,
                  'epoch': epoch,
                  'best_accuracy': best_accuracy}
    if optimizer is not None:
        checkpoint['optimizer_state'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, device=torch.device('cpu'), with_checkpoint=False):
    ''' Loads a checkpoint and rebuilds the model in eval mode on the given device.
        With with_checkpoint=True, also returns the raw checkpoint dict (for resuming training).
    '''
    checkpoint = torch.load(path, map_location=device)
    # The pretrained weights are overwritten by the state_dict, so skip downloading them
    model = build_model(checkpoint['structure'],
                        checkpoint['hidden_layer1'],
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
    img_pil = Image.open(image_path).convert('RGB')
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])
    return adjustments(img_pil)


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
def predict(image_path, model, topk=5, device=torch.device('cpu')):
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
