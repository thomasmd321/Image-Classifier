#########################################################################################################
# Shared helpers for train.py and predict.py
# Builds the model, saves/loads checkpoints, preprocesses images and runs inference
#########################################################################################################
import os
from collections import OrderedDict

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

##########################
# Possible models to use
##########################
# Maps each supported architecture to the number of input features its classifier receives
ARCHS = {"densenet121": 1024,
         "alexnet": 9216}

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


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


def _load_pretrained(arch, pretrained=True):
    ''' Loads a torchvision model, supporting both the new (weights=) and old (pretrained=) APIs.
    '''
    constructor = getattr(models, arch)
    try:
        return constructor(weights='DEFAULT' if pretrained else None)
    except TypeError:
        # Older torchvision versions (< 0.13) only understand pretrained=
        return constructor(pretrained=pretrained)


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

    model.classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('inputs', nn.Linear(ARCHS[arch], hidden_units)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(hidden_units, 90)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(90, 80)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(80, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return model


################################
# Save / load the checkpoint
################################
def save_checkpoint(model, arch, hidden_units, num_classes, dropout, class_to_idx,
                    save_dir='.', file_name='check_point.pt'):
    ''' Saves everything needed to rebuild the model into save_dir/file_name.
    '''
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    model.cpu()
    torch.save({'structure': arch,
                'hidden_layer1': hidden_units,
                'num_classes': num_classes,
                'dropout': dropout,
                'state_dict': model.state_dict(),
                'class_to_idx': class_to_idx},
               path)
    return path


def load_checkpoint(path, device=torch.device('cpu')):
    ''' Loads a checkpoint and rebuilds the model in eval mode on the given device.
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
    probs, indices = torch.exp(output).topk(topk)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[index] for index in indices[0].tolist()]
    return probs[0].tolist(), classes
