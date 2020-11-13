###################################################################################################################
# Uses a trained network to predict the class for an input image
# Notes - Run train.py first before this script
# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu
# Typical run: python predict.py --gpu --category_names cat_to_name.json --top_k 3 check_point.pt
#####################################################################################################################

###########################################
# Get the arguments from the command line
###########################################
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('image_checkpoint', metavar='image_checkpoint', 
                    help='/path/to/image_checkpoint')
parser.add_argument('--category_names', action="store",
                    dest="category_names", default='cat_to_name.json',  
                    help='a mapping of categories to real names ')
parser.add_argument('--top_k', metavar='top_k', 
                    default=3, type=int,
                    help='top KK most likely classes (default: 3)')
parser.add_argument('--gpu', dest='use_gpu', action="store_true",
                    default=False, 
                    help='Use GPU for training (default: True)')
parser.add_argument('--version', action='version', version='%(prog)s 1.0  There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.') # Decided to pull some wording from GCC
parser.add_argument('--load_dir', action="store",
                    dest="load_dir", default='./',  
                    help='directory_to_saved_checkpoints')
parser.add_argument('--test_image_dir', action="store",
                    dest="test_image_dir", default='./flowers/test/10',  
                    help='directory location to image used to test prediction')
parser.add_argument('--test_image', action="store",
                    dest="test_image", default='image_07104.jpg',  
                    help='Image file used to test prediction')
args = parser.parse_args()

### DEBUG ###
print(vars(args))
print(args.use_gpu)

#########################
# Various Python imports
#########################
import os
import sys
import numpy as np
import torch
import time
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
#%matplotlib inline
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict

image_checkpoint = args.image_checkpoint
category_names = args.category_names
top_k = args.top_k
use_gpu = args.use_gpu
load_dir = args.load_dir
test_image_dir = args.test_image_dir
test_image = args.test_image

device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
if use_gpu:
    #############################
    # Check if CUDA is available
    #############################
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Exiting ...')
        sys.exit()
    else:
        print('CUDA is available!  Training on GPU ...')

### DEBUG ###
#print('Passed GPU Check')
##########################
# Possible models to use
##########################
structures = {"densenet121" : 1024,
              "alexnet" : 9216}

def model_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 120,lr = 0.001):
#def model_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 512,lr = 0.01):
    ### DEBUG ###
    #print('Model Setup Function...')
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model. Did you mean densenet121 or alexnet?".format(structure))
        sys.exit()
        
    classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[structure], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))        
            
    for param in model.parameters():
        param.requires_grad = False
           
                    
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr )# Observe that all parameters are being optimized
        if use_gpu:
            model.cuda()
       
        return model, optimizer, criterion, structure

####################################################################
# Loads a checkpoint and rebuilds the model
####################################################################
def load_model(path='./',file_name='check_point.pt'):
    ### DEBUG ###
    #print('Load Model Function...')
    checkpoint = torch.load((path + file_name))
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    model,_,_,_ = model_setup(structure , 0.5,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    ### DEBUG ###
    #print('Exiting Load Model Function...')
    return model

 #############    
# Load Model
 #############
model2 = load_model(path=load_dir,file_name=image_checkpoint)

### DEBUG ###
#print(model2)
#print(model2.state_dict())

 ###########################
# Label mapping for DEBUG
 ###########################
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

#######################
# Image Preprocessing
#######################
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    ### DEBUG ###
    #print('Image Preprocessing Function...')
    img_pil = Image.open(image)
    
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor
    
####################    
# Class Prediction
####################
#model.class_to_idx =train_data.class_to_idx

### DEBUG ###
#print('Pre Class Prediction')
ctx = model2.class_to_idx
#use_gpu = True

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

#    if use_gpu:
#        model.to('cuda:0')
#    else:
#        model.to('cpu')
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        if use_gpu:
            output = model.forward(img_torch.cuda())
        else:
            output = model.forward(img_torch)
    
    probability = F.softmax(output.data,dim=1)
     
     ############################
    # Pulled from check_sanity()
     ############################
    probabilities = probability.topk(topk)
    #b = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    b = [cat_to_name[idx_to_class[index]] for index in np.array(probabilities[1][0])]
    print(b)
    
    return probability.topk(topk)   
    # Implement the code to predict the class from an image file

 ####################
# Get an test image
 ####################
#data_dir = 'flowers'
#img = (data_dir + '/test' + '/10/' + 'image_07104.jpg')
img = os.path.join(test_image_dir,test_image)
val1, val2 = predict(img, model2, top_k)
print(val1)
print(val2)
