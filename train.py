#########################################################################################################
# Train a new network on a dataset and save the model as a checkpoint
# Notes - 
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "densenet121"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu
# Typical run: python train.py flowers --learning_rate 0.01 --hidden_units 512 --arch "densenet121" --epochs 20 --gpu
#########################################################################################################

###########################################
# Get the arguments from the command line
###########################################
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', metavar='data_dir', 
                    help='data directory location')
parser.add_argument('--save_dir', action="store",
                    dest="save_dir", default='./',  
                    help='directory_to_save_checkpoints')
parser.add_argument('--arch', action="store",
                    dest="arch", default='densenet121',  
                    help='model architecture')
parser.add_argument('--learning_rate', metavar='learning_rate', 
                    default=0.01,
                    help='hyperparameter (default: 0.01')
parser.add_argument('--hidden_units', metavar='hidden_units',
                    default=512, type=int,
                    help='hyperparameter (default: 512)')
parser.add_argument('--epochs', metavar='epochs',
                    default=20, type=int,
                    help='hyperparameter (default: 20)')
parser.add_argument('--gpu', dest='use_gpu', action="store_true",
                    default=False, 
                    help='Use GPU for training (default: True)')
parser.add_argument('--version', action='version', version='%(prog)s 1.0  There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.') # Decided to pull some wording from GCC

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

use_gpu = args.use_gpu
epochs = args.epochs
hidden_units = args.hidden_units
learning_rate = args.learning_rate
arch = args.arch
save_dir = args.save_dir
data_dir = args.data_dir

device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
if use_gpu:
    #############################
    # Check if CUDA is available
    #############################
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
else:
    print('Training on CPU ...')

 ###############
# Load The Data 
 ###############
#data_dir = 'flowers' # Set from command line
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 #####################################
# Load the datasets with ImageFolder
 #####################################
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)
                                      
 ###########################   
# print out some data stats
 ###########################
print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))
print('Num valid images: ', len(validation_data))
#image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)

 #####################################################################
# Using the image datasets and the trainforms, define the dataloaders
# define dataloader parameters
 #####################################################################
batch_size = 20
num_workers=1

 ######################
# prepare data loaders
 ######################
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)

 #########################
# Label Mapping for DEBUG
 #########################
#with open('cat_to_name.json', 'r') as f:
#    cat_to_name = json.load(f)

################################    
# Build and train your network
################################
Start_Time = time.time()

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
else:
    model.to('cpu')

##########################
# Possible models to use
##########################
structures = {"densenet121" : 1024,
              "alexnet" : 9216}

def model_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 120,lr = 0.001,classes_in_dataset=102):
#def model_setup(structure='densenet121',dropout=0.5, hidden_layer1 = 512,lr = 0.01):
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
            ('hidden_layer3',nn.Linear(80,classes_in_dataset)), # There are 102 classes in the flowers dataset
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

#################################    
# Call the model setup function  
#################################
#model,optimizer,criterion,model_structure = model_setup('densenet121')# Training complete in 69m 15s e=6 GPU=enabled
model,optimizer,criterion,model_structure = model_setup(arch, 0.5, int(hidden_units), float(learning_rate))
#model,optimizer,criterion,model_structure = model_setup('alexnet')# Training complete in 55m 59s e=6 GPU=enabled

### DEBUG ###
#epochs = 6
#epochs = 1

##########################
# Training function
##########################
def model_train(model, train_loader, valid_loader, epochs=6, use_gpu=True):
    print_every = 5
    steps = 0
    loss_show=[]

    if use_gpu and torch.cuda.is_available():
        ##################
        # change to cuda
        ##################
        model.to(device)
    else:
        model.to(device)
    ##########
    # Train
    ##########
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
        
            if use_gpu:
                inputs,labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            ##############################
            # Forward and backward passes
            ##############################
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0
             
                for ii, (inputs2,labels2) in enumerate(valid_loader):
                    optimizer.zero_grad()
                
                    if use_gpu:
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                vlost = vlost / len(valid_loader)
                accuracy = accuracy /len(valid_loader)
                # Print out status info  
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy),
                       "Time Elipsed elapsed: {:.0f}m {:.0f}s".format((time.time() - Start_Time) // 60, (time.time() - Start_Time) % 60))
            
                running_loss = 0
    return

#########################
# Call Training Function
#########################
#model_train(model, train_loader, valid_loader, epochs=1, use_gpu=True) # Quick test 11.5 min
model_train(model, train_loader, valid_loader, epochs, use_gpu) # Quick test 11.5 min
#model_train(model, train_loader, valid_loader, epochs=6, use_gpu=True) # Long test 69 min

#######################################################
### Prind out some data to show Training completion ###
#######################################################
print("*** Finished: Build and train your network ***")   
time_elapsed = time.time() - Start_Time
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Model: ', model_structure)
print("Epochs: ", epochs)
print('\a') # would like to ring bell

################################
# Do validation on the test set
################################
Start_Time = time.time()

def check_accuracy_on_test(test_loader):    
    correct = 0
    total = 0
    if use_gpu and torch.cuda.is_available():
        ##################
        # change to cuda
        ##################
        model.to(device)
    else:
        model.to(device)
        
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test images network accuracy: %d %%' % (100 * correct / total))
    return (100 * correct / total)
        
check_accuracy_on_test(test_loader)
print("Time Elipsed elapsed: {:.0f}m {:.0f}s".format((time.time() - Start_Time) // 60, (time.time() - Start_Time) % 60))
#print("\a")
#import subprocess
#subprocess.call(["printf", '\7' ])

#################
# Save the model
#################
print('Model: ', model_structure)
################################
# Save the checkpoint 
################################
def save_checkpoint(model, train_data, path='./', file_name='check_point.pt'):
    model.class_to_idx = train_data.class_to_idx
          
    ##################
    # change to CPU
    ##################
    model.cpu

    #############################
    # Remove old checkpoint file
    #############################
    try:
        os.remove(os.path.join(path, file_name))
    except OSError:
        pass

    torch.save({'structure' :model_structure,
                'hidden_layer1':hidden_units,
                #'hidden_layer1':120,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                file_name)
    return

print('Saving Model: {}/{}'.format(save_dir, 'check_point.pt'))
save_checkpoint(model, train_data, path=save_dir, file_name='check_point.pt')

