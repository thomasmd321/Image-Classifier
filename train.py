#########################################################################################################
# Train a new network on a dataset and save the model as a checkpoint
# Notes -
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "densenet121"
# Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu
# Typical run: python train.py flowers --learning_rate 0.001 --hidden_units 512 --arch "densenet121" --epochs 20 --gpu
#########################################################################################################
import argparse
import contextlib
import os
import time

import torch
from torch import nn, optim
from torchvision import datasets, transforms

from model_utils import ARCHS, NORM_MEAN, NORM_STD, build_model, get_device, save_checkpoint


###########################################
# Get the arguments from the command line
###########################################
def get_args():
    parser = argparse.ArgumentParser(description='Train an image classifier and save it as a checkpoint.')
    parser.add_argument('data_dir', metavar='data_dir',
                        help='data directory location (must contain train/, valid/ and test/)')
    parser.add_argument('--save_dir', dest='save_dir', default='.',
                        help='directory to save checkpoints (default: current directory)')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=sorted(ARCHS),
                        help='model architecture (default: densenet121)')
    parser.add_argument('--learning_rate', metavar='learning_rate', default=0.001, type=float,
                        help='hyperparameter (default: 0.001)')
    parser.add_argument('--hidden_units', metavar='hidden_units', default=512, type=int,
                        help='hyperparameter (default: 512)')
    parser.add_argument('--epochs', metavar='epochs', default=20, type=int,
                        help='hyperparameter (default: 20)')
    parser.add_argument('--dropout', metavar='dropout', default=0.5, type=float,
                        help='hyperparameter (default: 0.5)')
    parser.add_argument('--batch_size', metavar='batch_size', default=20, type=int,
                        help='hyperparameter (default: 20)')
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=False,
                        help='Use GPU for training (default: False)')
    parser.add_argument('--keep_alive', action='store_true', default=False,
                        help='Keep the Udacity workspace session alive while training')
    parser.add_argument('--version', action='version', version='%(prog)s 1.1  There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.')  # Decided to pull some wording from GCC
    return parser.parse_args()


###############
# Load The Data
###############
def load_data(data_dir, batch_size, num_workers=1):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(NORM_MEAN, NORM_STD)])

    # Validation and test sets use the same deterministic transforms
    eval_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(NORM_MEAN, NORM_STD)])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    validation_data = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=eval_transforms)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=eval_transforms)

    print('Num training images: ', len(train_data))
    print('Num test images: ', len(test_data))
    print('Num valid images: ', len(validation_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)
    return train_data, train_loader, valid_loader, test_loader


##########################
# Evaluation function
##########################
def evaluate(model, loader, criterion, device):
    ''' Returns (average loss, accuracy) of the model over a data loader.
    '''
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    model.train()
    return total_loss / len(loader), correct / total


##########################
# Training function
##########################
def model_train(model, train_loader, valid_loader, criterion, optimizer, epochs, device, print_every=50):
    start_time = time.time()
    steps = 0
    model.to(device)
    model.train()
    running_loss = 0

    for e in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            ##############################
            # Forward and backward passes
            ##############################
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = evaluate(model, valid_loader, criterion, device)
                elapsed = time.time() - start_time
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Loss: {:.4f}".format(valid_loss),
                      "Accuracy: {:.4f}".format(accuracy),
                      "Time Elapsed: {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))
                running_loss = 0

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def main():
    args = get_args()
    device = get_device(args.use_gpu)

    train_data, train_loader, valid_loader, test_loader = load_data(args.data_dir, args.batch_size)
    num_classes = len(train_data.classes)

    #################################
    # Build and train your network
    #################################
    model = build_model(args.arch, args.hidden_units, num_classes, args.dropout)
    criterion = nn.NLLLoss()
    # Only the classifier parameters are optimized; the feature extractor stays frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    if args.keep_alive:
        # Keeps the Udacity workspace alive during long runs; only works inside that workspace
        from workspace_utils import active_session
        session = active_session()
    else:
        session = contextlib.nullcontext()

    with session:
        model_train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)

        ################################
        # Do validation on the test set
        ################################
        start_time = time.time()
        _, test_accuracy = evaluate(model, test_loader, criterion, device)
        print('Test images network accuracy: {:.1f} %'.format(100 * test_accuracy))
        elapsed = time.time() - start_time
        print("Time Elapsed: {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))

    ################################
    # Save the checkpoint
    ################################
    path = save_checkpoint(model, args.arch, args.hidden_units, num_classes, args.dropout,
                           train_data.class_to_idx, save_dir=args.save_dir)
    print('Model: ', args.arch)
    print('Saved Model: {}'.format(path))


if __name__ == '__main__':
    main()
