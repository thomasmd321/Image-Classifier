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
# Resume an interrupted run: python train.py data_dir --gpu --resume save_directory/last_checkpoint.pt
# Typical run: python train.py flowers --learning_rate 0.001 --hidden_units 512 --arch "densenet121" --epochs 20 --gpu
#########################################################################################################
import argparse
import contextlib
import os
import time

import torch
from torch import nn, optim
from torchvision import datasets, transforms

from model_utils import (ARCHS, NORM_MEAN, NORM_STD, build_model, get_classifier, get_device,
                         load_checkpoint, save_checkpoint, set_seed)

BEST_CHECKPOINT = 'check_point.pt'
LAST_CHECKPOINT = 'last_checkpoint.pt'


###########################################
# Get the arguments from the command line
###########################################
def get_args(argv=None):
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
                        help='maximum number of epochs (default: 20)')
    parser.add_argument('--dropout', metavar='dropout', default=0.5, type=float,
                        help='hyperparameter (default: 0.5)')
    parser.add_argument('--batch_size', metavar='batch_size', default=20, type=int,
                        help='hyperparameter (default: 20)')
    parser.add_argument('--patience', metavar='patience', default=5, type=int,
                        help='stop after this many epochs without a validation accuracy improvement; '
                             '0 disables early stopping (default: 5)')
    parser.add_argument('--num_workers', metavar='num_workers', default=4, type=int,
                        help='data loading worker processes (default: 4)')
    parser.add_argument('--print_every', metavar='print_every', default=50, type=int,
                        help='print the training loss every this many batches (default: 50)')
    parser.add_argument('--seed', metavar='seed', default=None, type=int,
                        help='random seed for reproducible runs (default: not seeded)')
    parser.add_argument('--resume', metavar='checkpoint', default=None,
                        help='continue training from a checkpoint (use last_checkpoint.pt); '
                             'the architecture and hidden units come from the checkpoint')
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=False,
                        help='Use GPU for training (default: False)')
    parser.add_argument('--keep_alive', action='store_true', default=False,
                        help='Keep the Udacity workspace session alive while training')
    parser.add_argument('--version', action='version', version='%(prog)s 1.2  There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.')  # Decided to pull some wording from GCC
    return parser.parse_args(argv)


###############
# Load The Data
###############
def load_data(data_dir, batch_size, num_workers=4, pin_memory=False):
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

    # pin_memory speeds up copying batches to the GPU
    loader_args = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': pin_memory}
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **loader_args)
    valid_loader = torch.utils.data.DataLoader(validation_data, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **loader_args)
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
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    model.train()
    return total_loss / len(loader), correct / total


def format_time(seconds):
    return '{:.0f}m {:.0f}s'.format(seconds // 60, seconds % 60)


##########################
# Training function
##########################
def model_train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, args,
                save_kwargs, start_epoch=0, best_accuracy=None):
    ''' Trains the classifier, validating after every epoch.
        Saves the best model (by validation accuracy) to check_point.pt and the latest epoch
        to last_checkpoint.pt, and stops early after args.patience epochs without improvement.
    '''
    start_time = time.time()
    epochs_without_improvement = 0
    model.to(device)
    model.train()

    for e in range(start_epoch, args.epochs):
        running_loss = 0
        for step, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            ##############################
            # Forward and backward passes
            ##############################
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % args.print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, args.epochs),
                      "Batch: {}/{}... ".format(step, len(train_loader)),
                      "Loss: {:.4f}".format(running_loss / args.print_every))
                running_loss = 0

        ####################################
        # Validate at the end of each epoch
        ####################################
        valid_loss, accuracy = evaluate(model, valid_loader, criterion, device)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(valid_loss)
        new_lr = optimizer.param_groups[0]['lr']

        improved = best_accuracy is None or accuracy > best_accuracy
        if improved:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print("Epoch: {}/{}... ".format(e + 1, args.epochs),
              "Validation Loss: {:.4f}".format(valid_loss),
              "Accuracy: {:.4f}".format(accuracy),
              "(best {:.4f}{})".format(best_accuracy, ', saved' if improved else ''),
              "Time Elapsed: {}".format(format_time(time.time() - start_time)))
        if new_lr != old_lr:
            print('Validation loss stopped improving; learning rate reduced to {:g}'.format(new_lr))

        state = dict(optimizer=optimizer, scheduler=scheduler, epoch=e + 1, best_accuracy=best_accuracy)
        if improved:
            save_checkpoint(model, file_name=BEST_CHECKPOINT, **save_kwargs, **state)
        save_checkpoint(model, file_name=LAST_CHECKPOINT, **save_kwargs, **state)

        if args.patience and epochs_without_improvement >= args.patience:
            print('No improvement for {} epochs; stopping early.'.format(args.patience))
            break

    print('Training complete in {}'.format(format_time(time.time() - start_time)))
    return best_accuracy


def main(argv=None):
    args = get_args(argv)
    device = get_device(args.use_gpu)
    if args.seed is not None:
        set_seed(args.seed)

    train_data, train_loader, valid_loader, test_loader = load_data(
        args.data_dir, args.batch_size, args.num_workers, pin_memory=device.type == 'cuda')
    num_classes = len(train_data.classes)

    #################################
    # Build and train your network
    #################################
    start_epoch, best_accuracy, checkpoint = 0, None, None
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device, with_checkpoint=True)
        if checkpoint['class_to_idx'] != train_data.class_to_idx:
            raise SystemExit('The classes in {} do not match the checkpoint.'.format(args.data_dir))
        args.arch = model.arch
        args.hidden_units = checkpoint['hidden_layer1']
        args.dropout = checkpoint.get('dropout', args.dropout)
        start_epoch = checkpoint.get('epoch') or 0
        best_accuracy = checkpoint.get('best_accuracy')
        print('Resuming {} after epoch {}'.format(args.arch, start_epoch))
    else:
        model = build_model(args.arch, args.hidden_units, num_classes, args.dropout)
    model.to(device)

    criterion = nn.NLLLoss()
    # Only the classifier parameters are optimized; the feature extractor stays frozen
    optimizer = optim.Adam(get_classifier(model).parameters(), lr=args.learning_rate)
    # Cut the learning rate by 10x when the validation loss stops improving for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    if checkpoint is not None:
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

    save_kwargs = dict(hidden_units=args.hidden_units, num_classes=num_classes, dropout=args.dropout,
                       class_to_idx=train_data.class_to_idx, save_dir=args.save_dir)

    if args.keep_alive:
        # Keeps the Udacity workspace alive during long runs; only works inside that workspace
        from workspace_utils import active_session
        session = active_session()
    else:
        session = contextlib.nullcontext()

    with session:
        model_train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, args,
                    save_kwargs, start_epoch, best_accuracy)

        ##############################################
        # Do validation on the test set with the best
        # model (by validation accuracy)
        ##############################################
        best_path = os.path.join(args.save_dir, BEST_CHECKPOINT)
        if not os.path.exists(best_path):
            raise SystemExit('No checkpoint was saved; is --epochs larger than the resumed epoch?')
        model = load_checkpoint(best_path, device)
        start_time = time.time()
        _, test_accuracy = evaluate(model, test_loader, criterion, device)
        print('Test images network accuracy: {:.1f} %'.format(100 * test_accuracy))
        print("Time Elapsed: {}".format(format_time(time.time() - start_time)))

    print('Model: ', args.arch)
    print('Best model saved to: {}'.format(best_path))
    print('Latest epoch saved to: {}'.format(os.path.join(args.save_dir, LAST_CHECKPOINT)))


if __name__ == '__main__':
    main()
