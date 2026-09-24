#########################################################################################################
# Train a new network on a dataset and save the model as a checkpoint
# Notes -
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "densenet121"
# Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_units 512 256 --epochs 20
# Fine-tune the last backbone block afterwards: python train.py data_dir --finetune_epochs 5
# Use GPU for training: python train.py data_dir --gpu
# Resume an interrupted run: python train.py data_dir --gpu --resume save_directory/last_checkpoint.pt
# Typical run: python train.py flowers --arch "densenet121" --epochs 20 --finetune_epochs 5 --gpu --seed 42
#########################################################################################################
import argparse
import contextlib
import csv
import os
import time

import torch
from torch import nn, optim
from torchvision import datasets

from model_utils import (
    ARCHS,
    DEFAULT_HIDDEN_UNITS,
    build_model,
    checkpoint_hidden_units,
    eval_transforms,
    get_classifier,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_transforms,
    unfreeze_last_block,
)

BEST_CHECKPOINT = 'check_point.pt'
LAST_CHECKPOINT = 'last_checkpoint.pt'
HISTORY_CSV = 'history.csv'
HISTORY_PLOT = 'history.png'
HISTORY_FIELDS = ['epoch', 'phase', 'train_loss', 'valid_loss', 'valid_accuracy', 'learning_rate', 'seconds']


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
    parser.add_argument('--hidden_units', metavar='hidden_units', default=DEFAULT_HIDDEN_UNITS, type=int,
                        nargs='+',
                        help='size of each hidden layer (default: {})'.format(
                            ' '.join(map(str, DEFAULT_HIDDEN_UNITS))))
    parser.add_argument('--epochs', metavar='epochs', default=20, type=int,
                        help='maximum number of epochs training the classifier (default: 20)')
    parser.add_argument('--finetune_epochs', metavar='finetune_epochs', default=0, type=int,
                        help='epochs to fine-tune the last backbone block afterwards (default: 0 = off)')
    parser.add_argument('--finetune_lr', metavar='finetune_lr', default=1e-4, type=float,
                        help='learning rate while fine-tuning (default: 0.0001)')
    parser.add_argument('--dropout', metavar='dropout', default=0.5, type=float,
                        help='hyperparameter (default: 0.5)')
    parser.add_argument('--label_smoothing', metavar='label_smoothing', default=0.1, type=float,
                        help='label smoothing for the training loss (default: 0.1; 0 disables)')
    parser.add_argument('--batch_size', metavar='batch_size', default=20, type=int,
                        help='hyperparameter (default: 20)')
    parser.add_argument('--patience', metavar='patience', default=5, type=int,
                        help='stop a training phase after this many epochs without a validation accuracy '
                             'improvement; 0 disables early stopping (default: 5)')
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
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', default=True,
                        help='disable mixed precision on the GPU (it is always off on the CPU)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='also log metrics for TensorBoard to <save_dir>/runs (needs `pip install tensorboard`)')
    parser.add_argument('--keep_alive', action='store_true', default=False,
                        help='Keep the Udacity workspace session alive while training')
    parser.add_argument('--version', action='version',
                        version='%(prog)s 1.3  There is NO warranty; not even for MERCHANTABILITY or '
                                'FITNESS FOR A PARTICULAR PURPOSE.')  # Decided to pull some wording from GCC
    return parser.parse_args(argv)


###############
# Load The Data
###############
def load_data(data_dir, batch_size, num_workers=4, pin_memory=False):
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms())
    validation_data = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=eval_transforms())
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=eval_transforms())

    print('Num training images: ', len(train_data))
    print('Num test images: ', len(test_data))
    print('Num valid images: ', len(validation_data))

    # pin_memory speeds up copying batches to the GPU
    loader_args = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': pin_memory}
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **loader_args)
    valid_loader = torch.utils.data.DataLoader(validation_data, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **loader_args)
    return train_data, train_loader, valid_loader, test_loader


def autocast(device, enabled):
    ''' Mixed precision context for the forward pass; only used on the GPU. '''
    if enabled and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(enabled):
    ''' Scales the loss so float16 gradients don't underflow; a no-op when disabled. '''
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        # Older PyTorch versions only have the CUDA-specific class
        return torch.cuda.amp.GradScaler(enabled=enabled)


##########################
# Evaluation function
##########################
def evaluate(model, loader, criterion, device, use_amp=False):
    ''' Returns (average loss, accuracy) of the model over a data loader.
    '''
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device, use_amp):
                outputs = model(inputs)
            total_loss += criterion(outputs.float(), labels).item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    model.train()
    return total_loss / len(loader), correct / total


def format_time(seconds):
    return '{:.0f}m {:.0f}s'.format(seconds // 60, seconds % 60)


##########################
# Training history
##########################
def append_history(save_dir, row):
    ''' Appends one epoch's metrics to history.csv (writing the header for a new file). '''
    path = os.path.join(save_dir, HISTORY_CSV)
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_history(save_dir):
    path = os.path.join(save_dir, HISTORY_CSV)
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def plot_history(save_dir):
    ''' Saves learning curves (loss and validation accuracy per epoch) to history.png. '''
    rows = read_history(save_dir)
    if not rows:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Colors: categorical slots 1-2 of a colorblind-validated palette; text stays in neutral ink
    blue, orange = '#2a78d6', '#eb6834'
    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

    epochs = [int(r['epoch']) for r in rows]
    train_loss = [float(r['train_loss']) for r in rows]
    valid_loss = [float(r['valid_loss']) for r in rows]
    accuracy = [100 * float(r['valid_accuracy']) for r in rows]
    finetune_start = next((int(r['epoch']) for r in rows if r['phase'] == 'finetune'), None)
    best = max(range(len(rows)), key=lambda i: accuracy[i])

    fig, (ax_loss, ax_acc) = plt.subplots(nrows=2, figsize=(8, 7), sharex=True, facecolor=surface)
    for ax in (ax_loss, ax_acc):
        ax.set_facecolor(surface)
        ax.grid(axis='y', color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        if finetune_start is not None:
            ax.axvline(finetune_start - 0.5, color=muted, linewidth=1, linestyle='--')

    ax_loss.plot(epochs, train_loss, color=blue, linewidth=2, marker='o', markersize=4, label='Training loss')
    ax_loss.plot(epochs, valid_loss, color=orange, linewidth=2, marker='o', markersize=4,
                 label='Validation loss')
    # Direct labels at the end of each line, plus a legend, so identity never relies on color alone
    # (nudged apart vertically when the two lines end close together)
    span = (max(train_loss + valid_loss) - min(train_loss + valid_loss)) or 1
    close = abs(train_loss[-1] - valid_loss[-1]) < 0.06 * span
    upper = 'training' if train_loss[-1] >= valid_loss[-1] else 'validation'
    for values, name in ((train_loss, 'training'), (valid_loss, 'validation')):
        nudge = (6 if name == upper else -6) if close else 0
        ax_loss.annotate(name, (epochs[-1], values[-1]), xytext=(6, nudge), textcoords='offset points',
                         va='center', fontsize=9, color=ink)
    ax_loss.set_title('Loss', loc='left', fontsize=11, color=ink)
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=ink, loc='upper right')
    if finetune_start is not None:
        ax_loss.annotate('fine-tuning', (finetune_start - 0.5, 1), xycoords=('data', 'axes fraction'),
                         xytext=(4, -4), textcoords='offset points', va='top', fontsize=9, color=muted)

    ax_acc.plot(epochs, accuracy, color=blue, linewidth=2, marker='o', markersize=4)
    ax_acc.annotate('best {:.1f}%'.format(accuracy[best]), (epochs[best], accuracy[best]),
                    xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, color=ink)
    ax_acc.set_title('Validation accuracy (%)', loc='left', fontsize=11, color=ink)
    ax_acc.set_xlabel('Epoch', color=muted, fontsize=9)
    ax_acc.set_xticks(epochs if len(epochs) <= 20 else epochs[::max(1, len(epochs) // 10)])
    ax_acc.margins(x=0.08, y=0.15)
    ax_loss.margins(x=0.08)

    fig.tight_layout()
    path = os.path.join(save_dir, HISTORY_PLOT)
    fig.savefig(path, dpi=120, facecolor=surface)
    plt.close(fig)
    return path


##########################
# Training function
##########################
def make_optimizer(model, phase, args):
    ''' Adam over the trainable parameters, plus a scheduler that cuts the LR by 10x
        when the validation loss stops improving for 2 epochs.
    '''
    if phase == 'finetune':
        params = unfreeze_last_block(model) + list(get_classifier(model).parameters())
        lr = args.finetune_lr
    else:
        # Only the classifier parameters are optimized; the feature extractor stays frozen
        params = get_classifier(model).parameters()
        lr = args.learning_rate
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    return optimizer, scheduler


def model_train(model, train_loader, valid_loader, train_criterion, eval_criterion, optimizer, scheduler,
                device, args, save_kwargs, phase, first_epoch, last_epoch, best_accuracy=None, writer=None):
    ''' Trains for epochs [first_epoch, last_epoch), validating after every epoch.
        Saves the best model (by validation accuracy) to check_point.pt and the latest epoch
        to last_checkpoint.pt, and stops early after args.patience epochs without improvement.
    '''
    start_time = time.time()
    epochs_without_improvement = 0
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = make_grad_scaler(use_amp)
    total_epochs = args.epochs + args.finetune_epochs
    model.to(device)
    model.train()

    for e in range(first_epoch, last_epoch):
        epoch_start = time.time()
        running_loss = epoch_loss = 0
        for step, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            ##############################
            # Forward and backward passes
            ##############################
            optimizer.zero_grad()
            with autocast(device, use_amp):
                outputs = model(inputs)
            loss = train_criterion(outputs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if step % args.print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, total_epochs),
                      "Batch: {}/{}... ".format(step, len(train_loader)),
                      "Loss: {:.4f}".format(running_loss / args.print_every))
                running_loss = 0

        ####################################
        # Validate at the end of each epoch
        ####################################
        valid_loss, accuracy = evaluate(model, valid_loader, eval_criterion, device, use_amp)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(valid_loss)
        new_lr = optimizer.param_groups[0]['lr']

        improved = best_accuracy is None or accuracy > best_accuracy
        if improved:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print("Epoch: {}/{}... ".format(e + 1, total_epochs),
              "Validation Loss: {:.4f}".format(valid_loss),
              "Accuracy: {:.4f}".format(accuracy),
              "(best {:.4f}{})".format(best_accuracy, ', saved' if improved else ''),
              "Time Elapsed: {}".format(format_time(time.time() - start_time)))
        if new_lr != old_lr:
            print('Validation loss stopped improving; learning rate reduced to {:g}'.format(new_lr))

        append_history(args.save_dir, {'epoch': e + 1, 'phase': phase,
                                       'train_loss': '{:.6f}'.format(epoch_loss / len(train_loader)),
                                       'valid_loss': '{:.6f}'.format(valid_loss),
                                       'valid_accuracy': '{:.6f}'.format(accuracy),
                                       'learning_rate': '{:g}'.format(old_lr),
                                       'seconds': '{:.1f}'.format(time.time() - epoch_start)})
        if writer is not None:
            writer.add_scalars('loss', {'train': epoch_loss / len(train_loader), 'validation': valid_loss}, e + 1)
            writer.add_scalar('accuracy/validation', accuracy, e + 1)
            writer.add_scalar('learning_rate', old_lr, e + 1)

        state = dict(optimizer=optimizer, scheduler=scheduler, epoch=e + 1, best_accuracy=best_accuracy,
                     phase=phase)
        if improved:
            save_checkpoint(model, file_name=BEST_CHECKPOINT, **save_kwargs, **state)
        save_checkpoint(model, file_name=LAST_CHECKPOINT, **save_kwargs, **state)

        if args.patience and epochs_without_improvement >= args.patience:
            print('No improvement for {} epochs; stopping this phase early.'.format(args.patience))
            break

    print('{} training complete in {}'.format('Fine-tuning' if phase == 'finetune' else 'Classifier',
                                              format_time(time.time() - start_time)))
    return best_accuracy


def make_writer(args):
    ''' A TensorBoard SummaryWriter logging to <save_dir>/runs, or None without --tensorboard. '''
    if not args.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise SystemExit('--tensorboard needs the tensorboard package: pip install tensorboard') from None
    log_dir = os.path.join(args.save_dir, 'runs')
    print('Logging to TensorBoard; view with: tensorboard --logdir {}'.format(log_dir))
    return SummaryWriter(log_dir)


def main(argv=None):
    args = get_args(argv)
    device = get_device(args.use_gpu)
    if args.seed is not None:
        set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_data, train_loader, valid_loader, test_loader = load_data(
        args.data_dir, args.batch_size, args.num_workers, pin_memory=device.type == 'cuda')
    num_classes = len(train_data.classes)
    best_path = os.path.join(args.save_dir, BEST_CHECKPOINT)

    #################################
    # Build and train your network
    #################################
    start_epoch, best_accuracy, checkpoint, phase = 0, None, None, 'head'
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device, with_checkpoint=True)
        if checkpoint['class_to_idx'] != train_data.class_to_idx:
            raise SystemExit('The classes in {} do not match the checkpoint.'.format(args.data_dir))
        args.arch = model.arch
        args.hidden_units = checkpoint_hidden_units(checkpoint)
        args.dropout = checkpoint.get('dropout', args.dropout)
        start_epoch = checkpoint.get('epoch') or 0
        best_accuracy = checkpoint.get('best_accuracy')
        phase = checkpoint.get('phase', 'head')
        print('Resuming {} after epoch {} ({} phase)'.format(args.arch, start_epoch, phase))
    else:
        # A fresh run starts a fresh history
        for name in (HISTORY_CSV, HISTORY_PLOT):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(args.save_dir, name))
        model = build_model(args.arch, args.hidden_units, num_classes, args.dropout)
    model.to(device)

    # Label smoothing only applies to the training loss; validation/test report the plain loss.
    # The model outputs log-probabilities; CrossEntropyLoss re-applies log_softmax, which leaves
    # log-probabilities unchanged, so this equals NLLLoss plus label smoothing.
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.NLLLoss()

    save_kwargs = dict(hidden_units=args.hidden_units, num_classes=num_classes, dropout=args.dropout,
                       class_to_idx=train_data.class_to_idx, save_dir=args.save_dir)
    phases = [('head', 0, args.epochs), ('finetune', args.epochs, args.epochs + args.finetune_epochs)]

    if args.keep_alive:
        # Keeps the Udacity workspace alive during long runs; only works inside that workspace
        from workspace_utils import active_session
        session = active_session()
    else:
        session = contextlib.nullcontext()

    writer = make_writer(args)
    with session:
        for name, first, last in phases:
            if last <= first or (name == 'head' and phase == 'finetune'):
                continue
            first = max(first, start_epoch)
            if first >= last:
                continue
            resuming_this_phase = checkpoint is not None and checkpoint.get('phase', 'head') == name
            if name == 'finetune' and not resuming_this_phase and os.path.exists(best_path):
                # Fine-tune starting from the best classifier found so far
                model.load_state_dict(torch.load(best_path, map_location=device)['state_dict'])
                print('Fine-tuning the last {} block from the best classifier ...'.format(args.arch))
            optimizer, scheduler = make_optimizer(model, name, args)
            if resuming_this_phase:
                if 'optimizer_state' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                if 'scheduler_state' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state'])
            best_accuracy = model_train(model, train_loader, valid_loader, train_criterion, eval_criterion,
                                        optimizer, scheduler, device, args, save_kwargs, name, first, last,
                                        best_accuracy, writer)

        plot_path = plot_history(args.save_dir)

        ##############################################
        # Do validation on the test set with the best
        # model (by validation accuracy)
        ##############################################
        if not os.path.exists(best_path):
            raise SystemExit('No checkpoint was saved; are the epochs larger than the resumed epoch?')
        model = load_checkpoint(best_path, device)
        start_time = time.time()
        _, test_accuracy = evaluate(model, test_loader, eval_criterion, device)
        print('Test images network accuracy: {:.1f} %'.format(100 * test_accuracy))
        if writer is not None:
            writer.add_scalar('accuracy/test', test_accuracy)
            writer.close()
        print("Time Elapsed: {}".format(format_time(time.time() - start_time)))

    print('Model: ', args.arch)
    print('Best model saved to: {}'.format(best_path))
    print('Latest epoch saved to: {}'.format(os.path.join(args.save_dir, LAST_CHECKPOINT)))
    if plot_path:
        print('Training history saved to: {} and {}'.format(os.path.join(args.save_dir, HISTORY_CSV), plot_path))


if __name__ == '__main__':
    main()
