###################################################################################################################
# Evaluates a trained checkpoint on a dataset split and reports where the model struggles
# Notes - Run train.py first before this script
# Basic usage: python evaluate.py data_directory checkpoint
# Options:
# Evaluate the validation split instead of test: python evaluate.py data_dir checkpoint --split valid
# Use flower names in the report: python evaluate.py data_dir checkpoint --category_names cat_to_name.json
# Save the CSV reports elsewhere: python evaluate.py data_dir checkpoint --output_dir reports
# Test-time augmentation: python evaluate.py data_dir checkpoint --tta
# Writes per_class_accuracy.csv (every class, worst first), confused_pairs.csv (most common mistakes),
# confusion_matrix.png (which flowers get mixed up) and misclassified.png (the model's most confident mistakes)
#####################################################################################################################
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from model_utils import eval_transforms, get_device, load_checkpoint, predict_probs

# Chart colors: a single-hue blue ramp for counts (light -> dark); text stays in neutral ink
BLUE_RAMP = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
INK, MUTED, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Per-class accuracy and most-confused classes for a checkpoint.')
    parser.add_argument('data_dir', metavar='data_dir',
                        help='data directory location (must contain the chosen split folder)')
    parser.add_argument('checkpoint', metavar='checkpoint',
                        help='/path/to/checkpoint created by train.py')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'],
                        help='which dataset split to evaluate (default: test)')
    parser.add_argument('--category_names', dest='category_names', default=None,
                        help='JSON file mapping categories to real names (e.g. cat_to_name.json)')
    parser.add_argument('--output_dir', default='.',
                        help='where to write the CSV reports (default: current directory)')
    parser.add_argument('--top', default=10, type=int,
                        help='how many worst classes / confused pairs to print (default: 10)')
    parser.add_argument('--tta', action='store_true', default=False,
                        help='test-time augmentation: average predictions for each image and its mirror image')
    parser.add_argument('--gallery', default=16, type=int,
                        help='how many misclassified images to show in misclassified.png (default: 16; 0 = none)')
    parser.add_argument('--batch_size', default=32, type=int, help='images per batch (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, help='data loading worker processes (default: 4)')
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=False,
                        help='Use GPU for evaluation (default: False)')
    return parser.parse_args(argv)


def collect_predictions(model, loader, device, tta=False):
    ''' Returns (true labels, predicted labels, confidence of each prediction), in dataset order. '''
    model.eval()
    truths, predictions, confidences = [], [], []
    for inputs, labels in loader:
        probs = predict_probs(model, inputs.to(device), tta)
        confidence, predicted = probs.max(dim=1)
        truths += labels.tolist()
        predictions += predicted.cpu().tolist()
        confidences += confidence.cpu().tolist()
    return truths, predictions, confidences


def build_report(truths, predictions, idx_to_class, cat_to_name=None):
    ''' Returns (per-class rows sorted worst first, confused-pair rows sorted most common first). '''
    cat_to_name = cat_to_name or {}

    def name(idx):
        cls = idx_to_class[idx]
        return cat_to_name.get(cls, cls)

    totals = Counter(truths)
    correct = Counter(t for t, p in zip(truths, predictions) if t == p)
    per_class = [{'class': idx_to_class[idx], 'name': name(idx), 'images': totals[idx],
                  'correct': correct[idx], 'accuracy': correct[idx] / totals[idx]}
                 for idx in totals]
    per_class.sort(key=lambda row: (row['accuracy'], -row['images'], row['name']))

    mistakes = Counter((t, p) for t, p in zip(truths, predictions) if t != p)
    confused = [{'true_class': idx_to_class[t], 'true_name': name(t),
                 'predicted_class': idx_to_class[p], 'predicted_name': name(p),
                 'count': count, 'share_of_true_class': count / totals[t]}
                for (t, p), count in mistakes.items()]
    confused.sort(key=lambda row: (-row['count'], -row['share_of_true_class'], row['true_name']))
    return per_class, confused


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: '{:.4f}'.format(value) if isinstance(value, float) else value
                             for key, value in row.items()})


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(truths, predictions, names, path):
    ''' Heatmap of the mistakes only (the diagonal is left out): row = true flower, column = predicted flower. '''
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    n = len(names)
    counts = np.zeros((n, n), dtype=int)
    for t, p in zip(truths, predictions):
        if t != p:
            counts[t, p] += 1
    cmap = LinearSegmentedColormap.from_list('mistakes', BLUE_RAMP)
    cmap.set_bad(SURFACE)
    masked = np.ma.masked_equal(counts, 0)

    size = max(6, 0.17 * n)
    fontsize = 9 if n <= 30 else 6
    fig, ax = plt.subplots(figsize=(size + 1.5, size), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(masked, cmap=cmap, vmin=1, vmax=max(1, counts.max()), interpolation='nearest')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=fontsize, color=INK)
    ax.set_yticklabels(names, fontsize=fontsize, color=INK)
    ax.set_xlabel('Predicted flower', color=MUTED)
    ax.set_ylabel('True flower', color=MUTED)
    ax.set_title('Misclassifications ({} of {} images; correct predictions not shown)'.format(
        int(counts.sum()), len(truths)), loc='left', color=INK)
    # Faint grid between cells so rows and columns can be followed across the chart
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which='minor', color='#e4e3df', linewidth=0.3)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_color('#e4e3df')
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label('Images', color=MUTED)
    colorbar.locator = MaxNLocator(integer=True)
    colorbar.update_ticks()
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_gallery(samples, truths, predictions, confidences, names, path, limit=16, columns=4):
    ''' Grid of the most confident mistakes, each labeled with the true and predicted flower. '''
    mistakes = [i for i, (t, p) in enumerate(zip(truths, predictions)) if t != p]
    mistakes.sort(key=lambda i: -confidences[i])
    mistakes = mistakes[:limit]
    if not mistakes:
        return None
    plt = _pyplot()

    rows = (len(mistakes) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.4 * rows), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, i in zip(axes.flat, mistakes):
        image = Image.open(samples[i][0]).convert('RGB')
        # Same resize (shortest side 256) + 224 center crop the model sees, without the normalization
        scale = 256 / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
        left, top = (image.width - 224) // 2, (image.height - 224) // 2
        ax.imshow(image.crop((left, top, left + 224, top + 224)))
        ax.set_title('True: {}\nPredicted: {} ({:.0%})'.format(
            names[truths[i]], names[predictions[i]], confidences[i]), fontsize=9, color=INK, loc='left')
    fig.suptitle('Most confident mistakes', x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    return path


def main(argv=None):
    args = get_args(argv)
    device = get_device(args.use_gpu)
    model = load_checkpoint(args.checkpoint, device)

    data = datasets.ImageFolder(os.path.join(args.data_dir, args.split), transform=eval_transforms())
    if data.class_to_idx != model.class_to_idx:
        raise SystemExit('The classes in {} do not match the checkpoint.'.format(args.data_dir))
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers,
                                         pin_memory=device.type == 'cuda')

    cat_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    truths, predictions, confidences = collect_predictions(model, loader, device, args.tta)
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}
    per_class, confused = build_report(truths, predictions, idx_to_class, cat_to_name)

    overall = sum(t == p for t, p in zip(truths, predictions)) / len(truths)
    print('{} accuracy{}: {:.1f} % ({} images, {} classes)'.format(
        args.split.capitalize(), ' with TTA' if args.tta else '', 100 * overall, len(truths), len(per_class)))

    print('\nLowest accuracy classes:')
    for row in per_class[:args.top]:
        print('  {:<30} {:6.1f} %  ({}/{})'.format(row['name'], 100 * row['accuracy'],
                                                     row['correct'], row['images']))

    print('\nMost confused pairs (true -> predicted):')
    if not confused:
        print('  none - every image was classified correctly')
    for row in confused[:args.top]:
        print('  {:<30} -> {:<30} {} image(s)'.format(row['true_name'], row['predicted_name'], row['count']))

    os.makedirs(args.output_dir, exist_ok=True)
    per_class_path = os.path.join(args.output_dir, 'per_class_accuracy.csv')
    confused_path = os.path.join(args.output_dir, 'confused_pairs.csv')
    write_csv(per_class_path, per_class, ['class', 'name', 'images', 'correct', 'accuracy'])
    write_csv(confused_path, confused, ['true_class', 'true_name', 'predicted_class', 'predicted_name',
                                        'count', 'share_of_true_class'])
    names = [(cat_to_name or {}).get(idx_to_class[i], idx_to_class[i]) for i in range(len(idx_to_class))]
    charts = [plot_confusion(truths, predictions, names, os.path.join(args.output_dir, 'confusion_matrix.png'))]
    if args.gallery:
        charts.append(plot_gallery(data.samples, truths, predictions, confidences, names,
                                   os.path.join(args.output_dir, 'misclassified.png'), args.gallery))
    print('\nReports saved to {} and {}'.format(per_class_path, confused_path))
    print('Charts saved to {}'.format(' and '.join(c for c in charts if c)))
    return overall, per_class, confused


if __name__ == '__main__':
    main()
