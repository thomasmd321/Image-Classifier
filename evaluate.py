###################################################################################################################
# Evaluates a trained checkpoint on a dataset split and reports where the model struggles
# Notes - Run train.py first before this script
# Basic usage: python evaluate.py data_directory checkpoint
# Options:
# Evaluate the validation split instead of test: python evaluate.py data_dir checkpoint --split valid
# Use flower names in the report: python evaluate.py data_dir checkpoint --category_names cat_to_name.json
# Save the CSV reports elsewhere: python evaluate.py data_dir checkpoint --output_dir reports
# Writes per_class_accuracy.csv (every class, worst first) and confused_pairs.csv (most common mistakes)
#####################################################################################################################
import argparse
import csv
import json
import os
from collections import Counter

import torch
from torchvision import datasets

from model_utils import eval_transforms, get_device, load_checkpoint


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
    parser.add_argument('--batch_size', default=32, type=int, help='images per batch (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, help='data loading worker processes (default: 4)')
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=False,
                        help='Use GPU for evaluation (default: False)')
    return parser.parse_args(argv)


def collect_predictions(model, loader, device):
    ''' Returns (true labels, predicted labels) as lists of class indices. '''
    model.eval()
    truths, predictions = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            truths += labels.tolist()
            predictions += outputs.argmax(dim=1).cpu().tolist()
    return truths, predictions


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

    truths, predictions = collect_predictions(model, loader, device)
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}
    per_class, confused = build_report(truths, predictions, idx_to_class, cat_to_name)

    overall = sum(t == p for t, p in zip(truths, predictions)) / len(truths)
    print('{} accuracy: {:.1f} % ({} images, {} classes)'.format(
        args.split.capitalize(), 100 * overall, len(truths), len(per_class)))

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
    print('\nReports saved to {} and {}'.format(per_class_path, confused_path))
    return overall, per_class, confused


if __name__ == '__main__':
    main()
