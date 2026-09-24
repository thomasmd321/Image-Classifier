###################################################################################################################
# Publishes a trained checkpoint to the Hugging Face Hub with a generated model card
# Notes - Train (and ideally evaluate) first, then log in once with: huggingface-cli login
# Basic usage: python publish_to_hub.py checkpoints/check_point.pt --repo_id your-name/flower-classifier
# Options:
# Preview the files and model card locally without uploading: python publish_to_hub.py checkpoint --dry_run preview
# Include the evaluation report from evaluate.py: python publish_to_hub.py checkpoint --repo_id ... --report_dir report
# Make the model private: python publish_to_hub.py checkpoint --repo_id ... --private
# The model card includes the architecture, classes, validation/test accuracy, the learning curves (history.png
# next to the checkpoint) and the evaluation charts, plus how to download and use the model.
#####################################################################################################################
import argparse
import csv
import json
import os
import shutil
import tempfile

from export import labels_metadata
from model_utils import checkpoint_hidden_units, load_checkpoint

# Files copied into the model repository when they exist
HISTORY_FILES = ['history.png', 'history.csv']
REPORT_FILES = ['summary.json', 'per_class_accuracy.csv', 'confused_pairs.csv', 'confusion_matrix.png',
                'misclassified.png']


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Publish a checkpoint to the Hugging Face Hub.')
    parser.add_argument('checkpoint', help='checkpoint created by train.py (usually check_point.pt)')
    parser.add_argument('--repo_id', default=None, help='Hub repository, e.g. your-name/flower-classifier')
    parser.add_argument('--history_dir', default=None,
                        help='folder with history.png / history.csv (default: the checkpoint folder)')
    parser.add_argument('--report_dir', default=None, help='folder with the evaluate.py report (optional)')
    parser.add_argument('--category_names', default='cat_to_name.json',
                        help='JSON file mapping categories to flower names (default: cat_to_name.json)')
    parser.add_argument('--private', action='store_true', default=False, help='create a private repository')
    parser.add_argument('--dry_run', metavar='folder', default=None,
                        help='write the repository files to this folder instead of uploading')
    args = parser.parse_args(argv)
    if not args.dry_run and not args.repo_id:
        parser.error('--repo_id is required unless --dry_run is used')
    return args


SPLIT_NAMES = {'test': 'Test', 'valid': 'Validation', 'train': 'Training'}


def report_accuracy(report_dir):
    ''' (accuracy, label) from evaluate.py's report, e.g. (0.93, 'Test accuracy'), or (None, None).
        The label names the split that was evaluated, so validation accuracy is never called test accuracy.
    '''
    if not report_dir:
        return None, None
    summary_path = os.path.join(report_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        label = '{} accuracy{}'.format(SPLIT_NAMES.get(summary['split'], summary['split']),
                                       ' (TTA)' if summary.get('tta') else '')
        return summary['accuracy'], label
    # Reports from before summary.json existed don't say which split they are
    path = os.path.join(report_dir, 'per_class_accuracy.csv')
    if not os.path.exists(path):
        return None, None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    images = sum(int(r['images']) for r in rows)
    return (sum(int(r['correct']) for r in rows) / images, 'Evaluation accuracy') if images else (None, None)


def history_summary(history_dir):
    ''' (epochs trained, fine-tuning epochs) from history.csv, or (None, None). '''
    path = os.path.join(history_dir, 'history.csv')
    if not os.path.exists(path):
        return None, None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return len(rows), sum(r['phase'] == 'finetune' for r in rows)


def model_card(repo_id, checkpoint, metadata, eval_accuracy, eval_label, epochs, finetune_epochs, files):
    ''' README.md for the model repository: Hub metadata header, results, usage and limitations. '''
    arch = checkpoint['structure']
    hidden = checkpoint_hidden_units(checkpoint)
    num_classes = len(metadata['labels'])
    valid_accuracy = checkpoint.get('best_accuracy')
    repo = repo_id or 'your-name/flower-classifier'

    lines = ['---', 'library_name: pytorch', 'license: mit', 'pipeline_tag: image-classification',
             'tags:', '- image-classification', '- pytorch', '- flowers', '- transfer-learning']
    if valid_accuracy is not None or eval_accuracy is not None:
        lines += ['model-index:', '- name: {}'.format(repo.split('/')[-1]), '  results:',
                  '  - task:', '      type: image-classification', '    dataset:',
                  '      name: 102 Category Flower Dataset', '      type: image-classification', '    metrics:']
        if eval_accuracy is not None:
            lines += ['    - type: accuracy', '      name: {}'.format(eval_label),
                      '      value: {:.4f}'.format(eval_accuracy)]
        if valid_accuracy is not None:
            lines += ['    - type: accuracy', '      name: Best validation accuracy (during training)',
                      '      value: {:.4f}'.format(valid_accuracy)]
    lines += ['---', '']

    lines += ['# Flower classifier ({})'.format(arch), '',
              'Classifies photos of flowers into {} species. A pretrained `{}` from torchvision extracts image '
              'features and a new feed-forward classifier (hidden layers: {}) predicts the species.'.format(
                  num_classes, arch, ', '.join(map(str, hidden))),
              '', 'Trained with the [Image Classifier project](https://github.com/thomasmd321/Image-Classifier) '
              'on the 102 Category Flower Dataset (Nilsback & Zisserman).', '', '## Results', '',
              '| Metric | Value |', '| --- | --- |']
    if eval_accuracy is not None:
        lines.append('| {} | {:.1%} |'.format(eval_label, eval_accuracy))
    if valid_accuracy is not None:
        lines.append('| Best validation accuracy | {:.1%} |'.format(valid_accuracy))
    if epochs is not None:
        lines.append('| Epochs trained | {} ({} fine-tuning) |'.format(epochs, finetune_epochs))
    lines.append('| Classes | {} |'.format(num_classes))
    lines.append('')
    if 'history.png' in files:
        lines += ['### Learning curves', '', '![Learning curves](history.png)', '']
    if 'confusion_matrix.png' in files:
        lines += ['### Most common mistakes', '', '![Confusion matrix](confusion_matrix.png)', '']
    if 'misclassified.png' in files:
        lines += ['![Most confident mistakes](misclassified.png)', '']

    lines += ['## Usage', '', '```python',
              'from huggingface_hub import hf_hub_download',
              'from model_utils import load_checkpoint, predict  # from the Image Classifier repository', '',
              "path = hf_hub_download('{}', 'check_point.pt')".format(repo),
              'model = load_checkpoint(path)',
              "probs, classes = predict('flower.jpg', model, topk=5)", '```', '',
              'Or run the web demo from the repository: `python app.py --hub_repo {}`.'.format(repo), '',
              '`labels.json` lists the class labels, flower names and the preprocessing (resize the shortest side '
              'to 256, center-crop 224, normalize with the ImageNet mean and std).', '',
              '## Limitations', '',
              '- Only recognizes the {} flower species it was trained on; any other image is still assigned '
              'to one of them.'.format(num_classes),
              '- Trained on about 6,500 photos, mostly close-ups of a single flower; accuracy drops on unusual '
              'angles, several species in one photo, or drawings.',
              '- Not intended for identifying plants where mistakes matter (for example, edibility or toxicity).',
              '']
    return '\n'.join(lines)


def build_repository(args, folder):
    ''' Writes the checkpoint, labels, charts and model card into folder; returns the file names. '''
    model, checkpoint = load_checkpoint(args.checkpoint, with_checkpoint=True)
    cat_to_name = None
    if args.category_names and os.path.exists(args.category_names):
        with open(args.category_names) as f:
            cat_to_name = json.load(f)

    os.makedirs(folder, exist_ok=True)
    shutil.copy(args.checkpoint, os.path.join(folder, 'check_point.pt'))
    metadata = labels_metadata(model, cat_to_name)
    with open(os.path.join(folder, 'labels.json'), 'w') as f:
        json.dump(metadata, f, indent=1)
    files = ['check_point.pt', 'labels.json']

    history_dir = args.history_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    for directory, names in ((history_dir, HISTORY_FILES), (args.report_dir, REPORT_FILES)):
        for name in names:
            if directory and os.path.exists(os.path.join(directory, name)):
                shutil.copy(os.path.join(directory, name), os.path.join(folder, name))
                files.append(name)

    epochs, finetune_epochs = history_summary(history_dir)
    eval_accuracy, eval_label = report_accuracy(args.report_dir)
    card = model_card(args.repo_id, checkpoint, metadata, eval_accuracy, eval_label, epochs, finetune_epochs,
                      files)
    with open(os.path.join(folder, 'README.md'), 'w') as f:
        f.write(card)
    files.append('README.md')
    return files


def main(argv=None):
    args = get_args(argv)
    if args.dry_run:
        files = build_repository(args, args.dry_run)
        print('Wrote {} to {} (nothing uploaded)'.format(', '.join(files), args.dry_run))
        return args.dry_run

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit('Publishing needs the huggingface_hub package: pip install huggingface_hub') from None
    with tempfile.TemporaryDirectory() as folder:
        files = build_repository(args, folder)
        api = HfApi()
        api.create_repo(args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=folder, commit_message='Upload flower classifier')
    url = 'https://huggingface.co/{}'.format(args.repo_id)
    print('Published {} to {}'.format(', '.join(files), url))
    return url


if __name__ == '__main__':
    main()
