###################################################################################################################
# Exports a trained checkpoint so it can run without this project's code (e.g. in an app or another language)
# Notes - Run train.py first before this script
# Basic usage: python export.py checkpoint
# Options:
# Choose the format: python export.py checkpoint --format torchscript|onnx|both   (ONNX needs `pip install onnx`)
# Choose where to write: python export.py checkpoint --output_dir exported --name flowers
# Outputs (the exported model takes a normalized 1x3x224x224 image batch and returns class probabilities):
#   <name>.torchscript.pt  - load with torch.jit.load(); class labels are embedded as class_to_idx.json
#   <name>.onnx            - run with onnxruntime or any ONNX runtime
#   <name>.labels.json     - index -> class label (and flower name with --category_names), plus preprocessing
#####################################################################################################################
import argparse
import json
import os

import torch
from torch import nn

from model_utils import NORM_MEAN, NORM_STD, load_checkpoint


class Probabilities(nn.Module):
    ''' Wraps the model so the exported graph returns probabilities instead of log-probabilities. '''

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return torch.exp(self.model(images))


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Export a trained checkpoint to TorchScript and/or ONNX.')
    parser.add_argument('checkpoint', metavar='checkpoint', help='/path/to/checkpoint created by train.py')
    parser.add_argument('--format', default='torchscript', choices=['torchscript', 'onnx', 'both'],
                        help='export format (default: torchscript)')
    parser.add_argument('--output_dir', default='.', help='where to write the exported files (default: .)')
    parser.add_argument('--name', default='flower_classifier', help='base file name (default: flower_classifier)')
    parser.add_argument('--category_names', default=None,
                        help='JSON file mapping categories to real names, included in the labels file')
    return parser.parse_args(argv)


def labels_metadata(model, cat_to_name=None):
    ''' Everything a consumer needs to turn an image into a named prediction. '''
    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()}
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    metadata = {'labels': labels,
                'input': {'shape': [1, 3, 224, 224], 'resize': 256, 'center_crop': 224,
                          'mean': NORM_MEAN, 'std': NORM_STD},
                'output': 'probabilities'}
    if cat_to_name:
        metadata['names'] = [cat_to_name.get(label, label) for label in labels]
    return metadata


def export_torchscript(wrapped, example, path, metadata):
    traced = torch.jit.trace(wrapped, example)
    traced.save(path, _extra_files={'class_to_idx.json': json.dumps(metadata)})
    return path


def export_onnx(wrapped, example, path):
    try:
        import onnx  # noqa: F401  (required by the exporter)
    except ImportError:
        raise SystemExit('ONNX export needs the onnx package: pip install onnx') from None
    kwargs = dict(input_names=['image'], output_names=['probabilities'],
                  dynamic_axes={'image': {0: 'batch'}, 'probabilities': {0: 'batch'}})
    try:
        # The classic exporter is faster and works across PyTorch versions
        torch.onnx.export(wrapped, example, path, dynamo=False, **kwargs)
    except TypeError:
        # PyTorch versions before the dynamo exporter don't accept the dynamo argument
        torch.onnx.export(wrapped, example, path, **kwargs)
    return path


def main(argv=None):
    args = get_args(argv)
    model = load_checkpoint(args.checkpoint, torch.device('cpu'))
    wrapped = Probabilities(model).eval()
    example = torch.zeros(1, 3, 224, 224)

    cat_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    metadata = labels_metadata(model, cat_to_name)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, args.name)
    written = []
    with torch.no_grad():
        if args.format in ('torchscript', 'both'):
            written.append(export_torchscript(wrapped, example, base + '.torchscript.pt', metadata))
        if args.format in ('onnx', 'both'):
            written.append(export_onnx(wrapped, example, base + '.onnx'))
    with open(base + '.labels.json', 'w') as f:
        json.dump(metadata, f, indent=1)
    written.append(base + '.labels.json')

    for path in written:
        print('Exported {}'.format(path))
    return written


if __name__ == '__main__':
    main()
