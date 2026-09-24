import os
import sys

import numpy as np
import pytest
from PIL import Image

# Make the project modules importable from the tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_utils  # noqa: E402

CLASSES = ['1', '10', '2']


@pytest.fixture(autouse=True)
def no_pretrained_download(monkeypatch):
    ''' Use randomly initialized weights so the tests run offline and quickly. '''
    original = model_utils._load_pretrained
    monkeypatch.setattr(model_utils, '_load_pretrained', lambda arch, pretrained=True: original(arch, False))


@pytest.fixture(scope='session')
def data_dir(tmp_path_factory):
    ''' A tiny ImageFolder dataset with train/valid/test splits of random images. '''
    root = tmp_path_factory.mktemp('flowers')
    rng = np.random.default_rng(0)
    for split in ['train', 'valid', 'test']:
        for cls in CLASSES:
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(4):
                pixels = (rng.random((260, 300, 3)) * 255).astype('uint8')
                Image.fromarray(pixels).save(folder / 'img{}.jpg'.format(i))
    return root


@pytest.fixture(scope='session')
def trained(data_dir, tmp_path_factory):
    ''' One small trained run (2 classifier epochs + 1 fine-tuning epoch) shared by several tests. '''
    import train
    save_dir = tmp_path_factory.mktemp('trained')
    original = model_utils._load_pretrained
    model_utils._load_pretrained = lambda arch, pretrained=True: original(arch, False)
    try:
        train.main([str(data_dir), '--save_dir', str(save_dir), '--epochs', '2', '--finetune_epochs', '1',
                    '--hidden_units', '64', '32', '--batch_size', '4', '--num_workers', '0', '--seed', '0'])
    finally:
        model_utils._load_pretrained = original
    return save_dir
