import json

import pytest
import torch
from PIL import Image

import model_utils
import predict
import train
from conftest import CLASSES

TRAIN_ARGS = ['--epochs', '2', '--batch_size', '4', '--num_workers', '0', '--seed', '0']


@pytest.mark.parametrize('arch', sorted(model_utils.ARCHS))
def test_build_model_freezes_backbone(arch):
    model = model_utils.build_model(arch, hidden_units=32, num_classes=5, pretrained=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    head = model_utils.ARCHS[arch]['head']
    assert trainable and all(name.startswith(head + '.') for name in trainable)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, 5)


def test_train_then_predict(data_dir, tmp_path, capsys):
    save_dir = tmp_path / 'ckpt'
    train.main([str(data_dir), '--save_dir', str(save_dir)] + TRAIN_ARGS)
    assert (save_dir / train.BEST_CHECKPOINT).exists()
    assert (save_dir / train.LAST_CHECKPOINT).exists()
    assert 'Test images network accuracy' in capsys.readouterr().out

    checkpoint = torch.load(save_dir / train.LAST_CHECKPOINT)
    assert checkpoint['epoch'] == 2
    assert 'optimizer_state' in checkpoint and 'scheduler_state' in checkpoint

    names = {cls: 'flower {}'.format(cls) for cls in CLASSES}
    names_file = tmp_path / 'names.json'
    names_file.write_text(json.dumps(names))

    # Single image, twice: predictions must be deterministic
    image = str(data_dir / 'test' / '10' / 'img0.jpg')
    args = [image, str(save_dir / train.BEST_CHECKPOINT), '--top_k', '2',
            '--category_names', str(names_file)]
    first, second = predict.main(args), predict.main(args)
    assert first == second
    _, probs, labels = first[0]
    assert len(probs) == 2 and probs[0] >= probs[1]
    assert set(labels) <= set(names.values())


def test_predict_folder_with_plots_and_rgba(data_dir, tmp_path):
    save_dir = tmp_path / 'ckpt'
    train.main([str(data_dir), '--save_dir', str(save_dir), '--arch', 'alexnet'] + TRAIN_ARGS)

    folder = tmp_path / 'images'
    folder.mkdir()
    Image.new('RGBA', (300, 260), (255, 0, 0, 128)).save(folder / 'rgba.png')
    Image.new('L', (300, 260), 128).save(folder / 'grey.jpg')
    (folder / 'notes.txt').write_text('not an image')

    plot_dir = tmp_path / 'plots'
    results = predict.main([str(folder), str(save_dir / train.BEST_CHECKPOINT),
                            '--top_k', '10', '--plot_dir', str(plot_dir)])
    assert [r[0] for r in results] == [str(folder / 'grey.jpg'), str(folder / 'rgba.png')]
    # top_k is capped at the number of classes
    assert all(len(r[1]) == len(CLASSES) for r in results)
    assert sorted(p.name for p in plot_dir.iterdir()) == ['grey.png', 'rgba.png']


def test_resume_continues_from_last_epoch(data_dir, tmp_path, capsys):
    save_dir = tmp_path / 'ckpt'
    train.main([str(data_dir), '--save_dir', str(save_dir)] + TRAIN_ARGS)
    last = save_dir / train.LAST_CHECKPOINT
    capsys.readouterr()

    train.main([str(data_dir), '--save_dir', str(save_dir), '--resume', str(last),
                '--epochs', '3', '--batch_size', '4', '--num_workers', '0'])
    out = capsys.readouterr().out
    assert 'Resuming densenet121 after epoch 2' in out
    assert 'Epoch: 3/3' in out and 'Epoch: 1/3' not in out
    assert torch.load(last)['epoch'] == 3


def test_seed_makes_runs_reproducible(data_dir, tmp_path):
    weights = []
    for run in range(2):
        save_dir = tmp_path / 'run{}'.format(run)
        train.main([str(data_dir), '--save_dir', str(save_dir)] + TRAIN_ARGS)
        weights.append(torch.load(save_dir / train.LAST_CHECKPOINT)['state_dict'])
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])


def test_early_stopping(data_dir, tmp_path, capsys):
    # A learning rate of 0 never improves the model, so training stops after `patience` epochs
    train.main([str(data_dir), '--save_dir', str(tmp_path), '--epochs', '10', '--patience', '2',
                '--learning_rate', '0', '--batch_size', '4', '--num_workers', '0'])
    out = capsys.readouterr().out
    assert 'stopping early' in out
    assert torch.load(tmp_path / train.LAST_CHECKPOINT)['epoch'] == 3
