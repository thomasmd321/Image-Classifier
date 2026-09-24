''' Regression tests for the issues found in the code review of the training, publishing and demo code. '''
import csv
import json

import pytest
import torch
from torch import nn

import app
import evaluate
import model_utils
import publish_to_hub
import train

FAST = ['--batch_size', '4', '--num_workers', '0', '--seed', '0']


def batchnorm_layers(module):
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]


def test_frozen_layers_stay_in_eval_mode():
    model = model_utils.build_model('densenet121', [32], num_classes=3, pretrained=False)
    model_utils.set_train_mode(model)
    assert model.classifier.training
    assert not any(bn.training for bn in batchnorm_layers(model.features))

    model_utils.set_train_mode(model, finetune=True)
    assert all(bn.training for bn in batchnorm_layers(model.features.denseblock4))
    assert not model.features.norm0.training and not model.features.denseblock1.training


def test_frozen_batchnorm_statistics_do_not_drift(data_dir, tmp_path):
    ''' A fresh (untrained) model starts with running_mean = 0; training the classifier must leave it there. '''
    train.main([str(data_dir), '--save_dir', str(tmp_path), '--epochs', '1'] + FAST)
    state = torch.load(tmp_path / train.LAST_CHECKPOINT)['state_dict']
    assert torch.count_nonzero(state['features.norm0.running_mean']) == 0
    assert torch.count_nonzero(state['features.denseblock4.denselayer16.norm2.running_mean']) == 0


def test_resume_after_early_stop_moves_on_to_finetuning(data_dir, tmp_path, capsys):
    # A learning rate of 0 never improves, so with patience 1 the classifier phase stops after epoch 2 of 5
    train.main([str(data_dir), '--save_dir', str(tmp_path), '--epochs', '5', '--patience', '1',
                '--learning_rate', '0'] + FAST)
    last = torch.load(tmp_path / train.LAST_CHECKPOINT)
    assert (last['epoch'], last['phase'], last['phase_complete']) == (2, 'head', True)
    capsys.readouterr()

    train.main([str(data_dir), '--save_dir', str(tmp_path), '--resume', str(tmp_path / train.LAST_CHECKPOINT),
                '--epochs', '5', '--finetune_epochs', '1'] + FAST)
    out = capsys.readouterr().out
    assert 'already stopped early' in out
    assert 'Epoch: 6/6' in out and 'Epoch: 3/6' not in out


def test_fresh_run_replaces_stale_checkpoints(data_dir, tmp_path, capsys):
    stale = model_utils.build_model('resnet50', [16], num_classes=3, pretrained=False)
    model_utils.save_checkpoint(stale, [16], 3, 0.5, {'1': 0, '10': 1, '2': 2}, save_dir=str(tmp_path))
    # No epochs at all: nothing is trained, so the old run's checkpoint must not be reported as this run's
    with pytest.raises(SystemExit, match='No checkpoint was saved'):
        train.main([str(data_dir), '--save_dir', str(tmp_path), '--epochs', '0'] + FAST)
    assert 'replacing the previous' in capsys.readouterr().out
    assert not (tmp_path / train.BEST_CHECKPOINT).exists()


def test_resume_does_not_duplicate_history(data_dir, tmp_path):
    train.main([str(data_dir), '--save_dir', str(tmp_path), '--epochs', '2'] + FAST)
    # Simulate an interruption after epoch 3 was logged but before its checkpoint was saved
    with open(tmp_path / train.HISTORY_CSV, 'a', newline='') as f:
        csv.writer(f).writerow([3, 'head', 1, 1, 0, 0.001, 1])
    train.main([str(data_dir), '--save_dir', str(tmp_path), '--resume', str(tmp_path / train.LAST_CHECKPOINT),
                '--epochs', '3'] + FAST)
    with open(tmp_path / train.HISTORY_CSV, newline='') as f:
        assert [row['epoch'] for row in csv.DictReader(f)] == ['1', '2', '3']


def test_only_last_checkpoint_carries_optimizer_state(trained):
    best = torch.load(trained / train.BEST_CHECKPOINT)
    last = torch.load(trained / train.LAST_CHECKPOINT)
    assert 'optimizer_state' not in best and 'scheduler_state' not in best
    assert 'optimizer_state' in last and 'scheduler_state' in last


def test_evaluate_leaves_model_in_eval_mode(trained, data_dir):
    model = model_utils.load_checkpoint(trained / train.BEST_CHECKPOINT)
    _, _, _, test_loader = train.load_data(str(data_dir), 4, num_workers=0)
    train.evaluate(model, test_loader, nn.NLLLoss(), torch.device('cpu'))
    assert not model.training


def test_model_card_names_the_evaluated_split(trained, data_dir, tmp_path):
    report = tmp_path / 'report'
    evaluate.main([str(data_dir), str(trained / train.BEST_CHECKPOINT), '--split', 'valid', '--tta',
                   '--output_dir', str(report), '--num_workers', '0', '--gallery', '0'])
    summary = json.loads((report / 'summary.json').read_text())
    assert summary['split'] == 'valid' and summary['tta'] is True

    preview = tmp_path / 'preview'
    publish_to_hub.main([str(trained / train.BEST_CHECKPOINT), '--report_dir', str(report),
                         '--dry_run', str(preview)])
    card = (preview / 'README.md').read_text()
    assert 'Test accuracy' not in card
    assert '| Validation accuracy (TTA) |' in card and 'name: Validation accuracy (TTA)' in card


def test_old_reports_are_not_labelled_test_accuracy(tmp_path):
    (tmp_path / 'per_class_accuracy.csv').write_text('class,name,images,correct,accuracy\n1,a,4,3,0.75\n')
    assert publish_to_hub.report_accuracy(str(tmp_path)) == (0.75, 'Evaluation accuracy')
    assert publish_to_hub.report_accuracy(None) == (None, None)


def test_demo_examples_skip_stray_files(tmp_path):
    test_dir = tmp_path / 'test'
    (test_dir / '1').mkdir(parents=True)
    (test_dir / '1' / 'a.jpg').write_bytes(b'')
    (test_dir / '1' / 'notes.txt').write_text('not an image')
    (test_dir / '.DS_Store').write_text('')
    (test_dir / 'README').write_text('')
    (test_dir / '2').mkdir()
    assert app.find_examples(str(test_dir)) == [[str(test_dir / '1' / 'a.jpg')]]
    assert app.find_examples(str(tmp_path / 'missing')) is None
