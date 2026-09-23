import csv
import json

import pytest
import torch
from torch import nn

import app
import evaluate
import export
import model_utils
import train

FAST = ['--batch_size', '4', '--num_workers', '0', '--seed', '0']


@pytest.fixture(scope='module')
def trained(data_dir, tmp_path_factory):
    ''' One small trained run (2 classifier epochs + 1 fine-tuning epoch) shared by the tests below. '''
    save_dir = tmp_path_factory.mktemp('trained')
    original = model_utils._load_pretrained
    model_utils._load_pretrained = lambda arch, pretrained=True: original(arch, False)
    try:
        train.main([str(data_dir), '--save_dir', str(save_dir), '--epochs', '2', '--finetune_epochs', '1',
                    '--hidden_units', '64', '32'] + FAST)
    finally:
        model_utils._load_pretrained = original
    return save_dir


def test_hidden_units_are_configurable():
    model = model_utils.build_model('densenet121', [64, 32], num_classes=5, pretrained=False)
    sizes = [layer.out_features for layer in model.classifier if isinstance(layer, nn.Linear)]
    assert sizes == [64, 32, 5]
    single = model_utils.build_model('resnet50', 128, num_classes=5, pretrained=False)
    assert [m.out_features for m in single.fc if isinstance(m, nn.Linear)] == [128, 5]


def test_old_checkpoints_still_load(tmp_path):
    ''' Checkpoints from before --hidden_units took a list stored only hidden_layer1 (then 90 and 80). '''
    model = model_utils.build_model('densenet121', [256, 90, 80], num_classes=3, pretrained=False)
    path = tmp_path / 'old.pt'
    torch.save({'structure': 'densenet121', 'hidden_layer1': 256, 'num_classes': 3, 'dropout': 0.5,
                'state_dict': model.state_dict(), 'class_to_idx': {'1': 0, '10': 1, '2': 2}}, path)
    loaded = model_utils.load_checkpoint(path)
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(loaded(x), model(x))


def test_label_smoothing_loss_on_log_probabilities():
    ''' CrossEntropyLoss on log-probabilities equals NLLLoss, so label smoothing can be layered on top. '''
    log_probs = torch.log_softmax(torch.randn(8, 5), dim=1)
    labels = torch.randint(0, 5, (8,))
    assert torch.allclose(nn.CrossEntropyLoss()(log_probs, labels), nn.NLLLoss()(log_probs, labels))


def test_mixed_precision_is_off_on_cpu():
    assert isinstance(train.autocast(torch.device('cpu'), True), type(train.contextlib.nullcontext()))
    assert not train.make_grad_scaler(False).is_enabled()


def test_finetuning_updates_only_the_last_block(trained, data_dir, tmp_path):
    # Same seed, classifier only: gives the starting weights the fine-tuning run had
    head_only = tmp_path / 'head_only'
    train.main([str(data_dir), '--save_dir', str(head_only), '--epochs', '2',
                '--hidden_units', '64', '32'] + FAST)
    before = torch.load(head_only / train.LAST_CHECKPOINT)['state_dict']
    after = torch.load(trained / train.LAST_CHECKPOINT)
    assert after['phase'] == 'finetune' and after['epoch'] == 3
    last_block = 'features.denseblock4.denselayer16.conv2.weight'
    assert not torch.equal(before[last_block], after['state_dict'][last_block])
    assert torch.equal(before['features.conv0.weight'], after['state_dict']['features.conv0.weight'])


def test_history_csv_and_plot(trained):
    with open(trained / train.HISTORY_CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    assert [(r['epoch'], r['phase']) for r in rows] == [('1', 'head'), ('2', 'head'), ('3', 'finetune')]
    assert rows[2]['learning_rate'] == '0.0001'
    assert (trained / train.HISTORY_PLOT).stat().st_size > 0


def test_resume_during_finetuning(trained, data_dir, tmp_path, capsys):
    save_dir = tmp_path / 'resumed'
    save_dir.mkdir()
    for name in (train.BEST_CHECKPOINT, train.LAST_CHECKPOINT, train.HISTORY_CSV):
        (save_dir / name).write_bytes((trained / name).read_bytes())
    capsys.readouterr()
    train.main([str(data_dir), '--save_dir', str(save_dir), '--resume', str(save_dir / train.LAST_CHECKPOINT),
                '--epochs', '2', '--finetune_epochs', '2'] + FAST)
    out = capsys.readouterr().out
    assert 'after epoch 3 (finetune phase)' in out
    assert 'Epoch: 4/4' in out and 'Epoch: 1/4' not in out
    assert torch.load(save_dir / train.LAST_CHECKPOINT)['epoch'] == 4
    with open(save_dir / train.HISTORY_CSV, newline='') as f:
        assert [r['epoch'] for r in csv.DictReader(f)] == ['1', '2', '3', '4']


def test_build_report():
    idx_to_class = {0: 'a', 1: 'b', 2: 'c'}
    truths = [0, 0, 0, 1, 1, 2]
    predictions = [0, 1, 1, 1, 1, 0]
    per_class, confused = evaluate.build_report(truths, predictions, idx_to_class, {'a': 'Aster'})
    assert [(r['name'], r['correct'], r['images']) for r in per_class] == [('c', 0, 1), ('Aster', 1, 3),
                                                                          ('b', 2, 2)]
    assert [(r['true_name'], r['predicted_name'], r['count']) for r in confused] == [('Aster', 'b', 2),
                                                                                    ('c', 'Aster', 1)]


def test_evaluate_script(trained, data_dir, tmp_path):
    overall, per_class, _ = evaluate.main([str(data_dir), str(trained / train.BEST_CHECKPOINT),
                                           '--output_dir', str(tmp_path), '--num_workers', '0'])
    assert 0 <= overall <= 1 and len(per_class) == 3
    assert (tmp_path / 'per_class_accuracy.csv').exists() and (tmp_path / 'confused_pairs.csv').exists()


def test_export_torchscript(trained, tmp_path):
    export.main([str(trained / train.BEST_CHECKPOINT), '--output_dir', str(tmp_path), '--name', 'm'])
    files = {'class_to_idx.json': ''}
    scripted = torch.jit.load(str(tmp_path / 'm.torchscript.pt'), _extra_files=files)
    assert json.loads(files['class_to_idx.json'])['labels'] == ['1', '10', '2']

    model = model_utils.load_checkpoint(trained / train.BEST_CHECKPOINT)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(scripted(x), torch.exp(model(x)), atol=1e-5)
    labels = json.loads((tmp_path / 'm.labels.json').read_text())
    assert labels['output'] == 'probabilities' and labels['input']['center_crop'] == 224


def test_export_onnx(trained, tmp_path):
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnx')
    export.main([str(trained / train.BEST_CHECKPOINT), '--output_dir', str(tmp_path), '--name', 'm',
                 '--format', 'onnx'])
    session = ort.InferenceSession(str(tmp_path / 'm.onnx'))
    model = model_utils.load_checkpoint(trained / train.BEST_CHECKPOINT)
    x = torch.randn(2, 3, 224, 224)
    (probs,) = session.run(None, {'image': x.numpy()})
    with torch.no_grad():
        assert torch.allclose(torch.from_numpy(probs), torch.exp(model(x)), atol=1e-4)


def test_demo_classifier(trained, data_dir):
    model = model_utils.load_checkpoint(trained / train.BEST_CHECKPOINT)
    classify = app.make_classifier(model, torch.device('cpu'), {'10': 'globe thistle'}, top_k=3)
    result = classify(str(data_dir / 'test' / '10' / 'img0.jpg'))
    assert len(result) == 3 and abs(sum(result.values()) - 1) < 1e-4
    assert set(result) <= {'1', '2', 'globe thistle'}
    assert classify(None) == {}


def test_demo_interface_builds(trained):
    pytest.importorskip('gradio')
    demo = app.build_demo(lambda path: {}, top_k=5)
    assert demo is not None
