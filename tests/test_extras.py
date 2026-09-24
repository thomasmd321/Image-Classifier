import io
import json
import pathlib
import subprocess
import sys
import tarfile

import pytest
import torch

import app
import download_data
import evaluate
import model_utils
import predict
import publish_to_hub
import train

FAST = ['--batch_size', '4', '--num_workers', '0', '--seed', '0']


def make_archive(path, prefix='flower_data/', extra=None):
    ''' A tiny .tar.gz shaped like the real dataset archive. '''
    with tarfile.open(path, 'w:gz') as archive:
        for split in download_data.SPLITS:
            for cls in ('1', '2'):
                data = b'not really a jpeg'
                info = tarfile.TarInfo('{}{}/{}/img.jpg'.format(prefix, split, cls))
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        for name in extra or []:
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b'x'))


def test_download_data_extracts_and_skips(tmp_path, capsys):
    archive = tmp_path / 'flower_data.tar.gz'
    make_archive(archive)
    target = tmp_path / 'flowers'
    download_data.main(['--data_dir', str(target), '--url', archive.as_uri()])
    assert sorted(p.name for p in target.iterdir()) == ['test', 'train', 'valid']
    assert (target / 'train' / '2' / 'img.jpg').exists()
    assert 'Dataset ready' in capsys.readouterr().out
    # Only the dataset is left behind, no temporary files
    assert sorted(p.name for p in tmp_path.iterdir()) == ['flower_data.tar.gz', 'flowers']

    download_data.main(['--data_dir', str(target), '--url', archive.as_uri()])
    assert 'nothing to do' in capsys.readouterr().out


def test_download_data_rejects_unsafe_archives(tmp_path):
    archive = tmp_path / 'evil.tar.gz'
    make_archive(archive, extra=['../outside.txt'])
    with pytest.raises(SystemExit, match='outside'):
        download_data.main(['--data_dir', str(tmp_path / 'flowers'), '--url', archive.as_uri()])
    assert not (tmp_path / 'outside.txt').exists()


def test_tta_averages_image_and_mirror():
    model = model_utils.build_model('densenet121', [32], num_classes=4, pretrained=False).eval()
    images = torch.randn(2, 3, 224, 224)
    plain = model_utils.predict_probs(model, images)
    mirrored = model_utils.predict_probs(model, torch.flip(images, dims=[3]))
    tta = model_utils.predict_probs(model, images, tta=True)
    assert torch.allclose(tta, (plain + mirrored) / 2, atol=1e-6)
    assert torch.allclose(tta.sum(dim=1), torch.ones(2), atol=1e-5)


def test_predict_with_tta(trained, data_dir):
    results = predict.main([str(data_dir / 'test' / '10' / 'img0.jpg'), str(trained / train.BEST_CHECKPOINT),
                            '--tta', '--top_k', '2'])
    assert len(results[0][1]) == 2


def test_evaluate_charts_and_tta(trained, data_dir, tmp_path, capsys):
    evaluate.main([str(data_dir), str(trained / train.BEST_CHECKPOINT), '--output_dir', str(tmp_path),
                   '--num_workers', '0', '--tta', '--gallery', '4'])
    assert 'accuracy with TTA' in capsys.readouterr().out
    assert (tmp_path / 'confusion_matrix.png').stat().st_size > 0
    # The untrained model gets most of the 12 test images wrong, so the gallery has mistakes to show
    assert (tmp_path / 'misclassified.png').stat().st_size > 0


def test_gallery_skipped_when_everything_is_right(tmp_path):
    assert evaluate.plot_gallery([], [0, 1], [0, 1], [0.9, 0.8], ['a', 'b'], str(tmp_path / 'g.png')) is None


def test_tensorboard_logging(data_dir, tmp_path):
    pytest.importorskip('tensorboard')
    train.main([str(data_dir), '--save_dir', str(tmp_path), '--epochs', '1', '--tensorboard'] + FAST)
    assert list((tmp_path / 'runs').rglob('events.out.tfevents.*'))


def test_publish_dry_run(trained, data_dir, tmp_path):
    report = tmp_path / 'report'
    evaluate.main([str(data_dir), str(trained / train.BEST_CHECKPOINT), '--output_dir', str(report),
                   '--num_workers', '0'])
    preview = tmp_path / 'preview'
    publish_to_hub.main([str(trained / train.BEST_CHECKPOINT), '--report_dir', str(report),
                         '--repo_id', 'someone/flowers', '--dry_run', str(preview)])
    names = {p.name for p in preview.iterdir()}
    assert {'check_point.pt', 'labels.json', 'README.md', 'history.png', 'confusion_matrix.png'} <= names
    card = (preview / 'README.md').read_text()
    assert card.startswith('---\nlibrary_name: pytorch')
    assert 'Test accuracy' in card and "hf_hub_download('someone/flowers', 'check_point.pt')" in card
    assert '| Epochs trained | 3 (1 fine-tuning) |' in card
    # The published checkpoint is a working copy
    assert model_utils.load_checkpoint(preview / 'check_point.pt').class_to_idx == {'1': 0, '10': 1, '2': 2}


def test_publish_uploads_folder(trained, monkeypatch):
    hub = pytest.importorskip('huggingface_hub')
    calls = {}

    class FakeApi:
        def create_repo(self, repo_id, private=False, exist_ok=False):
            calls['create'] = (repo_id, private, exist_ok)

        def upload_folder(self, repo_id, folder_path, commit_message):
            calls['files'] = sorted(p.name for p in pathlib.Path(folder_path).iterdir())

    monkeypatch.setattr(hub, 'HfApi', FakeApi)
    url = publish_to_hub.main([str(trained / train.BEST_CHECKPOINT), '--repo_id', 'someone/flowers', '--private'])
    assert url == 'https://huggingface.co/someone/flowers'
    assert calls['create'] == ('someone/flowers', True, True)
    assert 'check_point.pt' in calls['files'] and 'README.md' in calls['files']


def test_publish_requires_repo_or_dry_run(trained):
    with pytest.raises(SystemExit):
        publish_to_hub.main([str(trained / train.BEST_CHECKPOINT)])


def test_app_loads_model_from_hub(trained, monkeypatch):
    hub = pytest.importorskip('huggingface_hub')
    requested = []

    def fake_download(repo_id, filename):
        requested.append((repo_id, filename))
        return str(trained / train.BEST_CHECKPOINT)

    launched = {}

    class FakeDemo:
        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr(hub, 'hf_hub_download', fake_download)
    monkeypatch.setattr(app, 'build_demo', lambda classify, top_k, examples: FakeDemo())
    monkeypatch.setattr(sys, 'argv', ['app.py'])
    app.main(['--hub_repo', 'someone/flowers', '--port', '7999'])
    assert requested == [('someone/flowers', 'check_point.pt')]
    assert launched['server_port'] == 7999


def test_colab_notebook_uses_real_options():
    ''' The Colab notebook calls the scripts with options that must exist. '''
    root = pathlib.Path(__file__).resolve().parent.parent
    cells = json.loads((root / 'colab_train.ipynb').read_text())['cells']
    source = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    assert 'python download_data.py' in source
    for script, flags in (('train.py', ['--gpu', '--arch', '--epochs', '--finetune_epochs', '--seed',
                                        '--num_workers', '--save_dir', '--resume']),
                          ('evaluate.py', ['--category_names', '--tta', '--output_dir', '--num_workers', '--gpu']),
                          ('predict.py', ['--category_names', '--top_k', '--plot_dir', '--gpu'])):
        assert 'python {}'.format(script) in source
        usage = subprocess.run([sys.executable, script, '--help'], cwd=root, capture_output=True, text=True,
                               check=True).stdout
        missing = [flag for flag in flags if flag not in usage]
        assert not missing, '{} has no {}'.format(script, missing)
