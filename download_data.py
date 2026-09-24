###################################################################################################################
# Downloads the 102 Category Flower dataset used by this project and unpacks it into flowers/train, valid and test
# Basic usage: python download_data.py
# Options:
# Unpack somewhere else: python download_data.py --data_dir data/flowers
# Download again even if the data is already there: python download_data.py --force
#####################################################################################################################
import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

DATA_URL = 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
SPLITS = ('train', 'valid', 'test')


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Download and unpack the flowers dataset.')
    parser.add_argument('--data_dir', default='flowers',
                        help='where to put the train/, valid/ and test/ folders (default: flowers)')
    parser.add_argument('--url', default=DATA_URL, help='dataset archive URL (default: the Udacity copy)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='download again even if the data directory already has the splits')
    return parser.parse_args(argv)


def has_splits(path):
    return all(os.path.isdir(os.path.join(path, split)) for split in SPLITS)


def download(url, path):
    ''' Downloads url to path, printing progress. '''
    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write('\rDownloading {:5.1f}% of {:.0f} MB'.format(100 * done / total, total / 1e6))
            sys.stdout.flush()

    urllib.request.urlretrieve(url, path, reporthook=progress)
    print()


def safe_extract(archive_path, destination):
    ''' Extracts a .tar.gz archive, refusing any entry that would land outside destination. '''
    destination = os.path.realpath(destination)
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            target = os.path.realpath(os.path.join(destination, member.name))
            if os.path.commonpath([destination, target]) != destination:
                raise SystemExit('Refusing to extract {}: it points outside the target folder'.format(member.name))
            if member.issym() or member.islnk():
                raise SystemExit('Refusing to extract link {} from the archive'.format(member.name))
        archive.extractall(destination, members=members)


def find_split_root(path):
    ''' Returns the folder under path that contains train/, valid/ and test/. '''
    for root, dirs, _ in os.walk(path):
        if all(split in dirs for split in SPLITS):
            return root
    raise SystemExit('The archive does not contain train/, valid/ and test/ folders.')


def main(argv=None):
    args = get_args(argv)
    if has_splits(args.data_dir) and not args.force:
        print('{} already contains {}; nothing to do (use --force to download again).'.format(
            args.data_dir, ', '.join(SPLITS)))
        return args.data_dir

    parent = os.path.dirname(os.path.abspath(args.data_dir))
    os.makedirs(parent, exist_ok=True)
    # Work in a temporary folder next to the target, so a failed download never leaves half a dataset behind
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        archive_path = os.path.join(tmp, 'flower_data.tar.gz')
        download(args.url, archive_path)
        print('Unpacking ...')
        extracted = os.path.join(tmp, 'extracted')
        safe_extract(archive_path, extracted)
        root = find_split_root(extracted)

        os.makedirs(args.data_dir, exist_ok=True)
        for split in SPLITS:
            target = os.path.join(args.data_dir, split)
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.move(os.path.join(root, split), target)

    counts = {split: sum(len(files) for _, _, files in os.walk(os.path.join(args.data_dir, split)))
              for split in SPLITS}
    print('Dataset ready in {}: {}'.format(
        args.data_dir, ', '.join('{} {} images'.format(counts[s], s) for s in SPLITS)))
    return args.data_dir


if __name__ == '__main__':
    main()
