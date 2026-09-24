###################################################################################################################
# Uses a trained network to predict the class for an input image
# Notes - Run train.py first before this script
# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top K most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu
# Predict every image in a folder: python predict.py /path/to/folder checkpoint
# Save a chart of each prediction: python predict.py input checkpoint --plot_dir plots
# Test-time augmentation (slightly more accurate): python predict.py input checkpoint --tta
# Typical run:
#   python predict.py flowers/test/10/image_07104.jpg check_point.pt --gpu --category_names cat_to_name.json --top_k 3
#####################################################################################################################
import argparse
import json
import os

from model_utils import find_images, get_device, load_checkpoint, plot_prediction, predict


###########################################
# Get the arguments from the command line
###########################################
def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained checkpoint.')
    parser.add_argument('image_path', metavar='image_path',
                        help='/path/to/image, or a folder of images')
    parser.add_argument('checkpoint', metavar='checkpoint',
                        help='/path/to/checkpoint created by train.py')
    parser.add_argument('--category_names', dest='category_names', default=None,
                        help='JSON file mapping categories to real names (e.g. cat_to_name.json)')
    parser.add_argument('--top_k', metavar='top_k', default=3, type=int,
                        help='top K most likely classes (default: 3)')
    parser.add_argument('--plot_dir', metavar='plot_dir', default=None,
                        help='save an image + bar chart of each prediction into this folder')
    parser.add_argument('--tta', action='store_true', default=False,
                        help='test-time augmentation: average the predictions for the image and its mirror image')
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=False,
                        help='Use GPU for inference (default: False)')
    parser.add_argument('--version', action='version',
                        version='%(prog)s 1.3  There is NO warranty; not even for MERCHANTABILITY or '
                                'FITNESS FOR A PARTICULAR PURPOSE.')  # Decided to pull some wording from GCC
    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    device = get_device(args.use_gpu)

    image_paths = find_images(args.image_path)
    if not image_paths:
        raise SystemExit('No images found in {}'.format(args.image_path))

    model = load_checkpoint(args.checkpoint, device)

    cat_to_name = {}
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    results = []
    for image_path in image_paths:
        probs, classes = predict(image_path, model, args.top_k, device, tta=args.tta)
        names = [cat_to_name.get(cls, cls) for cls in classes]
        results.append((image_path, probs, names))

        print('Top {} predictions for {}:'.format(len(probs), image_path))
        for rank, (prob, name) in enumerate(zip(probs, names), start=1):
            print('{:>2}. {:<30} {:6.2f} %'.format(rank, name, 100 * prob))

        if args.plot_dir:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            out_path = plot_prediction(image_path, names, probs,
                                       os.path.join(args.plot_dir, stem + '.png'))
            print('    Chart saved to {}'.format(out_path))
    return results


if __name__ == '__main__':
    main()
