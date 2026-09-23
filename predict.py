###################################################################################################################
# Uses a trained network to predict the class for an input image
# Notes - Run train.py first before this script
# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top K most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu
# Typical run: python predict.py flowers/test/10/image_07104.jpg check_point.pt --gpu --category_names cat_to_name.json --top_k 3
#####################################################################################################################
import argparse
import json

from model_utils import get_device, load_checkpoint, predict


###########################################
# Get the arguments from the command line
###########################################
def get_args():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained checkpoint.')
    parser.add_argument('image_path', metavar='image_path',
                        help='/path/to/image')
    parser.add_argument('checkpoint', metavar='checkpoint',
                        help='/path/to/checkpoint created by train.py')
    parser.add_argument('--category_names', dest='category_names', default=None,
                        help='JSON file mapping categories to real names (e.g. cat_to_name.json)')
    parser.add_argument('--top_k', metavar='top_k', default=3, type=int,
                        help='top K most likely classes (default: 3)')
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=False,
                        help='Use GPU for inference (default: False)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.1  There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.')  # Decided to pull some wording from GCC
    return parser.parse_args()


def main():
    args = get_args()
    device = get_device(args.use_gpu)

    model = load_checkpoint(args.checkpoint, device)
    probs, classes = predict(args.image_path, model, args.top_k, device)

    cat_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    print('Top {} predictions for {}:'.format(args.top_k, args.image_path))
    for rank, (prob, cls) in enumerate(zip(probs, classes), start=1):
        name = cat_to_name.get(cls, cls) if cat_to_name else cls
        print('{:>2}. {:<30} {:6.2f} %'.format(rank, name, 100 * prob))


if __name__ == '__main__':
    main()
