###################################################################################################################
# Web demo: upload a flower photo and see the top 5 predictions
# Notes - Run train.py first, then install the demo dependency: pip install -r requirements-demo.txt
# Basic usage: python app.py --checkpoint check_point.pt
# Options:
# Use flower names: python app.py --checkpoint check_point.pt --category_names cat_to_name.json (the default)
# Share a temporary public link: python app.py --checkpoint check_point.pt --share
# Use a model published with publish_to_hub.py: python app.py --hub_repo your-name/flower-classifier
# The checkpoint / Hub repository can also be set with the CHECKPOINT / HUB_REPO environment variables
# (handy on Hugging Face Spaces and in Docker)
#####################################################################################################################
import argparse
import json
import os

from model_utils import find_images, get_device, load_checkpoint, predict


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Web demo for the flower classifier.')
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', 'check_point.pt'),
                        help='checkpoint created by train.py (default: $CHECKPOINT or check_point.pt)')
    parser.add_argument('--hub_repo', default=os.environ.get('HUB_REPO'),
                        help='download check_point.pt from this Hugging Face Hub repository instead '
                             '(default: $HUB_REPO)')
    parser.add_argument('--category_names', default='cat_to_name.json',
                        help='JSON file mapping categories to real names (default: cat_to_name.json)')
    parser.add_argument('--top_k', default=5, type=int, help='number of predictions to show (default: 5)')
    parser.add_argument('--gpu', dest='use_gpu', action='store_true', default=False,
                        help='Use GPU for inference (default: False)')
    parser.add_argument('--share', action='store_true', default=False,
                        help='create a temporary public link to the demo')
    parser.add_argument('--port', default=None, type=int, help='port to serve on (default: 7860)')
    return parser.parse_args(argv)


def make_classifier(model, device, cat_to_name=None, top_k=5):
    ''' Returns a function mapping an image file path to {flower name: probability}. '''
    cat_to_name = cat_to_name or {}

    def classify(image_path):
        if image_path is None:
            return {}
        probs, classes = predict(image_path, model, top_k, device)
        return {cat_to_name.get(cls, cls): prob for cls, prob in zip(classes, probs)}

    return classify


def find_examples(example_dir, count=4):
    ''' A few test images offered as one-click examples when the dataset is next to the app (else None). '''
    if not os.path.isdir(example_dir):
        return None
    examples = []
    for cls in sorted(os.listdir(example_dir)):
        class_dir = os.path.join(example_dir, cls)
        if not os.path.isdir(class_dir):
            continue  # e.g. .DS_Store or a README
        images = find_images(class_dir)
        if images:
            examples.append([images[0]])
        if len(examples) == count:
            break
    return examples or None


def build_demo(classify, top_k=5, examples=None):
    import gradio as gr

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type='filepath', label='Flower photo'),
        outputs=gr.Label(num_top_classes=top_k, label='Top predictions'),
        title='Flower Classifier',
        description='Upload a photo of a flower to see which of the 102 species the model thinks it is.',
        examples=examples,
        flagging_mode='never',
    )


def main(argv=None):
    args = get_args(argv)
    if args.hub_repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise SystemExit('--hub_repo needs the huggingface_hub package: pip install huggingface_hub') from None
        args.checkpoint = hf_hub_download(args.hub_repo, 'check_point.pt')
    if not os.path.exists(args.checkpoint):
        raise SystemExit('Checkpoint {} not found. Train a model with train.py first.'.format(args.checkpoint))
    device = get_device(args.use_gpu)
    model = load_checkpoint(args.checkpoint, device)

    cat_to_name = None
    if args.category_names and os.path.exists(args.category_names):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    examples = find_examples(os.path.join('flowers', 'test'))
    classify = make_classifier(model, device, cat_to_name, args.top_k)
    build_demo(classify, args.top_k, examples).launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()
