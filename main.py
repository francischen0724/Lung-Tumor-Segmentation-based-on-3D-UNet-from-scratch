# main.py

import sys
import argparse
import os
from utils import preprocess
from experiments.inf_seg import Inference
from experiments.inf_class import Inf_class
from experiments.train_seg_bbx import train_seg
#from experiments.train_seg_RAseg import train_seg


def path(string):
    """Check if the path exists."""
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'❌ Path not found: {string}')

def main():
    parser = argparse.ArgumentParser(
        description="Main entry point for medical image segmentation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'process',
        type=str,
        choices=['preprocess', 'train_seg', 'inf_seg', 'inf_class'],
        help="""Task to run:
- preprocess: Run preprocessing on raw datasets (nsclc, msd, rad, radgen)
- train_seg: Train segmentation model using split data
- inf_seg: Run segmentation inference
- inf_class: Run classification inference
"""
    )
    parser.add_argument(
        'input',
        type=path,
        help='Path to the input dataset (e.g., datasets_XW/split/NSCLC)'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Path to output folder (e.g., output/NSCLC)'
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='Name of dataset/model tag (e.g., NSCLC, MSD)'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "-D", "--debug",
        default=False,
        action="store_true",
        help="Run in debug mode (minimal steps, extra logs)"
    )

    args = parser.parse_args()

    if args.process == "preprocess":
        data_preprocess = preprocess.Preprocess(args.input, args.output, args.dataset)
        if args.dataset == "radgen":
            data_preprocess.radiogenomics()
        elif args.dataset == "rad":
            data_preprocess.radiomics()
        elif args.dataset == "msd":
            data_preprocess.msd()
        else:
            sys.exit("❌ Unknown dataset type for preprocessing.")
    
    elif args.process == "train_seg":
        train_seg(args)

    elif args.process == "inf_seg":
        inference = Inference(args.input, args.output, args.dataset)
        inference.run()

    elif args.process == "inf_class":
        inf_class = Inf_class(args.input, args.output, args.dataset)
        inf_class.run()

    else:
        sys.exit("❌ Unknown process command. Use --help to see valid options.")

if __name__ == "__main__":
    main()
