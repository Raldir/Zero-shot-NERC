""" Adapt MedMentions or OntoNotes datastet to zero-shot domain"""
import os
import argparse
from preprocess.ontonotes_dataset import create_zero_shot_ontonotes_dataset
from preprocess.medmentions_dataset import preprocess_medmentions, alternative_splits_medmentions, alternative_splits_medmentions_extreme, create_zero_shot_medmentions_dataset


def main(args):
    if args.dataset == 'ontonotes':
        create_zero_shot_ontonotes_dataset(args)
    elif args.dataset == 'medmentions':
        if args.mode == 'preprocess':
            preprocess_medmentions(args)
        elif args.mode == 'custom_splits':
            alternative_splits_medmentions(args)
        elif args.mode == 'zero_shot':
            create_zero_shot_medmentions_dataset(args)
        elif args.mode =="custom_splits_extreme":
            alternative_splits_medmentions_extreme(args)

def parseargs():
    parser = argparse.ArgumentParser(description="Args for program")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--mode', type=str, help="Select statistic")
    parser.add_argument('--input_path', default = '../data/OntoNotes_5.0_NER_BIO/conll-ontonotes-2012-wo-nt',  help = "Input path")
    parser.add_argument('--output_path', default = '../data/OntoNotes_5.0_NER_BIO/conll-ontonotes-2012-zero-shot',  help = "Input path")
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    parseargs()
