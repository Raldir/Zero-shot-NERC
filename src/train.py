"""Training pipeline"""

import os
import argparse
from utils.readers import read_data_medmentions, read_data_ontonotes
from models.entity_descriptions import get_entity_descriptions,  get_entity_descriptions_negative
from models.trainer import Trainer
from models.dataloader import InputFeatures, EntityRecognitionDataset, EntityRecognitionMultiClassDataset,  build_vocab
import torch
import sys
import logging
logger = logging.getLogger(__name__)

def run(args):
    #Load the OntoNotes-ZS or MedMentions-ZS in BIO format
    logger.info("READ DATA: {}".format(args.dataset))
    if args.dataset == 'ontonotes':
        data_train = read_data_ontonotes(args.input_path, 'train')
        data_dev = read_data_ontonotes(args.input_path, 'dev')
    elif args.dataset == 'medmentions':
        data_train = read_data_medmentions(args.input_path, 'train')
        data_dev = read_data_medmentions(args.input_path, 'dev')

    logger.info("USING ENTITY DESCRIPTION: {}".format( args.entity_descriptions_mode))
    #Loads the descriptions inclusive negative description if we are in tagging and multiclass mode
    if 'multiclass' in args.mode and 'tagger' in args.mode:
        entity_labels, entity_descriptions = get_entity_descriptions_negative(args.entity_descriptions_mode, args.dataset, 'train',  filter_classes = 'filtered_classes' in args.mode)
        entity_labels_dev, entity_descriptions_dev = get_entity_descriptions_negative(args.entity_descriptions_mode, args.dataset, 'dev',  filter_classes = 'filtered_classes' in args.mode)
        entity_labels_test, entity_descriptions_test = get_entity_descriptions_negative(args.entity_descriptions_mode, args.dataset, 'conll-2012-test',  filter_classes = 'filtered_classes' in args.mode)
    else:
        entity_labels, entity_descriptions = get_entity_descriptions(args.entity_descriptions_mode, args.dataset, 'train', filter_classes = 'filtered_classes' in args.mode)
        entity_labels_dev, entity_descriptions_dev = get_entity_descriptions(args.entity_descriptions_mode, args.dataset, 'dev',  filter_classes = 'filtered_classes' in args.mode)
        entity_labels_test, entity_descriptions_test = get_entity_descriptions(args.entity_descriptions_mode, args.dataset, 'conll-2012-test',  filter_classes = 'filtered_classes' in args.mode)

    #Load the dataloader given the task
    if args.mode in ['known_boundaries', 'known_boundaries_filtered_classes', 'known_boundaries_difficult_classes']:
        dataloader = EntityClassificationDataset(args, 'train', data_train, entity_labels, entity_descriptions)
        dataloader_dev = EntityClassificationDataset(args, 'dev', data_dev, entity_labels_dev, entity_descriptions_dev)
    elif args.mode in ['known_boundaries_multiclass', 'known_boundaries_multiclass_filtered_classes']:
        dataloader = EntityClassificationMultiClassDataset(args, 'train', data_train, entity_labels, entity_descriptions)
        dataloader_dev = EntityClassificationMultiClassDataset(args, 'dev', data_dev, entity_labels_dev, entity_descriptions_dev)
    elif args.mode in ['tagger', 'tagger_filtered_classes']:
        dataloader = EntityRecognitionDataset(args, 'train', data_train, entity_labels, entity_descriptions)
        dataloader_dev = EntityRecognitionDataset(args, 'dev', data_dev, entity_labels_dev, entity_descriptions_dev)
    elif args.mode in ['tagger_multiclass', 'tagger_multiclass_filtered_classes']:
        dataloader = EntityRecognitionMultiClassDataset(args, 'train', data_train, entity_labels, entity_descriptions)
        dataloader_dev = EntityRecognitionMultiClassDataset(args, 'dev', data_dev, entity_labels_dev, entity_descriptions_dev)

    trainer = Trainer(args, dataloader, dataloader_dev)
    step, loss, validation_score = trainer.train()
    return validation_score


def parseargs():
    parser = argparse.ArgumentParser(description="Args for program")
    #General program settings and dataset setting
    parser.add_argument('--mode', type=str, help="Select mode")
    parser.add_argument('--model', type=str, help="Select model")
    parser.add_argument('--dataset', type=str, help="Select statistic")
    parser.add_argument('--input_path', default = '../data/OntoNotes_5.0_NER_BIO/conll-ontonotes-paper-wo-nt-zero-shot',  help = "Input path")
    parser.add_argument('--entity_descriptions_mode', default = 'wordnet',  help = "Input path")

    #Loading settings,
    parser.add_argument('--limit', type = int, help = "Input path")
    parser.add_argument('--continue_training', action="store_true", help="Select model")
    parser.add_argument( "--output_dir", default=None, type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument( "--checkpoint", default=None, type=str, help="The checkpoint from which to continue training.")

    #Transformer settings entity classification
    parser.add_argument("--add_boundaries", action="store_true", help="Mask the entity in the text to assist generalization")
    parser.add_argument("--add_negative_samples", action="store_true", help="Mask the entity in the text to assist generalization")

    #Transformer settings entity recognition
    parser.add_argument("--add_iob", action="store_true", help="Mask the entity in the text to assist generalization")
    parser.add_argument("--num_classes", default=4, type=int, help="Number of classes for enttiy recognition")
    parser.add_argument("--only_keep_relevant_classes", action="store_true", help="Mask the entity in the text to assist generalization")


    #General task settings
    parser.add_argument("--max_description_length", default=100, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences")
    parser.add_argument("--max_sequence_length", default=512, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences")
    parser.add_argument("--mask_entity", action="store_true", help="Mask the entity in the text to assist generalization")
    parser.add_argument("--mask_probability", default=0.5, type=float, help="Dropout of linear layer.")


    #Model settings
    parser.add_argument('--model_type', type=str, help="Select model")
    parser.add_argument('--embedding_path', type=str, help="Specify embedding path", default = "../data/cc.en.300.vec")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.")
    parser.add_argument("--early_stopping", action="store_true", help="Run evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--scheduler", default='linear_scheduler', type=str, choices= ['linear_scheduler', "plateau_scheduler"], help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for eval.")
    parser.add_argument("--linear_dropout", default=0.2, type=float, help="Dropout of linear layer.")
    parser.add_argument("--linear_units_symbol", default=100, type=int, help="Dropout of linear layer.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.continue_training:
            print('Output path exists but continue training not activated. Abort...')
            sys.exit()

    if args.checkpoint and not os.path.exists(os.path.join(args.output_dir, args.checkpoint)):
            print('Warning. Wants to load checkpoint but does not exist...')
            print(os.path.exists(os.path.join(args.output_dir, args.checkpoint)))

    import json
    params = vars(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in params.items():
            f.write("{} : {}\n".format(str(key), str(value)))

    logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s',datefmt='%H:%M:%S', level = logging.INFO, handlers=[ logging.FileHandler(os.path.join(args.output_dir, 'logs.txt')), logging.StreamHandler(sys.stdout)])

    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    return args


if __name__ == '__main__':
    args = parseargs()
    run(args)
