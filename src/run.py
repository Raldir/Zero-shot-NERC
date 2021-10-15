""" Queue up training of multiple models by using config files"""
import os
import argparse
import logging
from argparse import Namespace
from train import run
from test import run_test
import json
import sys
import shutil
import torch
import random

logger = logging.getLogger(__name__)


def load_hyperparameters_from_file(dir, config_files, overwrites):
    print(config_files)
    args_dict = {}
    if config_files:
        for file in config_files:
            with open(os.path.join(dir,file + ".json"), 'r') as f:
                hyperparameter_dict = json.load(f)
                hyperparameter_dict.update(overwrites)
            args = argparse.Namespace(**hyperparameter_dict)
            args_dict[file] = args
    else:
        for file in os.listdir(dir):
            with open(os.path.join(dir,file), 'r') as f:
                hyperparameter_dict = json.load(f)
                hyperparameter_dict.update(overwrites)
            args = argparse.Namespace(**hyperparameter_dict)
            args_dict[file] = args
    return args_dict

def main():
    parser = argparse.ArgumentParser(description="Args for program")
    parser.add_argument('--config_mode', type=str, help="Select mode")
    parser.add_argument('--config_files', type=str, help="Select mode")
    parser.add_argument('--description', type=str, help="Add additional description to output_dir")
    parser.add_argument('--overwrites', type=str, help="Select mode")
    config_args = parser.parse_args()

    overwrites_dict = {}
    if config_args.overwrites:
        overwrites = config_args.overwrites.split('[')[1].split(']')[0].split(',')
        for overwrite in overwrites:
            overwrites_dict[overwrite.split(":")[0]] = overwrite.split(":")[1]
            if overwrites_dict[overwrite.split(":")[0]].isdigit():
                overwrites_dict[overwrite.split(":")[0]] = int(overwrite.split(":")[1])
            elif overwrites_dict[overwrite.split(":")[0]] == 'true':
                overwrites_dict[overwrite.split(":")[0]] = True
            elif overwrites_dict[overwrite.split(":")[0]] == 'false':
                overwrites_dict[overwrite.split(":")[0]] = False
    if not os.path.exists(os.path.join('..', 'configs', config_args.config_mode)):
        print("Config folder not found. Abort...")
        sys.exit()
    else:
        args_dicts = load_hyperparameters_from_file(os.path.join('..', 'configs', config_args.config_mode), config_args.config_files.split('[')[1].split(']')[0].split(',') if config_args.config_files else None, overwrites_dict)
    #scores = {'acc': 0., 'f1' : 0., "recall": 0., "precision": 0.}
    max_f1 = 0
    best_arguments = None
    id = 0
    f1_test = 0

    for file, args in args_dicts.items():
        args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.output_dir = os.path.join('..', 'dumped', config_args.config_mode +  "_" + file.split(".")[0] + "_" + config_args.overwrites.split('[')[1].split(']')[0].replace(":", "_").replace(",","__")) + "__" + config_args.description if config_args.description else os.path.join('..', 'dumped', config_args.config_mode + "_" + file.split(".")[0] + "_" +  config_args.overwrites.split('[')[1].split(']')[0].replace(":", "_").replace(",","__"))
        if os.path.exists(args.output_dir) and not args.continue_training:
                print('Output path exists. Remove existing dir.')
                shutil.rmtree(args.output_dir, ignore_errors=True)
        params = vars(args)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
            for key, value in params.items():
                f.write("{} : {}\n".format(str(key), str(value)))

        logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s',datefmt='%H:%M:%S', level = logging.INFO, handlers=[ logging.FileHandler(os.path.join(args.output_dir, 'logs.txt')), logging.StreamHandler(sys.stdout)])
        logging.info('Program arguments: %s', args)

        f1 = run(args)

        if f1 > max_f1:
            max_f1 = f1
            best_arguments = args
        id +=1

        if args.dataset == 'ontonotes':
            args.split = 'conll-2012-test'
        else:
            args.split = 'test'
        args.checkpoint = 'checkpoint'
        args.add_boundaries = True

        f1_test = run_test(args)
        logging.info('Seed: {}'.format(args.seed))


    logging.info('Best program arguments: %s', best_arguments)
    logging.info("Best scores : {}".format(max_f1))
    logging.info("Best scores test : {}".format(f1_test))
    file_out = '../final_results_new.txt'
    with open(file_out, 'a') as f:
        f.write('Best program arguments: {}\n'.format(best_arguments))
        f.write('Best scores : {}\n'.format(max_f1))
        f.write("Best scores test : {}\n\n".format(f1_test))



if __name__ == '__main__':
    main()
