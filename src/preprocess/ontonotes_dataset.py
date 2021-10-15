import os
import sys
from utils.wrappers import Token
from utils.readers import read_data_ontonotes, get_zero_shot_splits
import matplotlib.pyplot as plt
import numpy as np


def write_zero_shot(data, types, output_path, file_name):
    output_file = os.path.join(output_path, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_file, "w") as f:
        for sentence in data:
            f.write("\n")
            for tok in sentence:
                #Not annotated parts are also replace with O which is fine. Removes unnecessary distinctuions
                f.write("{}\t{}\t{}\t{}\n".format(tok.word, tok.pos, tok.const, tok.type if len(tok.type.split('-')) > 1 and tok.type.split('-')[1] in types else "O"))
        f.write('\n')
        f.write('\n')
        f.write('\n')



def create_zero_shot_ontonotes_dataset(args):
    train_data = read_data_ontonotes(args.input_path, "train")
    dev_data = read_data_ontonotes(args.input_path, "dev")
    test_data = read_data_ontonotes(args.input_path, "conll-2012-test")
    train_types, dev_types, test_types = get_zero_shot_splits("ontonotes")
    write_zero_shot(train_data, train_types, args.output_path,  "onto.train.ner")
    write_zero_shot(dev_data, dev_types, args.output_path,  "onto.dev.ner")
    write_zero_shot(test_data, test_types, args.output_path,  "onto.conll-2012-test.ner")
