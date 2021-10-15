""" Prepares the data given in BIO format for respective models and the respective task, i.e. transformer or LSTM"""
from transformers import BertTokenizer, DistilBertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor
import pickle
import os
import sys
import logging
from random import random
from models.entity_descriptions import get_impossible_classes
import math
from collections import Counter
import itertools
import numpy as np
logger = logging.getLogger(__name__)

class InputFeaturesBaseline(object):
    def __init__(self, text, label_id):
        self.text = text
        self.label_id= label_id

    def __str__(self):
        rep = '\n'.join(["Text: " + self.text, "label_id: " + str(self.label_id)])
        return rep


class InputFeatures(object):
    """A single set of features of data."""
    #Adjust for baseline model

    def __init__(self, text, label_id, input_ids = None, input_mask = None, segment_ids = None, entity_index = None, description_type = None, tagging = False, sep_index = None, entity_num = None):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity_index = entity_index
        self.description_type = description_type
        self.entity_num = entity_num
        self.tagging = tagging
        self.sep_index = sep_index

    def __str__(self):
        if not self.tagging:
            rep = '\n'.join(["Text: " + ' '.join(self.text), "input_ids: " + " ".join([str(x) for x in self.input_ids]), "input_mask: " + " ".join([str(x) for x in self.input_mask]), "segment_ids: " + " ".join([str(x) for x in self.segment_ids]), "label_id: " + str(self.label_id), "description_type: " + str(self.description_type), "entity_index: " + str(self.entity_index[0]) + "," + str(self.entity_index[1])])
        else:
            if self.input_mask:
                rep = '\n'.join(["Text: " + ' '.join(self.text), "input_ids: " + " ".join([str(x) for x in self.input_ids]), "input_mask: " + " ".join([str(x) for x in self.input_mask]), "segment_ids: " + " ".join([str(x) for x in self.segment_ids]), "label_id: " + " ".join(str(x) for x in self.label_id), "description_type: " + str(self.description_type), str("Sep index: " + str(self.sep_index))])
            else:
                rep = '\n'.join(["Text: " + ' '.join(self.text), "label_id: " + " ".join(str(x) for x in self.label_id), "description_type: " + str(self.description_type)])
        return rep


    def __repr__(self):
        return  " ".join(self.text)

class InputFeaturesMultiClass(object):
    """A single set of features of data."""

    def __init__(self, text_list, label_id, labels, label_dict = None, input_ids_list = None, input_mask_list = None, segment_ids_list = None, sep_index = None, entity_index = None, tagging = None):
        self.text_list = text_list
        self.input_ids_list = input_ids_list
        self.input_mask_list = input_mask_list
        self.segment_ids_list = segment_ids_list
        self.label_id = label_id
        self.labels = labels
        self.label_dict = label_dict
        self.sep_index = sep_index
        self.tagging = tagging
        self.entity_index = entity_index

    def __str__(self):
        rep = ""
        if self.tagging:
            for i in range(len(self.text_list)):
                rep += '\n'.join(["Text: " + ' '.join(self.text_list[i]), "input_ids: " + " ".join([str(x) for x in self.input_ids_list[i]]), "input_mask: " + " ".join([str(x) for x in self.input_mask_list[i]]), "segment_ids: " + " ".join([str(x) for x in self.segment_ids_list[i]])])
            rep += '\n'
            rep += '\n'.join(["label_id: " + " ".join(str(x) for x in self.label_id), "labels: " + " ".join(str(x) for x in self.labels),  'label_dict: ' + ','.join(str(x) + ":" + str(y) for x,y in self.label_dict.items()), str("Sep index: " + str(self.sep_index))])
        else:
            for i in range(len(self.text_list)):
                rep += '\n'.join(["Text: " + ' '.join(self.text_list[i]), "input_ids: " + " ".join([str(x) for x in self.input_ids_list[i]]), "input_mask: " + " ".join([str(x) for x in self.input_mask_list[i]]), "segment_ids: " + " ".join([str(x) for x in self.segment_ids_list[i]])])
            rep += '\n'
            rep += '\n'.join(["label_id: " + str(self.label_id), str("Sep index: " + str(self.sep_index)), "entity_index: " + str(self.entity_index[0]) + "," + str(self.entity_index[1])])
        return rep


    def __repr__(self):
        return  " ".join(self.text_list[0])

class EntityRecognitionMultiClassDataset(Dataset):

    def __init__(self, args, split, sentences, entity_labels, entity_descriptions, vocabulary = None, vocabulary_inv = None, all_labels = None):
        # self.tokenizer_name = 'bert-large-uncased'
        # self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.sentences = sentences
        self.entity_labels = entity_labels
        self.features = []
        self.limit = args.limit
        self.max_seq_length = args.max_sequence_length
        self.dataset = args.dataset
        self.mode = args.mode
        self.model = args.model
        self.split = split
        self.overwrite = args.overwrite_dumped
        self.output_dir = args.output_dir
        self.only_keep_relevant_classes = args.only_keep_relevant_classes
        self.mask_probability = 1 - args.mask_probability #0.3

        if args.model == 'transformer':
            self.tokenizer_name = 'bert-large-cased'
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
            self.mask_entity_partially = args.mask_entity_partially
            self.entity_descriptions = entity_descriptions
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
            self.max_description_length = args.max_description_length
            self.mask_entity_partially = args.mask_entity_partially
            self.description_mode = args.entity_descriptions_mode
            self.features = self.process_text(sentences, entity_labels, entity_descriptions)

        else:
            logger.error("Error. No suitable mode found. Aborting...")
            sys.exit()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx]

    def get_name(self):
        if self.model == 'transformer':
            return "_".join([self.dataset, self.split, self.mode, self.model, self.tokenizer_name, self.description_mode, str(self.max_description_length), str(self.max_seq_length), str(self.mask_entity_partially), str(self.mask_probability), str(self.only_keep_relevant_classes)])



    def process_text(self, sentences, entity_labels, entity_descriptions):
        logger.info("Dataloader file {}".format(self.get_name()))
        #Currently ignoring sentences with no named entity of type
        if not self.limit and not self.overwrite and os.path.isfile(os.path.join('/tmp', self.get_name())):
            data_file = open(os.path.join('/tmp', self.get_name()), 'rb')
            features = pickle.load(data_file)
        else:

            features = []
            for i, sentence in enumerate(tqdm(sentences[:self.limit], desc="Sentence loop")):
                tokenized_text_list = []
                segment_ids_list = []
                input_ids_list = []
                input_mask_list = []
                break_it = False
                do_mask = random()
                mask_prob = self.mask_probability
                for k, description in enumerate(entity_descriptions):
                    premise = self.tokenizer.tokenize(' '.join(w.word for w in sentence))
                    premise = []
                    text_labels = []
                    text_labels_b = []
                    for word in sentence:
                        tok = self.tokenizer.tokenize(word.word)
                        premise += ["[unused1]"] * len(tok) if (len(word.type.split('-')) > 1 and word.type.split('-')[1] in entity_labels[k] and self.mask_entity_partially and do_mask > mask_prob and self.split == 'train') else tok
                        text_labels += [word.type] * len(tok) if len(word.type.split('-')) == 1 or word.type.split('-')[1] in entity_labels else ["O"] * len(tok)
                        text_labels_b += [word.type.split('B-')[1]] * len(tok) if len(word.type.split('B-')) > 1 and word.type.split('B-')[1] in entity_labels else ["NEG"] * len(tok)

                    if 'B-' + entity_labels[k] not in text_labels and self.only_keep_relevant_classes:
                        continue

                    if set(text_labels) == set(["O"]):
                        break_it = True
                        break
                    #text_labels = [label.split("-")[1]  if len(label.split("-")) > 1 else label for label in text_labels] if not self.add_iob else text_labels
                    hypothesis = self.tokenizer.tokenize(description)
                    diff_descr = len(hypothesis) - self.max_description_length

                    if diff_descr < 0:
                        diff_descr = len(hypothesis)

                    tokenized_text =  ["[CLS]"] + premise + ["[SEP]"] + hypothesis[:diff_descr] + ["[SEP]"]
                    ind_split = tokenized_text.index("[SEP]")
                    remainder = len(tokenized_text) - ind_split
                    max_input = self.max_seq_length - (remainder + 1)
                    samples_num = math.ceil(ind_split / max_input)

                    ind = [0]
                    if samples_num > 1:
                        ind.append(max_input)
                    else:
                        ind.append(ind_split)

                    for i in range(len(ind)-1):
                        tokenized_text_f = tokenized_text[ind[i]:ind[i +1]] + tokenized_text[ind_split:]
                        segment_ids = ([0] * (ind[i+1] - ind[i])) + ([1] * remainder)
                        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text_f)
                        #input_mask = ([1] * ind_split) + ([0] * (len(input_ids) - ind_split))

                        input_mask = [1] * len(input_ids)
                        assert(len(input_ids) == len(segment_ids))
                        assert(len(input_mask) == len(input_ids))

                        tokenized_text_list.append(tokenized_text_f)
                        segment_ids_list.append(segment_ids)
                        input_ids_list.append(input_ids)
                        input_mask_list.append(input_mask)


                if break_it:
                    continue

                text_labels = ["NEG"]
                for word in sentence:
                    tok = self.tokenizer.tokenize(word.word)
                    text_labels += len(tok) * [word.type.split('-')[1]] if len(word.type.split('-')) > 1 and word.type.split('-')[1] in entity_labels else len(tok) * ['NEG']


                text_labels_ids = [entity_labels.index(label) for label in text_labels]
                label_dict = {x:entity_labels[x] for x in set(text_labels_ids)}
                text_labels_ids = text_labels_ids #CLS Token

                sep_index = tokenized_text_list[0].index("[SEP]")
                features.append(InputFeaturesMultiClass(text_list = tokenized_text_list, input_ids_list=input_ids_list,
                              input_mask_list=input_mask_list,
                              segment_ids_list=segment_ids_list,
                              sep_index = sep_index,
                              label_dict = label_dict,
                              tagging = True,
                              label_id= text_labels_ids[:sep_index],
                              labels = text_labels[:sep_index]))


            if not self.limit:
                data_file = open(os.path.join('/tmp', self.get_name()), 'wb')
                pickle.dump(features, data_file)

        logger.info("Number samples: {}".format(len(features)))
        logger.info("Example features: {}".format(features[0]))

        with open(os.path.join(self.output_dir,  "processed_text_" + self.split + ".txt"), "w") as f:
            for element in features[:100]:
                f.write("{}\n".format(element))
        return features


class EntityRecognitionDataset(Dataset):

    def __init__(self, args, split, sentences, entity_labels, entity_descriptions):
        self.tokenizer_name = 'bert-large-cased'
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.sentences = sentences
        self.entity_labels = entity_labels
        self.entity_descriptions = entity_descriptions
        self.features = []
        self.max_description_length = args.max_description_length
        self.limit = args.limit
        self.mask_entity = args.mask_entity
        self.mask_entity_partially = args.mask_entity_partially
        self.max_seq_length = args.max_sequence_length
        self.description_mode = args.entity_descriptions_mode
        self.dataset = args.dataset
        self.mode = args.mode
        self.model = args.model
        self.split = split
        self.overwrite = args.overwrite_dumped
        self.output_dir = args.output_dir
        self.add_iob = args.add_iob
        self.only_keep_relevant_classes = args.only_keep_relevant_classes
        self.mask_probability = 1 - args.mask_probability #0.3

        if args.model == 'transformer':
            self.features = self.process_text(sentences, entity_labels, entity_descriptions)
        else:
            logger.error("Error. No suitable mode found. Aborting...")
            sys.exit()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx]

    def get_name(self):
        return "_".join([self.dataset, self.split, self.mode, self.model, self.tokenizer_name, self.description_mode, str(self.max_description_length), str(self.mask_entity), str(self.max_seq_length), str(self.mask_entity_partially), str(self.mask_probability), str(self.add_iob), str(self.only_keep_relevant_classes)])

    def process_text(self, sentences, entity_labels, entity_descriptions):
        logger.info("Dataloader file {}".format(self.get_name()))
        #Currently ignoring sentences with no named entity of type
        if not self.limit and not self.overwrite and os.path.isfile(os.path.join('/tmp', self.get_name())):
            data_file = open(os.path.join('/tmp', self.get_name()), 'rb')
            features = pickle.load(data_file)
        else:
            features = []
            for i, sentence in enumerate(tqdm(sentences[:self.limit], desc="Sentence loop")):
                for k, description in enumerate(entity_descriptions):
                    premise = self.tokenizer.tokenize(' '.join(w.word for w in sentence))
                    text_labels = [word.type if len(word.type.split('-')) == 1 or word.type.split('-')[1] in entity_labels else "O" for word in sentence]

                    if 'B-' + entity_labels[k] not in text_labels and self.only_keep_relevant_classes:
                        continue

                    if set(text_labels) == set(["O"]):
                        break

                    premise = []
                    text_labels = []
                    mask_prob = self.mask_probability
                    for word in sentence:
                        tok = self.tokenizer.tokenize(word.word)
                        premise += ["[unused1]"] * len(tok) if (len(word.type.split('-')) > 1 and word.type.split('-')[1] in entity_labels[k] and self.mask_entity_partially and random() > mask_prob and self.split == 'train') else tok
                        text_labels += [word.type] * len(tok) if len(word.type.split('-')) == 1 or word.type.split('-')[1] in entity_labels[k] else ["O"] * len(tok)


                    #text_labels = [label.split("-")[1]  if len(label.split("-")) > 1 else label for label in text_labels] if not self.add_iob else text_labels
                    text_labels = [1  if len(label.split("-")) > 1 else 0 for label in text_labels] if not self.add_iob else text_labels
                    hypothesis = self.tokenizer.tokenize(description)
                    diff_descr = len(hypothesis) - self.max_description_length
                    if diff_descr < 0:
                        diff_descr = len(hypothesis)

                    tokenized_text =  ["[CLS]"] + premise + ["[SEP]"] + hypothesis[:diff_descr] + ["[SEP]"]
                    text_labels = [0] + text_labels + [0] + [0 for t in hypothesis[:diff_descr]] + [0]


                    ind_split = tokenized_text.index("[SEP]")
                    remainder = len(tokenized_text) - ind_split
                    max_input = self.max_seq_length - (remainder + 1)

                    samples_num = math.ceil(ind_split / max_input)

                    ind = [0]
                    if samples_num > 1:
                        curr_pos = 0
                        for i in range(samples_num -1):
                            ind.append(max_input + curr_pos)
                            curr_pos += max_input
                        ind.append((ind_split % max_input) + curr_pos)
                    else:
                        ind.append(ind_split)

                    for i in range(len(ind)-1):
                        if i > 0:
                            tokenized_text_f = ["[CLS]"] +  tokenized_text[ind[i]:ind[i +1]] + tokenized_text[ind_split:]
                            text_labels_f = [0] + text_labels[ind[i]:ind[i + 1]] + text_labels[ind_split:]
                            segment_ids = ([0] * (ind[i +1] - ind[i] + 1)) + ([1] * remainder)
                        else:
                            tokenized_text_f = tokenized_text[ind[i]:ind[i +1]] + tokenized_text[ind_split:]
                            text_labels_f = text_labels[ind[i]:ind[i + 1]] + text_labels[ind_split:]
                            segment_ids = ([0] * (ind[i+1] - ind[i])) + ([1] * remainder)

                        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text_f)
                        #input_mask = ([1] * ind_split) + ([0] * (len(input_ids) - ind_split))
                        input_mask = [1] * len(input_ids)

                        assert(len(input_ids) == len(text_labels_f))
                        assert(len(input_ids) == len(segment_ids))
                        assert(len(input_mask) == len(input_ids))
                        assert(len(text_labels_f) == len(input_ids))

                        if i > 0:
                            logger.info("Longer than maximum sequence length.")
                            logger.info(','.join(str(x) for x in tokenized_text_f))
                            logger.info(','.join(str(x) for x in text_labels_f))

                        features.append(InputFeatures(text = tokenized_text_f, input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      sep_index = tokenized_text_f.index("[SEP]"),
                                      description_type = entity_labels[k],
                                      label_id= text_labels_f,
                                      tagging = True))


            if not self.limit:
                data_file = open(os.path.join('/tmp', self.get_name()), 'wb')
                pickle.dump(features, data_file)

        logger.info("Number samples: {}".format(len(features)))
        logger.info("Example features: {}".format(str(features[0])))

        with open(os.path.join(self.output_dir,  "processed_text_" + self.split + ".txt"), "w") as f:
            for element in features[:100]:
                f.write("{}\n".format(element))
        return features


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = ['PAD', "BND"] + list(sorted(vocabulary_inv))
    # Mapping from word to index    print ml.classes_
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    #print vocabulary_inv
    return [vocabulary, vocabulary_inv]
