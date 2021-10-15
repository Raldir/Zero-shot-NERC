"""Preprocess the original MedMentions dataset into desired BIO format"""
import os
import sys
from utils.wrappers import Token
from utils.readers import read_data_medmentions, get_zero_shot_splits, medmentions_type_dict_inv
import matplotlib.pyplot as plt
import numpy as np
import spacy
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import random

count_duplicate = 0

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
    return -1

def convert_to_iob(id, text, entities, nlp, end_index, start_index):
    count = 0
    sentence = []
    sentences = []
    current_sent_pos = 0
    id_to_type  = medmentions_type_dict_inv()
    for sent_o in sent_tokenize(text):
        sent = nlp(sent_o)
        char_list = [s.idx for s in sent]
        for i,tok in enumerate(sent):
            word = tok.text
            # if entities:
                # print(sent)
                # print(word, char_list[i], entities[0], start_index[0], end_index[0], current_sent_pos)
            if entities and char_list[i] + current_sent_pos == start_index[0]:
                count+=1
                app = 'B-'
                ent = entities[0][0]
                type = entities[0][1]
                token = Token(word = word, type = app + id_to_type[type], type_id = app + type)
            elif entities and char_list[i] + current_sent_pos > start_index[0] and char_list[i] + current_sent_pos <= end_index[0]:
                app = 'I-'
                ent = entities[0][0]
                type = entities[0][1]
                token = Token(word = word, type = app + id_to_type[type], type_id = app + type)
            else:
                token = Token(word = word, type = 'O', type_id = "O")
            if entities and  not (char_list[i] + current_sent_pos == start_index[0]) and "B-" in token.type:
                print(token)
                print("TREASON")
            if entities and len(word) + char_list[i] + current_sent_pos >= end_index[0]:
                start_index.pop(0)
                end_index.pop(0)
                entities.pop(0)
            sentence.append(token)
        current_sent_pos += len(sent_o) + 1
        sentences.append((id, sentence))
        sentence = []
    return sentences, count


def preprocess_medmentions(args):

    global count_duplicate
    data_path = os.path.join(args.input_path, 'corpus_pubtator.txt')
    train_id_path = os.path.join(args.input_path, 'corpus_pubtator_pmids_train.txt')
    dev_id_path = os.path.join(args.input_path, 'corpus_pubtator_pmids_dev.txt')
    test_id_path = os.path.join(args.input_path, 'corpus_pubtator_pmids_test.txt')

    nlp = spacy.load("en_core_web_sm")
    sentences = []
    with open(data_path, 'r') as f:
        data = f.readlines()
        ids = []
        titles = []
        abstracts = []
        entities_title = []
        entities_abstract = []
        cur_id = None
        entities_title_curr = []
        entities_abstract_curr = []
        end_indicies_title = []
        end_indicies_abstract = []
        end_indicies_curr_title = []
        end_indicies_curr_abstract = []
        start_indicies_title = []
        start_indicies_abstract = []
        start_indicies_curr_title = []
        start_indicies_curr_abstract = []
        for line in data:
            if '|t|' in line:
                cur_id = line.split("|t|")[0]
                ids.append(cur_id)
                titles.append(line.split("|t|")[1])
            elif '|a|' in line:
                abstracts.append(line.split('|a|')[1])
            elif cur_id in line:
                entity_information = line.split('\t')
                entity = entity_information[3]
                entity_type = entity_information[4]
                end_index = entity_information[2]
                start_index = entity_information[1]
                if len(end_indicies_curr_title) > 0 and int(end_index) <= end_indicies_curr_title[-1]:
                    #print(titles[-1])
                    #print("TITLE", end_index, end_indicies_curr_title)
                    count_duplicate+=1
                    continue
                elif len(end_indicies_curr_abstract) > 0 and int(end_index) - len(titles[-1]) <= end_indicies_curr_abstract[-1]:
                    #print("ABSTACT", int(end_index) - len(titles[-1]), end_indicies_curr_abstract)
                    count_duplicate+=1
                    continue
                if int(end_index) < len(titles[-1]):
                    entities_title_curr.append((entity, entity_type))
                    end_indicies_curr_title.append(int(end_index))
                    start_indicies_curr_title.append(int(start_index))
                else:
                    entities_abstract_curr.append((entity, entity_type))
                    end_indicies_curr_abstract.append(int(end_index) - len(titles[-1]))
                    start_indicies_curr_abstract.append(int(start_index) - len(titles[-1]))
            else:
                #print(line)
                entities_title.append(entities_title_curr)
                entities_abstract.append(entities_abstract_curr)
                end_indicies_title.append(end_indicies_curr_title)
                end_indicies_abstract.append(end_indicies_curr_abstract)
                end_indicies_curr_title = []
                end_indicies_curr_abstract = []
                start_indicies_title.append(start_indicies_curr_title)
                start_indicies_abstract.append(start_indicies_curr_abstract)
                start_indicies_curr_title = []
                start_indicies_curr_abstract = []
                entities_title_curr = []
                entities_abstract_curr = []
                cur_id = None

        print("TOTAL ENTITIES", sum(len(el) for el in entities_abstract + entities_title))
        print("DUPLICATES", count_duplicate)
        count = 0
        for i in tqdm(range(len(ids))):
            id = ids[i]
            abstract = abstracts[i]
            entities_a = entities_abstract[i]
            title = titles[i]
            entities_t = entities_title[i]
            end_index_title = end_indicies_title[i]
            end_index_abstract = end_indicies_abstract[i]
            start_index_title = start_indicies_title[i]
            start_index_abstract = start_indicies_abstract[i]
            sentences_title, count_t = convert_to_iob(id, title, entities_t, nlp, end_index_title, start_index_title)
            sentences += sentences_title
            sentences_abstract, count_a = convert_to_iob(id, abstract, entities_a, nlp, end_index_abstract, start_index_abstract)
            count += count_a + count_t
            sentences += sentences_abstract
        print("ENTITIES WRITTEN", count)
        print("SENTENCES", len(sentences))

    train_ids, dev_ids, test_ids = [], [], []

    train_sentences, dev_sentences, test_sentences = [], [] , []
    with open(train_id_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            train_ids.append(line.strip())
    with open(dev_id_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            dev_ids.append(line.strip())
    with open(test_id_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            test_ids.append(line.strip())

    for id, sentence in sentences:
        if id in train_ids:
            train_sentences.append(sentence)
        elif id in dev_ids:
            dev_sentences.append(sentence)
        elif id in test_ids:
            test_sentences.append(sentence)
        else:
            print("ID", id, "not found!")

    write_into_file(train_sentences, args.output_path, "medmentions.train.ner")
    write_into_file(dev_sentences, args.output_path, "medmentions.dev.ner")
    write_into_file(test_sentences, args.output_path, "medmentions.test.ner")

def write_into_file(sentences, output_path, filename, id = False):
    output_file = os.path.join(output_path, filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_file, "w") as f:
        for sentence in sentences:
            f.write("\n")
            for word in sentence:
                if id:
                    f.write("{}\t{}\n".format(word.word, word.type_id))
                else:
                    f.write("{}\t{}\n".format(word.word, word.type))
        f.write("\n")
        f.write("\n")

def alternative_splits_medmentions(args):
    sentences = []
    for split in ['train', 'dev', 'test']:
        sentences += read_data_medmentions(args.input_path, split)

    sentences_train = []
    sentences_dev = []
    sentences_test = []

    train_types, dev_types, test_types  = get_zero_shot_splits("medmentions")

    for sentence in sentences:
        count_train, count_dev, count_test = 0,0,0
        for word in sentence:
            if len(word.type.split('-')) > 1:
                if word.type.split('-')[1] in test_types:
                    count_test +=1
                elif word.type.split('-')[1] in dev_types:
                    count_dev +=1
                elif word.type.split('-')[1] in train_types:
                    count_train +=1

        if count_test > count_dev and count_test > count_train:
            sentences_test.append(sentence)
        elif count_dev > count_test and count_dev > count_train:
            sentences_dev.append(sentence)
        elif count_train > 0:
            sentences_train.append(sentence)
        else:
            rand = random.random()
            if rand <= 0.6:
                sentences_train.append(sentence)
            elif rand > 0.6 and rand <= 0.8:
                sentences_dev.append(sentence)
            else:
                sentences_test.append(sentence)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    write_into_file(sentences_train, args.output_path, "medmentions.train.ner")
    write_into_file(sentences_dev, args.output_path, "medmentions.dev.ner")
    write_into_file(sentences_test, args.output_path, "medmentions.test.ner")

def alternative_splits_medmentions_extreme(args):
    sentences = []
    for split in ['train', 'dev', 'test']:
        sentences += read_data_medmentions(args.input_path, split)

    sentences_train = []
    sentences_dev = []
    sentences_test = []

    train_types, dev_types, test_types  = get_zero_shot_splits("medmentions")

    for sentence in sentences:
        count_train, count_dev, count_test = 0,0,0
        for word in sentence:
            if len(word.type.split('-')) > 1:
                if word.type.split('-')[1] in test_types:
                    count_test +=1
                elif word.type.split('-')[1] in dev_types:
                    count_dev +=1
                elif word.type.split('-')[1] in train_types:
                    count_train +=1

        if count_test > 0 and count_train == 0 and count_dev == 0:
            sentences_test.append(sentence)
        elif count_dev > 0 and count_test == 0 and count_train == 0:
            sentences_dev.append(sentence)
        elif count_train > 0 and count_test == 0 and count_dev == 0:
            sentences_train.append(sentence)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    write_into_file(sentences_train, args.output_path, "medmentions.train.ner")
    write_into_file(sentences_dev, args.output_path, "medmentions.dev.ner")
    write_into_file(sentences_test, args.output_path, "medmentions.test.ner")


def write_zero_shot(data, types, output_path, file_name):
    output_file = os.path.join(output_path, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_file, "w") as f:
        for sentence in data:
            f.write("\n")
            for tok in sentence:
                #Not annotated parts are also replaced with O which is fine. Removes unnecessary distinctuions
                f.write("{}\t{}\n".format(tok.word, tok.type if (len(tok.type.split('-')) > 1 and tok.type.split('-')[1]) in types else "O"))
        f.write('\n')
        f.write('\n')
        f.write('\n')


def create_zero_shot_medmentions_dataset(args):
    train_data = read_data_medmentions(args.input_path, "train")
    dev_data = read_data_medmentions(args.input_path, "dev")
    test_data = read_data_medmentions(args.input_path, "test")

    train_types, dev_types, test_types = get_zero_shot_splits("medmentions")
    write_zero_shot(train_data, train_types, args.output_path,  "medmentions.train.ner")
    write_zero_shot(dev_data, dev_types, args.output_path,  "medmentions.dev.ner")
    write_zero_shot(test_data, test_types, args.output_path,  "medmentions.test.ner")
