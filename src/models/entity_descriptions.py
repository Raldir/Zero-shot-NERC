"""Loads and manages entity descriptions"""
import os
import argparse
from utils.readers import read_entity_descriptions, get_zero_shot_splits, medmentions_type_dict_inv



def get_impossible_classes(dataset):
    if dataset == 'ontonotes':
        trivial_classes = ["ORDINAL", "QUANTITY", "MONEY", "PERCENT", "CARDINAL", "LANGUAGE", "TIME"]
    elif dataset == 'medmentions':
        trivial_classes = []

    return trivial_classes


def get_entity_descriptions_negative(mode, dataset, split, filter_classes):
    if split == "conll-2012-test":
        split = "test"
    if mode == "combined":
        des = combined_entity_description(dataset)[split]
    else:
        if dataset == 'ontonotes':
            des = default_entity_descriptions_neg(os.path.join("../data/entity_descriptions/OntoNotes/", mode + "_negative.txt"), dataset, filter_classes)[split] #  "_negative.t
        elif dataset == 'medmentions':
            des = default_entity_descriptions_neg(os.path.join("../data/entity_descriptions/MedMentions/", mode + "_negative.txt"), dataset, filter_classes)[split]
    return des['labels'], des['descriptions']

def get_entity_descriptions(mode, dataset, split, filter_classes):
    if split == "conll-2012-test":
        split = "test"
    if mode == "combined":
        des = combined_entity_descriptions(dataset)[split]
    elif split == 'all':
        if dataset == 'ontonotes':
            k = default_entity_descriptions(os.path.join("../data/entity_descriptions/OntoNotes/", mode + ".txt"), dataset, filter_classes)
            des = k['train']
            des['labels'] += k['dev']['labels']
            des['labels'] += k['test']['labels']
            des['descriptions'] += k['dev']['descriptions']
            des['descriptions'] += k['test']['descriptions']
        elif dataset == 'medmentions':
            k = default_entity_descriptions(os.path.join("../data/entity_descriptions/MedMentions/", mode + ".txt"), dataset, filter_classes)
            des = k['train']
            des['labels'] += k['dev']['labels']
            des['labels'] += k['test']['labels']
            des['descriptions'] += k['dev']['descriptions']
            des['descriptions'] += k['test']['descriptions']
    else:
        if dataset == 'ontonotes':
            des = default_entity_descriptions(os.path.join("../data/entity_descriptions/OntoNotes/", mode + ".txt"), dataset, filter_classes)[split]
        elif dataset == 'medmentions':
            des = default_entity_descriptions(os.path.join("../data/entity_descriptions/MedMentions/", mode + ".txt"), dataset, filter_classes)[split]
    return des['labels'], des['descriptions']

def get_symbols(symbols_file, dataset):
    symbols_dict = {}
    if dataset == "ontonotes":
        des = os.path.join("../data/entity_descriptions/OntoNotes/", symbols_file + ".txt")
    elif dataset == 'medmentions':
        des = os.path.join("../data/entity_descriptions/MedMentions/", symbols_file + ".txt")
    with open(des, "r") as f:
        lines = f.readlines()
        for line in lines:
            content = line.split('\t')
            symbols_dict[content[0]] = content[1].strip()
    return symbols_dict

def default_entity_descriptions_neg(path, dataset, filter_classes):
    if filter_classes:
        filter =  get_impossible_classes(dataset)
    if dataset == "medmentions":
        id_to_type = medmentions_type_dict_inv()
    entity_descriptions = read_entity_descriptions(path)
    train_types, dev_types, test_types = get_zero_shot_splits(dataset)
    entity_descriptions_splits = {"train" : {'labels': [], 'descriptions' : []}, 'dev' : {'labels': [], 'descriptions' : []}, "test" : {'labels': [], 'descriptions' : []}}
    entity_descriptions_splits['train']['labels'].append('NEG')
    entity_descriptions_splits['train']['descriptions'].append(entity_descriptions['NEG'])
    entity_descriptions_splits['dev']['labels'].append('NEG')
    entity_descriptions_splits['dev']['descriptions'].append(entity_descriptions['NEG'])
    entity_descriptions_splits['test']['labels'].append('NEG')
    entity_descriptions_splits['test']['descriptions'].append(entity_descriptions['NEG'])
    for key, value in entity_descriptions.items():
        if filter_classes and key in filter:
            continue
        if dataset == "medmentions":
            key = id_to_type[key]
        if key in train_types:
            entity_descriptions_splits['train']['labels'].append(key)
            entity_descriptions_splits['train']['descriptions'].append(value)
        elif key in dev_types:
            entity_descriptions_splits['dev']['labels'].append(key)
            entity_descriptions_splits['dev']['descriptions'].append(value)
        elif key in test_types :
            entity_descriptions_splits['test']['labels'].append(key)
            entity_descriptions_splits['test']['descriptions'].append(value)
    return entity_descriptions_splits

def default_entity_descriptions(path, dataset, filter_classes):
    filter =  get_impossible_classes(dataset)
    if dataset == "medmentions":
        id_to_type = medmentions_type_dict_inv()
    entity_descriptions = read_entity_descriptions(path)
    train_types, dev_types, test_types = get_zero_shot_splits(dataset)
    entity_descriptions_splits = {"train" : {'labels': [], 'descriptions' : []}, 'dev' : {'labels': [], 'descriptions' : []}, "test" : {'labels': [], 'descriptions' : []}}
    for key, value in entity_descriptions.items():
        if key in filter and filter_classes:
            continue
        if dataset == "medmentions":
            key = id_to_type[key]
        if key in train_types:
            entity_descriptions_splits['train']['labels'].append(key)
            entity_descriptions_splits['train']['descriptions'].append(value)
        elif key in dev_types:
            entity_descriptions_splits['dev']['labels'].append(key)
            entity_descriptions_splits['dev']['descriptions'].append(value)
        elif key in test_types:
            entity_descriptions_splits['test']['labels'].append(key)
            entity_descriptions_splits['test']['descriptions'].append(value)
    return entity_descriptions_splits


def generate_entity_description_custom02(path):
    entity_descriptions = read_entity_descriptions(path)
    names = []
    brands = []
    with open("../data/entity_descriptions/resources/first_names.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            names.append(line.strip())
    with open("../data/entity_descriptions/resources/brands.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            brands.append(line.strip())
    custom_entity_description = {}
    for entity_type, description in entity_descriptions.items():
        if entity_type == "DATE":
            custom_entity_description[entity_type] = description  + " " + " ".join([str(i) for i in range(1900, 2010)]) + " " + " ".join(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        elif entity_type == "PERSON":
            custom_entity_description[entity_type] = description  + " " + " ".join(names)
        elif entity_type == "LOC":
            custom_entity_description[entity_type] = description + " " + " ".join(['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']) + " " + " ".join(["Atlantic Ocean", "Pacific Ocean", "Indian Ocian", "Southern Ocean", "Arctic Ocean", "Arabian Sea", "Bay of Bengal", "Red Sea", "Persian Golf", "Bali Sea", "Yellow Sea", "Hudson Bay", "Black Sea", "Baltic Sea", "North Sea", "English Channel", "Mediterranean Sea"])
        elif entity_type == "PRODUCT":
            custom_entity_description[entity_type] = description + " " + " ".join(brands)
        elif entity_type == "TIME":
            custom_entity_description[entity_type] = description + " " + " ".join(["p.m."] * 10) + " " + " ".join(["a.m."] * 10) + " " + " ".join(["time"] * 10) +  " "  + " ".join(["morning", 'evening', 'night'] * 2)
        elif entity_type == "PERCENT":
            custom_entity_description[entity_type] = description + " ".join(["%"] * 80)
        elif entity_type == "MONEY":
            custom_entity_description[entity_type] = description + " " + " ".join(["$"] * 30) + " " + " ".join(["£"] * 30) + " " + " ".join(["€"] * 30)
        elif entity_type == "QUANTITY":
            custom_entity_description[entity_type] = description + " " + " ".join(["cm", "mm", "m", "meter", "kg", "g", "gram"])
        elif entity_type == "ORDINAL":
            custom_entity_description[entity_type] = description + " " + " ".join([str(el) + "th" for el in range(4, 30)]) + " " + " ".join(["1st", "2nd", "first", "second", '3rd', "third", "forth", "fifth"])
        elif entity_type == "LANGUAGE":
            custom_entity_description[entity_type] = description + " " + " ".join(['German' ,'English', 'French' ,'Turkish', 'Finnish', 'Latin', 'Greek', 'Arabic', 'Japanese' ,'Chinese', 'Dutch', 'Russian', 'German', 'Spanish', 'Portuguese', 'Hindi', 'Urdu'] * 10)
        else:
            custom_entity_description[entity_type] = description
    with open("../data/entity_descriptions/OntoNotes/custom02.txt", 'w') as f:
        for entity_type, description in custom_entity_description.items():
            f.write(entity_type + "\t" + description + "\n")


def combined_entity_descriptions(dataset):
    if dataset == 'ontonotes':
        combine = ['annotation_guidelines.txt', 'wikipedia.txt', 'wordnet.txt']
    elif dataset == 'medmentions':
        combine = ['wikipedia.txt', 'umls.txt']
    if dataset == "ontonotes":
        path = "../data/entity_descriptions/OntoNotes/"
    elif dataset == 'medmentions':
        path = "../data/entity_descriptions/MedMentions/"
    all_descriptions_split = {"train" : {}, 'dev' : {}, "test" : {}}
    for des in combine:
        entity_descriptions_split = default_entity_descriptions(os.path.join(path, des), dataset)
        for split in entity_descriptions_split.keys():
            for i, label in enumerate(entity_descriptions_split[split]['labels']):
                if entity_descriptions_split[split]['labels'][i] in all_descriptions_split[split]:
                    all_descriptions_split[split][label] = all_descriptions_split[split][label] + " " + (entity_descriptions_split[split]['descriptions'][i])
                else:
                    all_descriptions_split[split][label] = (entity_descriptions_split[split]['descriptions'][i])
    all_entity_descriptions = {"train" : {'labels': list(all_descriptions_split['train'].keys()), 'descriptions' : list(all_descriptions_split['train'].values())}, 'dev' : {'labels': list(all_descriptions_split['dev'].keys()), 'descriptions' : list(all_descriptions_split['dev'].values())}, "test" : {'labels': list(all_descriptions_split['test'].keys()), 'descriptions' : list(all_descriptions_split['test'].values())}}
    return all_entity_descriptions
