import os
import sys
from utils.wrappers import Token
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize
import spacy


ontonotes_splits =  {"train_types" : ['ORG', 'GPE', 'DATE', 'PERSON'],
 "dev_types" : ['ORDINAL', 'EVENT', 'PERCENT', 'PRODUCT', 'LAW', 'NORP', 'MONEY'],
 "test_types" : ['LOC', 'FAC', 'TIME', 'QUANTITY', 'LANGUAGE', 'CARDINAL', 'WORK_OF_ART']}
 
medmentions_splits = {"train_types" : ['Biologic_Function', 'Chemical', 'Health_Care_Activity', 'Anotomical_Structure', "Finding", "Spatial_Concept", "Intellectual_Product", "Research_Activity", 'Medical_Device', 'Eukaryote', 'Population_Group'], "dev_types": [ 'Biomedical_Occupation_or_Discipline', 'Virus', 'Clinical_Attribute', 'Injury_or_Poisoning', 'Organization'], "test_types" : ['Body_System', 'Food', 'Body_Substance', 'Bacterium', 'Professional_or_Occupational_Group'] }

def get_zero_shot_splits(dataset):
    if dataset == "medmentions":
        return medmentions_splits["train_types"], medmentions_splits["dev_types"], medmentions_splits["test_types"]
    elif dataset == "ontonotes":
        return ontonotes_splits["train_types"], ontonotes_splits["dev_types"], ontonotes_splits["test_types"]

def medmentions_type_dict_inv():
    return {'T058': "Health_Care_Activity", "T062": "Research_Activity", "T037": "Injury_or_Poisoning", "T038": "Biologic_Function", "T005": "Virus", "T007": "Bacterium", "T204": "Eukaryote", "T017": "Anotomical_Structure", "T074": "Medical_Device", "T031": "Body_Substance", "T103": "Chemical", "T168": "Food", "T201": "Clinical_Attribute", "T033": "Finding", "T082": "Spatial_Concept", "T022": "Body_System", "T091": "Biomedical_Occupation_or_Discipline", "T092": "Organization", "T097": "Professional_or_Occupational_Group", "T098": "Population_Group", "T170": "Intellectual_Product", "NEG" :"NEG"}

def medmentions_type_dict():
    return {v: k for k, v in medmentions_type_dict_inv().items()}


def read_wikipedia(path):
    dump = []
    return open(path, 'r')


def read_results(input_path):
    labels = []
    predictions = []
    with open(input_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label, prediction = [int(x) for x in line.strip().split('\t')]
            labels.append(label)
            predictions.append(prediction)
    return [labels, predictions]

def read_data_medmentions(input_path, split, id = False, limit = None):
    #limit: number of words
    if split == "all":
        data = []
        for split in ["train", "dev", "test"]:
            read_in = os.path.join(input_path, "medmentions." + split + ".ner")
            with open(read_in, "r") as f:
                data += f.readlines()
    else:
        read_in = os.path.join(input_path, "medmentions." + split + ".ner")
        with open(read_in, "r") as f:
            data = f.readlines()[1:-2] #skipping empty first line
    content = []
    sentence = []
    type_to_id = medmentions_type_dict()
    id_to_type = medmentions_type_dict_inv()
    for l in data[:limit]:
        l = l.strip()
        if not l:
            if sentence: #sometimes more than one linebreak so make sure that sentence is kept and not empty list
                content.append(sentence)
                sentence = []
            continue
        line = l.split('\t')
        if id:
            token = Token(word = line[0], type =  line[1] if line[1] == "O" else line[1].split("-")[0] + '-' + id_to_type[line[1].split("-")[1]], type_id = line[1])
        else:
            token = Token(word = line[0], type =  line[1], type_id = line[1] if line[1] == "O" else line[1].split("-")[0] + '-' + type_to_id[line[1].split("-")[1]])
        sentence.append(token)
    return content


def read_data_ontonotes(input_path, split, limit = None):
    #limit: number of words
    if "few-shot-paper" in input_path:
        read_in = os.path.join(input_path, split + ".txt")
        with open(read_in, "r") as f:
            data = f.readlines() #skipping empty first line
        content = []
        sentence = []
        for l in data:
            l = l.strip()
            if not l:
                if sentence: #sometimes more than one linebreak so make sure that sentence is kept and not empty list
                    content.append(sentence)
                    sentence = []
                continue
            line = l.split(' ')
            token = Token(word = line[0], type = line[1])
            sentence.append(token)
        return content
    else:
        read_in = os.path.join(input_path, "onto." + split + ".ner")
        with open(read_in, "r") as f:
            data = f.readlines()[1:-2] #skipping empty first line
    content = []
    sentence = []
    for l in data[:limit]:
        l = l.strip()
        if not l:
            if sentence: #sometimes more than one linebreak so make sure that sentence is kept and not empty list
                content.append(sentence)
                sentence = []
            continue
        line = l.split('\t')
        token = Token(word = line[0], type = line[3], pos = line[1], const = line[2])
        sentence.append(token)
    return content

def read_entity_descriptions(path):
    nlp = spacy.load("en_core_web_sm")
    descriptions = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.strip().split('\t')
            sent =  sent_tokenize(content[1])
            tokenized = " ".join([' '.join([el.text for el in  nlp(sen)]) for sen in sent])
            descriptions[content[0]] = tokenized
    return descriptions
