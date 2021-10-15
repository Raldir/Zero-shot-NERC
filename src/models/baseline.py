""" Tf-idf similarity baseline"""
import os
import sys
from utils.wrappers import Token
from utils.readers import read_data_medmentions
import matplotlib.pyplot as plt
import numpy as np
import spacy
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import scipy
from tqdm import tqdm
from itertools import chain
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_baseline(dataloader, samples_train, labels, entity_labels, entity_descriptions):
    print("Executing TF-idf baseline.")
    #samples_train, labels_train = convert_into_samples(data_train)

    samples = [f.text for f in dataloader.features]
    vectorizer = TfidfVectorizer(ngram_range = (1,1), token_pattern = "[^,\s]+")

    print("FITTING TF-DF ON TRAIN")

    samples_train_iterator = iter(samples_train)

    #data_iter = chain(samples_train_iterator)#,wikidata)
    vectorizer.fit(tqdm(samples_train))

    print(len(labels))
    print("Start evaluation...")
    predictions = predict(samples, labels, entity_labels, entity_descriptions, vectorizer)

    return predictions



def predict(samples, labels, entity_label, entity_descriptions, vectorizer):
    X = vectorizer.transform(samples)

    descriptions_tfidf = vectorizer.transform(entity_descriptions)
    predictions = []
    overlaps = {}

    pred_all = dict((ent,[]) for ent in entity_label)
    for s in tqdm(X):
        s = s.todense()
        pred = []
        for i,description in enumerate(descriptions_tfidf):
            description = description.todense()
            overlap = cosine_similarity(s, description)[0][0]
            # overlap = np.multiply(s, description)
            # overlap = np.sum(overlap[0])
            pred.append(overlap)
            pred_all[entity_label[i]].append(overlap)

        predictions.append(entity_label[np.argmax(np.asarray(pred))])
    # overlap_scores(pred_all, labels, entity_label)

    print(len(predictions))
    return predictions



def overlap_scores(dictE, true_labels, entity_labels):
    print("{}\\\\".format("& ".join(entity_labels)))
    for label in entity_labels:
        scores_all = dict((ent,0) for ent in entity_labels)
        count = 0
        for i, t in enumerate(true_labels):
            if t == label:
                count+=1
                for label2 in entity_labels:
                    scores_all[label2]+= dictE[label2][i]
        for key in scores_all:
            scores_all[key] = scores_all[key] / count
        str = label + " & "
        for lab in scores_all:
            str += "& {0:.2f} ".format(scores_all[lab])
        print(str + "\\\\")
