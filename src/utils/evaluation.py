"""Evaluation classes for tagging and classification"""

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, auc, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
import logging
import numpy as np
import io
import itertools
logger = logging.getLogger(__name__)
import os

import seqeval.metrics


class evaluator_tagger(object):

    def __init__(self, args, labels, predictions, entity_labels, sep_indicies, texts = None, prediction_probs = None, description_types = None, save_results = False):
        self.args = args
        self.labels = labels
        self.predictions = predictions
        self.entity_labels = entity_labels
        self.sep_indicies = sep_indicies
        self.prediction_probs = prediction_probs
        self.description_types = description_types
        self.texts = texts
        self.scores = {}
        self.scores_chunk = {}
        self.scores_recognition = {}
        self.scores_recognition_chunk = {}
        self.save_results = save_results


        if 'multiclass' in args.mode:
            #texts = [texts[i][1:] for i in range(len(texts))]
            self.evaluate_transformer(args, labels, predictions, entity_labels, sep_indicies, texts)
        else:
            labels, predictions = self.convert_binary_to_types_tagger(labels, predictions, entity_labels, description_types, texts)
            #Map predictions to sentence
            self.evaluate_transformer(args, labels, predictions, entity_labels, sep_indicies, texts)


    def save_results_to_file(self, labels, predictions):
        dir = '../results'
        f_name = os.path.join(dir, '_'.join([self.args.mode, self.args.model_type, self.args.dataset, self.args.split]))
        with open(f_name, 'w') as f:
            for i in range(len(labels)):
                f.write("{}\t{}\n".format(labels[i], predictions[i]))

    def flat_labels(self, labels, predictions):
        labels_flat = [item for sublist in labels for item in sublist]
        predictions_flat = [item for sublist in predictions for item in sublist]
        return labels_flat, predictions_flat

    def evaluate_transformer(self, args, labels, predictions, entity_labels, sep_indicies, texts):
        labels = [[tok for tok in sentence[:sep_indicies[i]]] for i, sentence in enumerate(labels)]
        predictions = [[tok for tok in sentence[:sep_indicies[i]]] for i, sentence in enumerate(predictions)]

        if texts != None:
            labels_no_sub = [[tok for j, tok in enumerate(sentence[:sep_indicies[i]]) if '#' not in texts[i][j]] for i, sentence in enumerate(labels)]
            predictions_no_sub = [[tok for j, tok in enumerate(sentence[:sep_indicies[i]]) if '#' not in texts[i][j]] for i, sentence in enumerate(predictions)]
        else:
            labels_no_sub = labels
            predictions_no_sub = predictions

        logger.info("######################### CHUNK BASED ####################################")
        # labels_chunk, predictions_chunk = self._convert_to_chunks(labels_no_sub, predictions_no_sub)
        # labels_chunk, predictions_chunk = self.flat_labels(labels_chunk, predictions_chunk)
        # self.scores_chunk =  self._calculate_scores(labels_chunk, predictions_chunk, entity_labels)

        labels_iob, predictions_iob = self._convert_to_iob(labels_no_sub, predictions_no_sub, entity_labels)
        f1 = seqeval.metrics.f1_score(labels_iob, predictions_iob, average='macro')
        acc = seqeval.metrics.accuracy_score(labels_iob, predictions_iob)
        prec = seqeval.metrics.precision_score(labels_iob, predictions_iob, average='macro')
        recall = seqeval.metrics.recall_score(labels_iob, predictions_iob, average='macro')
        self.scores_chunk['f1_macro_no_neg'] = f1
        self.scores_chunk['recall_no_neg'] = recall
        self.scores_chunk['precision_no_neg'] = prec
        class_report = seqeval.metrics.classification_report(labels_iob, predictions_iob)
        logger.info("acc: {}, rec: {}, precision: {}, f1 : {}".format(acc, recall, prec, f1))


        logger.info("######################### EXCLUDE SUBTOKENS################################")

        labels, predictions = self.flat_labels(labels_no_sub, predictions_no_sub)
        if self.save_results:
            self.save_results_to_file(labels, predictions)
        self.scores = self._calculate_scores(labels, predictions, entity_labels)

        return self.scores, self.scores_chunk, self.scores_recognition, self.scores_recognition_chunk

    def id_to_entity(self,labels, predictions):
        label_types = [self.entity_labels[x] for x in labels]
        prediction_types = [self.entity_labels[x] for x in predictions]
        return label_types, prediction_types

    def _calculate_scores(self, labels, predictions, entity_labels, recognition = False):
        scores = {}
        if 'multiclass' not in self.args.mode and not recognition:
            label_types = [entity_labels[x-1] if x > 0 else "NEG" for x in labels]
            prediction_types = [entity_labels[x-1] if x > 0 else "NEG" for x in predictions]
        else:
            label_types = [entity_labels[x] for x in labels]
            prediction_types = [entity_labels[x] for x in predictions]


        scores['acc'] = accuracy_score(label_types, prediction_types)
        scores['recall'] = recall_score(label_types, prediction_types, average='micro')
        scores['precision'] =  precision_score(label_types, prediction_types, average='micro')
        scores['f1'] = f1_score(label_types, prediction_types, average='micro')
        scores['f1_macro'] = f1_score(label_types, prediction_types, average='macro')

        entity_labels2 = [x for x in entity_labels if x != 'NEG']
        scores['recall_no_neg'] = recall_score(label_types, prediction_types, labels = entity_labels2, average='micro')
        scores['precision_no_neg'] =  precision_score(label_types, prediction_types, labels = entity_labels2, average='micro')
        scores['f1_no_neg'] = f1_score(label_types, prediction_types, labels = entity_labels2, average='micro')
        scores['f1_macro_no_neg'] = f1_score(label_types, prediction_types, labels = entity_labels2, average='macro')
        scores['acc_no_neg'] = accuracy_score([x for x in label_types if x != 'NEG'], [x for i, x in enumerate(prediction_types) if label_types[i] != 'NEG'])


        logger.info("########## Multi-class scores ###############")
        logger.info("acc: {}, rec: {}, precision: {}, f1 : {}, f1_macro : {}".format(scores['acc'], scores['recall'], scores['precision'], scores['f1'], scores['f1_macro']))
        logger.info("######## Multi-class scores w/o neg class (O) #######")
        logger.info("rec: {}, precision: {}, f1 : {}, f1_macro : {}, acc: {}".format(scores['recall_no_neg'], scores['precision_no_neg'], scores['f1_no_neg'], scores['f1_macro_no_neg'], scores['acc_no_neg']))

        scores['report_results'] = classification_report(label_types, prediction_types, output_dict = True)

        return scores


    def _convert_to_iob(self, labels_no_sub, predictions_no_sub, entity_labels, recognition=False):
        labels_iob = []
        predictions_iob = []


        for i in range(len(labels_no_sub)):
            labels_sent = []
            pred_sent = []
            for j in range(len(labels_no_sub[i])):
                curr_lab = labels_no_sub[i][j] - (1 if 'multiclass' not in self.args.mode else 0)
                lab = entity_labels[curr_lab] if not recognition else 'ENT'
                if labels_no_sub[i][j] != 0:
                    if j-1 < 0 or labels_no_sub[i][j -1] ==0:
                        labels_sent.append('B-' + lab)
                    else:# j+1 < len(labels_no_sub) and labels_no_sub[i][j +1] !=0:
                        labels_sent.append('I-' + lab)
                else:
                    labels_sent.append('O')

            for j in range(len(predictions_no_sub[i])):
                curr_lab = predictions_no_sub[i][j] - (1 if 'multiclass' not in self.args.mode else 0)
                lab = entity_labels[curr_lab] if not recognition else 'ENT'
                # if recognition:
                #     lab = 'ENT' if labels_no_sub[i][j] >= 1 else 'NEG'
                if predictions_no_sub[i][j] != 0:
                    if j-1 < 0 or predictions_no_sub[i][j -1] ==0:
                        pred_sent.append('B-' + lab)
                    else:# j+1 < len(labels_no_sub) and labels_no_sub[i][j +1] !=0:
                        pred_sent.append('I-' + lab)
                else:
                    pred_sent.append('O')
            labels_iob.append(labels_sent)
            predictions_iob.append(pred_sent)

            # print(labels_iob)

        return labels_iob, predictions_iob


    #Convert binary predictions to entity type
    def convert_binary_to_types_tagger(self, label, predictions, entity_labels, description_types, text = None):
        predictions_types = []
        label_types = []
        for i, pred in enumerate(predictions):
            label_list = []
            for j, l in enumerate(pred):
                if l == 1:
                    label_list.append(entity_labels.index(description_types[i]) + 1)
                else:
                    label_list.append(0)
            predictions_types.append(label_list)
        for i, lab in enumerate(label):
            label_list = []
            for j, l in enumerate(lab):
                if l == 1:
                    # print(entity_labels)
                    # print(description_types[i])
                    label_list.append(entity_labels.index(description_types[i])+ 1)
                else:
                    label_list.append(0)
            label_types.append(label_list)
        return label_types, predictions_types

    #Convert the token-based labels to chunks
    def _convert_to_chunks(self, labels_no_sub, predictions_no_sub):
        labels_chunk = []
        predictions_chunk = []

        for i in range(len(labels_no_sub)):

            in_chunk = False
            pred_chunk = True
            lab = []
            pred = []
            curr_lab = -1
            for j in range(len(labels_no_sub[i]) -1):
                if curr_lab != -1 and labels_no_sub[i][j +1] != curr_lab and in_chunk:
                    lab.append(curr_lab)
                    # lab.append(labels_no_sub[i][j])
                    pred.append(curr_lab if pred_chunk else 0)
                    pred.append(predictions_no_sub[i][j])
                    in_chunk = False
                    curr_lab = -1
                    pred_chunk = True
                elif labels_no_sub[i][j] > 0:
                    in_chunk = True
                    curr_lab = labels_no_sub[i][j]
                    pred_chunk = True if predictions_no_sub[i][j] == curr_lab and pred_chunk != False else False
                    continue
                else:
                    if (len(pred) > 0 and predictions_no_sub[i][j] == pred[-1]) or  predictions_no_sub[i][j] == 0:
                        continue
                    else:
                        lab.append(labels_no_sub[i][j])
                        pred.append(predictions_no_sub[i][j])
            labels_chunk.append(lab)
            predictions_chunk.append(pred)
        return labels_chunk, predictions_chunk
#
# class evaluator_tagger(object):
#
#     def __init__(self, args, labels, predictions, entity_labels, sep_indicies, texts = None, prediction_probs = None, description_types = None, save_results = False):
#         self.args = args
#         self.labels = labels
#         self.predictions = predictions
#         self.entity_labels = entity_labels
#         self.sep_indicies = sep_indicies
#         self.prediction_probs = prediction_probs
#         self.description_types = description_types
#         self.texts = texts
#         self.scores = {}
#         self.scores_chunk = {}
#         self.save_results = save_results
#
#
#         if 'multiclass' in args.mode:
#             #texts = [texts[i][1:] for i in range(len(texts))]
#             self.evaluate_transformer(args, labels, predictions, entity_labels, sep_indicies, texts)
#         else:
#             labels, predictions = self.convert_binary_to_types_tagger(labels, predictions, entity_labels, description_types, texts)
#             #Map predictions to sentence
#             self.evaluate_transformer(args, labels, predictions, entity_labels, sep_indicies, texts)
#
#
#     def save_results_to_file(self, labels, predictions):
#         dir = '../results'
#         f_name = os.path.join(dir, '_'.join([self.args.mode, self.args.model_type, self.args.dataset, self.args.split]))
#         with open(f_name, 'w') as f:
#             for i in range(len(labels)):
#                 f.write("{}\t{}\n".format(labels[i], predictions[i]))
#
#     def flat_labels(self, labels, predictions):
#         labels_flat = [item for sublist in labels for item in sublist]
#         predictions_flat = [item for sublist in predictions for item in sublist]
#         return labels_flat, predictions_flat
#
#     def evaluate_transformer(self, args, labels, predictions, entity_labels, sep_indicies, texts):
#         labels = [[tok for tok in sentence[:sep_indicies[i]]] for i, sentence in enumerate(labels)]
#         predictions = [[tok for tok in sentence[:sep_indicies[i]]] for i, sentence in enumerate(predictions)]
#
#         if texts != None:
#             labels_no_sub = [[tok for j, tok in enumerate(sentence[:sep_indicies[i]]) if '#' not in texts[i][j]] for i, sentence in enumerate(labels)]
#             predictions_no_sub = [[tok for j, tok in enumerate(sentence[:sep_indicies[i]]) if '#' not in texts[i][j]] for i, sentence in enumerate(predictions)]
#         else:
#             labels_no_sub = labels
#             predictions_no_sub = predictions
#
#         logger.info("######################### CHUNK BASED ####################################")
#         labels_iob, predictions_iob = self._convert_to_iob(labels_no_sub, predictions_no_sub, entity_labels)
#         f1 = seqeval.metrics.f1_score(labels_iob, predictions_iob)
#         acc = seqeval.metrics.accuracy_score(labels_iob, predictions_iob)
#         prec = seqeval.metrics.precision_score(labels_iob, predictions_iob)
#         recall = seqeval.metrics.recall_score(labels_iob, predictions_iob)
#         class_report = seqeval.metrics.classification_report(labels_iob, predictions_iob)
#         logger.info("acc: {}, rec: {}, precision: {}, f1 : {}".format(acc, recall, prec, f1))
#         logger.info(class_report)
#         print(class_report)
#
#
#         logger.info("######################### EXCLUDE SUBTOKENS################################")
#
#         labels, predictions = self.flat_labels(labels_no_sub, predictions_no_sub)
#         if self.save_results:
#             self.save_results_to_file(labels, predictions)
#         self.scores = self._calculate_scores(labels, predictions, entity_labels)
#
#         logger.info("######################### RECOGNITION CHUNK################################")
#
#         labels_iob, predictions_iob = self._convert_to_iob(labels_no_sub, predictions_no_sub, entity_labels, recognition = True)
#         f1 = seqeval.metrics.f1_score(labels_iob, predictions_iob)
#         acc = seqeval.metrics.accuracy_score(labels_iob, predictions_iob)
#         prec = seqeval.metrics.precision_score(labels_iob, predictions_iob)
#         recall = seqeval.metrics.recall_score(labels_iob, predictions_iob)
#         class_report = seqeval.metrics.classification_report(labels_iob, predictions_iob)
#         logger.info("acc: {}, rec: {}, precision: {}, f1 : {}".format(acc, recall, prec, f1))
#         logger.info(class_report)
#         print(class_report)
#
#         # self.scores_n = self._calculate_scores([1 if lab != 0 else 0 for lab in labels], [1 if lab != 0 else 0 for lab in predictions], ['NEG', 'ENT'])
#
#
#         logger.info("######################### RECOGNITION SUBTOKENS ################################")
#
#         labels, predictions = self.flat_labels(labels_no_sub, predictions_no_sub)
#         if self.save_results:
#             self.save_results_to_file(labels, predictions)
#         self.scores_n = self._calculate_scores([1 if lab != 0 else 0 for lab in labels], [1 if lab != 0 else 0 for lab in predictions], ['NEG', 'ENT'], recognition = True)
#
#         return self.scores, self.scores_chunk
#
#     def id_to_entity(self,labels, predictions):
#         label_types = [self.entity_labels[x] for x in labels]
#         prediction_types = [self.entity_labels[x] for x in predictions]
#         return label_types, prediction_types
#
#     def _calculate_scores(self, labels, predictions, entity_labels, recognition = False):
#         scores = {}
#         if 'multiclass' not in self.args.mode and not recognition:
#             label_types = [entity_labels[x-1] if x > 0 else "NEG" for x in labels]
#             prediction_types = [entity_labels[x-1] if x > 0 else "NEG" for x in predictions]
#         else:
#             label_types = [entity_labels[x] for x in labels]
#             prediction_types = [entity_labels[x] for x in predictions]
#
#
#         scores['acc'] = accuracy_score(label_types, prediction_types)
#         scores['recall'] = recall_score(label_types, prediction_types, average='micro')
#         scores['precision'] =  precision_score(label_types, prediction_types, average='micro')
#         scores['f1'] = f1_score(label_types, prediction_types, average='micro')
#         scores['f1_macro'] = f1_score(label_types, prediction_types, average='macro')
#
#         entity_labels2 = [x for x in entity_labels if x != 'NEG']
#         scores['recall_no_neg'] = recall_score(label_types, prediction_types, labels = entity_labels2, average='micro')
#         scores['precision_no_neg'] =  precision_score(label_types, prediction_types, labels = entity_labels2, average='micro')
#         scores['f1_no_neg'] = f1_score(label_types, prediction_types, labels = entity_labels2, average='micro')
#         scores['f1_macro_no_neg'] = f1_score(label_types, prediction_types, labels = entity_labels2, average='macro')
#         scores['acc_no_neg'] = accuracy_score([x for x in label_types if x != 'NEG'], [x for i, x in enumerate(prediction_types) if label_types[i] != 'NEG'])
#
#
#         logger.info("########## Multi-class scores ###############")
#         logger.info("acc: {}, rec: {}, precision: {}, f1 : {}, f1_macro : {}".format(scores['acc'], scores['recall'], scores['precision'], scores['f1'], scores['f1_macro']))
#         logger.info("######## Multi-class scores w/o neg class (O) #######")
#         logger.info("rec: {}, precision: {}, f1 : {}, f1_macro : {}, acc: {}".format(scores['recall_no_neg'], scores['precision_no_neg'], scores['f1_no_neg'], scores['f1_macro_no_neg'], scores['acc_no_neg']))
#
#         scores['report_results'] = classification_report(label_types, prediction_types, output_dict = True)
#
#         return scores
#
#
#     def _convert_to_iob(self, labels_no_sub, predictions_no_sub, entity_labels, recognition=False):
#         labels_iob = []
#         predictions_iob = []
#
#
#         for i in range(len(labels_no_sub)):
#             labels_sent = []
#             pred_sent = []
#             for j in range(len(labels_no_sub[i])):
#                 curr_lab = labels_no_sub[i][j] - (1 if 'multiclass' not in self.args.mode else 0)
#                 lab = entity_labels[curr_lab] if not recognition else 'ENT'
#                 if labels_no_sub[i][j] != 0:
#                     if j-1 < 0 or labels_no_sub[i][j -1] ==0:
#                         labels_sent.append('B-' + lab)
#                     else:# j+1 < len(labels_no_sub) and labels_no_sub[i][j +1] !=0:
#                         labels_sent.append('I-' + lab)
#                 else:
#                     labels_sent.append('O')
#
#             for j in range(len(predictions_no_sub[i])):
#                 curr_lab = predictions_no_sub[i][j] - (1 if 'multiclass' not in self.args.mode else 0)
#                 lab = entity_labels[curr_lab] if not recognition else 'ENT'
#                 # if recognition:
#                 #     lab = 'ENT' if labels_no_sub[i][j] >= 1 else 'NEG'
#                 if predictions_no_sub[i][j] != 0:
#                     if j-1 < 0 or predictions_no_sub[i][j -1] ==0:
#                         pred_sent.append('B-' + lab)
#                     else:# j+1 < len(labels_no_sub) and labels_no_sub[i][j +1] !=0:
#                         pred_sent.append('I-' + lab)
#                 else:
#                     pred_sent.append('O')
#             labels_iob.append(labels_sent)
#             predictions_iob.append(pred_sent)
#
#             # print(labels_iob)
#
#         return labels_iob, predictions_iob
#
#
#     #Convert binary predictions to entity type
#     def convert_binary_to_types_tagger(self, label, predictions, entity_labels, description_types, text = None):
#         predictions_types = []
#         label_types = []
#         for i, pred in enumerate(predictions):
#             label_list = []
#             for j, l in enumerate(pred):
#                 if l == 1:
#                     label_list.append(entity_labels.index(description_types[i]) + 1)
#                 else:
#                     label_list.append(0)
#             predictions_types.append(label_list)
#         for i, lab in enumerate(label):
#             label_list = []
#             for j, l in enumerate(lab):
#                 if l == 1:
#                     # print(entity_labels)
#                     # print(description_types[i])
#                     label_list.append(entity_labels.index(description_types[i])+ 1)
#                 else:
#                     label_list.append(0)
#             label_types.append(label_list)
#         return label_types, predictions_types
