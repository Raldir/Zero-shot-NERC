""" Prepares batch data for different models"""
from models.entity_descriptions import get_symbols
import gensim
import torch
import numpy as np
import sys


class BatchProcessor(object):

    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.model_type = args.model_type
        self.device = args.device
        self.model_groups = {
                'tagger_mrc': ['BertTaggerMRC'],
                'tagger' : ["BertTagger", "BertTaggerFewShot", "BertTaggerCRFFewShot"],
                'tagger_symbol' : ["BertTaggerSymbol", "BertTaggerSymbolCRF", "BertTaggerSymbolCRFNew"],
                'tagger_multiclass' : ["BertTaggerMultiClassCRF", "BertTaggerMultiClass", "BertTaggerMultiClassSoftmaxNoNegative", "BertTaggerMultiClassAlternative", 'BertTaggerMultiClassSoftmax'],
                'tagger_symbol_multiclass': ["BertTaggerSymbolMultiClass"],
                'classification_symbol' :  ['BertEntityOnlySymbol', 'BertEntitySymbol'],
                'classification' : ["BertEntity", 'BertEntityOnly', 'BertEntityOnlySigmoid', 'BertBase'],
                'classification_seperate': ['BertSeperate'],
                'classification_symbol_multiclass' : ['BertEntityOnlySymbolMultiClass', 'BertEntitySymbolMultiClass'],
                'classification_multiclass' : ['BertEntityMultiClass'],
                'classification_lstm_few_shot': ["LSTMClassification"],
                'tagger_lstm_few_shot': ["LSTMTagger", "LSTMTaggerCRF", "LSTMTagger"]
            }
        if self.model.use_symbols:
            self.symbol_embeddings = {}
            self.symbol_embeddings_list = []
            embeddings = gensim.models.KeyedVectors.load_word2vec_format(self.args.embedding_path, limit = 10000) # 5000000
            symbols = get_symbols(args.symbols_file, self.dataset.dataset)
            for label in self.dataset.entity_labels:
                if label in symbols:
                    emb = [embeddings[sym]  for sym in symbols[label] if sym in embeddings]
                    if label in ["PERSON", 'PRODUCT']:
                        emb = list(np.mean(emb, axis = 0))
                    else:
                        emb = [item for sublist in emb[:self.args.max_symbol_length] for item in sublist]
                    self.symbol_embeddings[label] = emb
                    self.symbol_embeddings_list.append(emb)

    def collator(self, data):
        if self.model_type in self.model_groups['tagger']:
            batch = self.tagger_collator(data)
        elif self.model_type in self.model_groups['tagger_multiclass']:
            batch = self.tagger_multiclass_collator(data)
        elif self.model_type in self.model_groups['tagger_mrc']:
            batch = self.tagger_mrc_collator(data)
        return batch

    def batch_to_dict(self, batch):
        if self.model_type in self.model_groups['tagger']:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "seq_mask": batch[3],
                "labels": batch[4],
            }
        elif self.model_type in self.model_groups['tagger_mrc']:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions" : batch[3],
                "end_positions" : batch[4],
                "span_positions" : batch[5],
                "labels": batch[6],
            }

        elif self.model_type in self.model_groups['tagger_multiclass']:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "sep_index" : batch[3],
                "seq_mask" : batch[4],
                "split" : batch[5],
                "labels": batch[6],
            }

        return inputs

    def tagger_collator(self, data):
        input_ids = [f.input_ids for f in data]
        input_masks = [f.input_mask for f in data]
        segment_ids = [f.segment_ids for f in data]
        label_ids = [f.label_id for f in data]
        batch_size = len(input_ids)

        sentence_lengths = [len(bert_tokens) for bert_tokens in input_ids]
        longest_sent = max(sentence_lengths)

        padded_bert_tokens = []
        padded_bert_masks = []
        padded_bert_segments = []
        padded_bert_labels = []

        for i in range(batch_size):
            padding = [0] * (longest_sent - len(input_ids[i]))
            padding_segment = [1] * (longest_sent - len(input_ids[i]))
            padded_bert_tokens.append(input_ids[i] + padding)
            padded_bert_masks.append(input_masks[i] + padding)
            padded_bert_segments.append(segment_ids[i] + padding_segment)
            padded_bert_labels.append(label_ids[i] + padding)

        bert_tokens_t = torch.tensor(padded_bert_tokens).to(device=self.device)
        bert_masks_t = torch.tensor(padded_bert_masks).to(device=self.device)
        bert_segments_t = torch.tensor(padded_bert_segments).to(device=self.device)
        seq_mask =  torch.tensor(padded_bert_masks).to(device=self.device, dtype=torch.uint8)
        bert_labels_t = torch.tensor(padded_bert_labels).to(device=self.device)

        return bert_tokens_t, bert_masks_t, bert_segments_t, seq_mask,bert_labels_t

    def tagger_mrc_collator(self, data):
        input_ids = [f.input_ids for f in data]
        input_masks = [f.input_mask for f in data]
        segment_ids = [f.segment_ids for f in data]
        label_ids = [f.label_id for f in data]
        batch_size = len(input_ids)

        sentence_lengths = [len(bert_tokens) for bert_tokens in input_ids]
        longest_sent = max(sentence_lengths)

        padded_bert_tokens = []
        padded_bert_masks = []
        padded_bert_segments = []
        padded_bert_start_positions = []
        padded_bert_end_positions = []
        padded_bert_span_positions = []
        padded_bert_labels = []

        for i in range(batch_size):
            padding = [0] * (longest_sent - len(input_ids[i]))
            padding_segment = [1] * (longest_sent - len(input_ids[i]))
            padded_bert_tokens.append(input_ids[i] + padding)
            padded_bert_masks.append(input_masks[i] + padding)
            padded_bert_segments.append(segment_ids[i] + padding_segment)
            padded_bert_labels.append(label_ids[i] + padding)

            # padded_bert_start_positions.append([1 if label_ids[i][0] == 1 else 0] + [0 if el == 0 or label_ids[i][j-1] == 1 else 1 for j,el in enumerate(label_ids[i][1:])] + padding)
            padded_bert_start_positions_i = []
            prev = 0
            for j, el in enumerate(padded_bert_labels[i]):
                if el == 1 and prev == 0:
                    padded_bert_start_positions_i.append(1)
                    prev = 1
                elif el == 1:
                    padded_bert_start_positions_i.append(0)
                else:
                    prev = 0
                    padded_bert_start_positions_i.append(0)
            padded_bert_start_positions.append(padded_bert_start_positions_i)


            padded_bert_end_positions_i = []
            prev = 0
            for j, el in enumerate(reversed(padded_bert_labels[i])):
                if el == 1 and prev == 0:
                    padded_bert_end_positions_i.append(1)
                    prev = 1
                elif el == 1:
                    padded_bert_end_positions_i.append(0)
                else:
                    prev = 0
                    padded_bert_end_positions_i.append(0)
            padded_bert_end_positions_i.reverse()
            padded_bert_end_positions.append(padded_bert_end_positions_i)

            # padded_bert_end_positions.append([0 if el == 0 or label_ids[i][j+1] == 1 else 1 for j,el in enumerate(label_ids[i][:-1])] + [1 if label_ids[i][len(label_ids[i]) -1] == 1 else 0] + padding)
            padded_bert_span_positions.append([])


        bert_tokens_t = torch.tensor(padded_bert_tokens).to(device=self.device)
        bert_masks_t = torch.tensor(padded_bert_masks).to(device=self.device)
        bert_segments_t = torch.tensor(padded_bert_segments).to(device=self.device)
        bert_start_positions_t = torch.tensor(padded_bert_start_positions).to(device=self.device)
        bert_end_positions_t = torch.tensor(padded_bert_end_positions).to(device=self.device)
        bert_span_positions_t = torch.tensor(padded_bert_span_positions).to(device=self.device)
        bert_labels_t = torch.tensor(padded_bert_labels).to(device=self.device)

        return bert_tokens_t, bert_masks_t, bert_segments_t, bert_start_positions_t, bert_end_positions_t, bert_span_positions_t, bert_labels_t


    def tagger_multiclass_collator(self, data):
            input_ids_lists = [f.input_ids_list for f in data]
            input_masks_lists = [f.input_mask_list for f in data]
            segment_ids_lists = [f.segment_ids_list for f in data]
            label_ids = [f.label_id for f in data]
            label_dict = [f.label_dict for f in data]
            batch_size = len(input_ids_lists)

            sep_index = [f.sep_index for f in data]
            max_sep_index = max(sep_index)

            #print(max_sep_index)

            padded_bert_tokens_list = []
            padded_bert_masks_list = []
            padded_bert_segments_list = []

            padded_bert_labels = []
            padded_sequence_mask = []

            sentence_lengths = [max([len(bert_tokens) for bert_tokens in input_ids_lists[i]]) for i in range(len(input_ids_lists))]
            longest_sent = max(sentence_lengths)

            for i in range(batch_size):
                padded_bert_tokens = []
                padded_bert_masks = []
                padded_bert_segments = []

                padding_label = [0] * (max_sep_index - len(label_ids[i]))
                for j in range(len(input_ids_lists[i])):
                    padding = [0] * (longest_sent - len(input_ids_lists[i][j]))
                    padded_bert_tokens.append(input_ids_lists[i][j] + padding)
                    padded_bert_masks.append(input_masks_lists[i][j] + padding)
                    padded_bert_segments.append(segment_ids_lists[i][j] + padding)

                bert_tokens_t = torch.tensor(padded_bert_tokens).to(device=self.device)
                bert_masks_t = torch.tensor(padded_bert_masks).to(device=self.device)
                bert_segments_t = torch.tensor(padded_bert_segments).to(device=self.device)

                padded_bert_tokens_list.append(bert_tokens_t)
                padded_bert_masks_list.append(bert_masks_t)
                padded_bert_segments_list.append(bert_segments_t)

                padded_bert_labels.append(label_ids[i] + padding_label) #MODIFY BACK
                padded_sequence_mask.append(sep_index[i] * [1] + (max_sep_index - sep_index[i]) *[0])

            bert_tokens_t = torch.stack(padded_bert_tokens_list).transpose(1,0)
            bert_masks_t = torch.stack(padded_bert_masks_list).transpose(1,0)
            bert_segments_t = torch.stack(padded_bert_segments_list).transpose(1,0)
            bert_labels_t = torch.tensor(padded_bert_labels).to(device=self.device, dtype=torch.long)
            bert_seq_mask_t = torch.tensor(padded_sequence_mask).to(device=self.device,dtype=torch.uint8)
            bert_sep_t = torch.tensor(sep_index).to(device=self.device)

            if self.dataset.split == 'train':
                split = 0
            elif self.dataset.split == 'dev':
                split = 1
            else:
                split = 2
            split = torch.tensor(split).to(device=self.device)

            return bert_tokens_t, bert_masks_t, bert_segments_t, bert_sep_t, bert_seq_mask_t, split, bert_labels_t
