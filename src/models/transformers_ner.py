""" Transformers for zero-shot NERC and NEC"""
import torch
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from torch.nn import CrossEntropyLoss, MSELoss
from utils.utils import RequiredParam
import sys
import os
import numpy as np
from typing import List, Optional
from models.entity_descriptions import get_entity_descriptions_negative
import random


def load_bert_model(args, num_classes= None):
    if args.model_type =="BertTagger":
        return  BertTagger.from_pretrained(os.path.join(args.output_dir, args.checkpoint)) if os.path.isfile(os.path.join(args.output_dir, args.checkpoint, "pytorch_model.bin")) else BertTagger.from_pretrained('../../../transformers/examples/mnli_output_large/', architectures = ["BertTagger"], finetuning_task = {"dropout_prob": args.linear_dropout, "transformer_finetune": True, "num_labels":2, "dataset" : args.dataset })
    elif args.model_type =="BertTaggerMRC":
        return  BertTagger.from_pretrained(os.path.join(args.output_dir, args.checkpoint)) if os.path.isfile(os.path.join(args.output_dir, args.checkpoint, "pytorch_model.bin")) else BertTagger.from_pretrained('bert-large-cased', architectures = ["BertTaggerMRC"], finetuning_task = {"dropout_prob": args.linear_dropout, "transformer_finetune": True, "num_labels":2, "dataset" : args.dataset})
    elif args.model_type == "BertTaggerMultiClass":
        return  BertTaggerMultiClass.from_pretrained(os.path.join(args.output_dir, args.checkpoint), output_hidden_states=True) if os.path.isfile(os.path.join(args.output_dir, args.checkpoint, "pytorch_model.bin")) else BertTaggerMultiClass.from_pretrained('bert-large-cased', architectures = ["BertTaggerMultiClass"], output_hidden_states=True, finetuning_task = {"dropout_prob": args.linear_dropout, "transformer_finetune": True, "num_labels":num_classes, "no_cuda":args.no_cuda, "description_mode" : args.entity_descriptions_mode, "dataset" : args.dataset, "filter_classes": 'filtered_classes' in args.mode, "dataset" : args.dataset})
    elif args.model_type == "BertTaggerMultiClassIndependent":
        return  BertTaggerMultiClassIndependent.from_pretrained(os.path.join(args.output_dir, args.checkpoint)) if os.path.isfile(os.path.join(args.output_dir, args.checkpoint, "pytorch_model.bin")) else BertTaggerMultiClassIndependent.from_pretrained('bert-large-cased', architectures = ["BertTaggerMultiClassAtlernative"], finetuning_task = {"dropout_prob": args.linear_dropout, "transformer_finetune": True, "num_labels":num_classes, "no_cuda":args.no_cuda, "description_mode" : args.entity_descriptions_mode, "dataset" : args.dataset, "filter_classes": 'filtered_classes' in args.mode})
    elif args.model_type == "BertTaggerMultiClassDescription":
        return  BertTaggerMultiClassDescription.from_pretrained(os.path.join(args.output_dir, args.checkpoint)) if os.path.isfile(os.path.join(args.output_dir, args.checkpoint, "pytorch_model.bin")) else BertTaggerMultiClassDescription.from_pretrained('bert-large-cased', architectures = ["BertTaggerMultiClassDescription"], finetuning_task = {"dropout_prob": args.linear_dropout, "transformer_finetune": True, "num_labels":num_classes, "no_cuda":args.no_cuda, "description_mode" : args.entity_descriptions_mode, "dataset" : args.dataset, "filter_classes": 'filtered_classes' in args.mode})


    else:
        print("MODEL TYPE NOT FOUND, ABORT...")
        sys.exit()

class BertTaggerMRC(BertPreTrainedModel):
    """
    @author Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu and Jiwei Li
    https://github.com/ShannonAI/mrc-for-flat-nested-ner
    """
    def __init__(self, config):

        super().__init__(config)
        self.bert = BertModel(config)
        self.start_outputs = torch.nn.Linear(config.hidden_size, 2)
        self.end_outputs = torch.nn.Linear(config.hidden_size, 2)
        self.use_symbols = False
        dataset =  config.finetuning_task['dataset']
        # self.device = 'cpu' if config.finetuning_task['no_cuda'] else 'cuda:0'

        self.span_embedding = MultiNonLinearClassifier(config.hidden_size*2, 1, config.finetuning_task['dropout_prob'])
        self.drop = torch.nn.Dropout(config.finetuning_task['dropout_prob'])
        self.hidden_size = config.hidden_size
        self.class_weight = [0.1 if self.dataset == 'medmentions' else 0.01, 1]
        self.class_weight =torch.FloatTensor(self.class_weight).cuda()
        self.loss_wb = 1.0
        self.loss_we = 1.0
        self.loss_ws = 1.0
        self.num_labels = 2
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
        start_positions=None, end_positions=None, span_positions=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            span_positions: (batch x max_len x max_len)
                span_positions[k][i][j] is one of [0, 1],
                span_positions[k][i][j] represents whether or not from start_pos{i} to end_pos{j} of the K-th sentence in the batch is an entity.
        """

        sequence_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0]

        sequence_heatmap = sequence_output # batch x seq_len x hidden
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(self.drop(sequence_heatmap)) # batch x seq_len x 2
        end_logits = self.end_outputs(self.drop(sequence_heatmap)) # batch x seq_len x 2

        start_logits = self.softmax(start_logits)
        end_logits = self.softmax(end_logits)

        loss = 0

        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss(weight = self.class_weight)

            active_loss = attention_mask.view(-1) == 1
            active_loss2 = token_type_ids.view(-1) == 0

            active_logits = start_logits.view(-1, self.num_labels)
            active_start_labels = torch.where(torch.mul(active_loss, active_loss2), start_positions.view(-1), torch.tensor(loss_fct.ignore_index).type_as(start_positions))

            start_loss = loss_fct(active_logits, active_start_labels)

            active_logits = end_logits.view(-1, self.num_labels)
            active_end_labels = torch.where(torch.mul(active_loss, active_loss2), end_positions.view(-1), torch.tensor(loss_fct.ignore_index).type_as(end_positions))


            end_loss = loss_fct(active_logits, active_end_labels)

            total_loss = self.loss_wb * start_loss + self.loss_we * end_loss #+ self.loss_ws * span_loss
            # print(start_loss, end_loss, span_loss)
            return None, total_loss
        else:

            start_logits = torch.argmax(start_logits, dim=2).bool()
            end_logits = torch.argmax(end_logits, dim=2).bool()
            return start_logits, end_logits, None #span_logits


class BertTaggerMultiClassIndependent(BertPreTrainedModel):
  def __init__(self, config):

    super().__init__(config)

    self.bert = BertModel(config)#from_pretrained(config.finetuning_task['model_name'], output_hidden_states = config.output_hidden_states, output_attentions = config.output_attentions)
    self.num_labels = config.finetuning_task['num_labels']
    self.drop = torch.nn.Dropout(config.finetuning_task['dropout_prob'])
    self.bert_output_size = config.hidden_size
    self.linear = torch.nn.Linear(self.bert_output_size, 1) #Number of classes 2
    self.linear_zero = torch.nn.Linear(self.bert_output_size, 1) #Number of classes 2
    self.softmax = torch.nn.LogSoftmax(dim=2)
    self.finetune = config.finetuning_task['transformer_finetune']
    self.use_symbols = False
    self.dataset = config.finetuning_task['dataset']
    self.class_weight = [0.1 if self.dataset == 'medmentions' else 0.01] + [1] * (self.num_labels - 1)
    self.class_weight =torch.FloatTensor(self.class_weight).cuda()

    # self.device = 'cpu' if config.finetuning_task['no_cuda'] else 'cuda:0'
    self.description_mode = config.finetuning_task['description_mode']
    self.filter_classes = config.finetuning_task['filter_classes']

    self.init_weights()

  def forward(self, *args, input_ids = RequiredParam, attention_mask= RequiredParam, token_type_ids= RequiredParam, sep_index = RequiredParam, seq_mask = RequiredParam, split = RequiredParam, labels = None, **kwargs):
    # seq_transformer = batch_size x max_seq_length (padded) : sentence
    sep_index_max = torch.max(sep_index)
    predictions = []
    #sel = random.randint(0, input_ids.size(0) - 1)
    for j in range(input_ids.size(0)):
        if j == 0:
            inp_zero = torch.stack([input_ids[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
            att_zero = torch.stack([attention_mask[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
            tok_type_zero = torch.stack([token_type_ids[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
            if not self.finetune:
                with torch.no_grad():
                    words_out = self.bert(input_ids = inp_zero, attention_mask = att_zero, token_type_ids = tok_type_zero)[0]
            else:
                words_out  = self.bert(input_ids = inp_zero, attention_mask = att_zero, token_type_ids = tok_type_zero)[0]

            pooled_out = self.drop(words_out)
            logits = self.linear_zero(pooled_out)
            predictions.append(logits)
        else:
            if not self.finetune:
                with torch.no_grad():
                    words_out = self.bert(input_ids = input_ids[j], attention_mask = attention_mask[j], token_type_ids = token_type_ids[j])[0]
            else:
                words_out  = self.bert(input_ids = input_ids[j], attention_mask = attention_mask[j], token_type_ids = token_type_ids[j])[0]

            words_out = torch.stack([words_out[i, :sep_index_max.item(), :] for i in range(words_out.size(0))])

            pooled_out = self.drop(words_out)
            logits = self.linear(pooled_out)

            predictions.append(logits)
        #predictions_zero.append(logits[:,:, 0])

    predictions = torch.stack(predictions)
    predictions = predictions.transpose(0,1).transpose(1,2).squeeze(3)

    logits = self.softmax(predictions)

    loss = 0
    if labels is not None:
        labels = torch.stack([labels[i, :sep_index_max.item()] for i in range(labels.size(0))])
        loss_fct = CrossEntropyLoss(weight = self.class_weight if split.item() == 0 else torch.FloatTensor([1] * logits.size(2)).cuda())
        active_logits = logits.view(-1, input_ids.size(0))
        active_labels = labels.view(-1)
        loss = loss_fct(active_logits, active_labels)

    return logits, loss

class BertTaggerMultiClass(BertPreTrainedModel):
  '''
   Use plain Bert Model for Textual Inference
   see https://github.com/huggingface/pytorch-pretrained-BERT/blob/ee0308f79ded65dac82c53dfb03e9ff7f06aeee4/pytorch_pretrained_bert/modeling.py#L938
   Small changes to the model described in the paper: max pooling over both class representations and negative representation from independent encoding -- resulted in slightly better results.
  '''

  def __init__(self, config):

    super().__init__(config)

    self.bert = BertModel(config)#from_pretrained(config.finetuning_task['model_name'], output_hidden_states = config.output_hidden_states, output_attentions = config.output_attentions)
    self.num_labels = config.finetuning_task['num_labels']#11#4#11#4#11#11
    self.drop = torch.nn.Dropout(config.finetuning_task['dropout_prob'])
    self.bert_output_size = config.hidden_size
    self.linear = torch.nn.Linear(self.bert_output_size, 1) #Number of classes 2
    self.linear_zero = torch.nn.Linear(self.num_labels *self.bert_output_size, 1) #Number of classes 2
    self.linear_zero2 = torch.nn.Linear(self.bert_output_size, 1) #Number of classes 2
    self.linear_zero3 = torch.nn.Linear(self.bert_output_size, 1) #Number of classes 2
    self.softmax = torch.nn.LogSoftmax(dim=2)
    self.finetune = config.finetuning_task['transformer_finetune']
    self.use_symbols = False
    self.dataset = config.finetuning_task['dataset']
    self.class_weight = [0.1 if self.dataset == 'medmentions' else 0.01]  + [1] * (self.num_labels - 1)# 0.01, 1 1 1 1 #0.1 for medmentions, ontonotes #FOR MEDMENTIONS use 0.1
    self.class_weight =torch.FloatTensor(self.class_weight).cuda()

    #self.device = 'cpu' if config.finetuning_task['no_cuda'] else 'cuda:0'
    self.description_mode = config.finetuning_task['description_mode']
    self.dataset = config.finetuning_task['dataset']
    self.filter_classes = config.finetuning_task['filter_classes']

    self.init_weights()

  def forward(self, *args, input_ids = RequiredParam, attention_mask= RequiredParam, token_type_ids= RequiredParam, sep_index = RequiredParam, seq_mask = RequiredParam, split = RequiredParam, labels = None, **kwargs):
    # seq_transformer = batch_size x max_seq_length (padded) : sentence
    sep_index_max = torch.max(sep_index)
    predictions = []
    predictions_zero = []
    predictions_zero_base = []
    for j in range(input_ids.size(0)):
        if j == 0:
            inp_zero = torch.stack([input_ids[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
            att_zero = torch.stack([attention_mask[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
            tok_type_zero = torch.stack([token_type_ids[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
            if not self.finetune:
                with torch.no_grad():
                    words_out = self.bert(input_ids = inp_zero, attention_mask = att_zero, token_type_ids = tok_type_zero)[0]
            else:
                words_out  = self.bert(input_ids = inp_zero, attention_mask = att_zero, token_type_ids = tok_type_zero)[0]

            pooled_out = self.drop(words_out)
            logits = self.linear_zero3(pooled_out)
            predictions_zero_base.append(logits)
        else:
            if not self.finetune:
                with torch.no_grad():
                    words_out = self.bert(input_ids = input_ids[j], attention_mask = attention_mask[j], token_type_ids = token_type_ids[j])[0]
            else:
                words_out = self.bert(input_ids = input_ids[j], attention_mask = attention_mask[j], token_type_ids = token_type_ids[j])[0] #res, layer_out

            words_out = torch.stack([words_out[i, :sep_index_max.item(), :] for i in range(words_out.size(0))])
            pooled_out = self.drop(words_out)
            predictions_zero.append(self.linear_zero2(pooled_out))
            logits = self.linear(pooled_out)
            predictions.append(logits)


    random.shuffle(predictions_zero)
    predictions_zero = torch.stack(predictions_zero_base + predictions_zero)#.transpose(0,1).transpose(1,2)
    predictions_zero = predictions_zero.transpose(0,1).transpose(1,2)
    predictions_zero = predictions_zero.contiguous().view(predictions_zero.size(0), predictions_zero.size(1), -1)


    predictions_zero = torch.max(predictions_zero, dim=2)[0].unsqueeze(2)

    predictions = torch.stack(predictions)
    predictions = predictions.transpose(0,1).transpose(1,2).squeeze(3)

    #logits = self.softmax(torch.cat((predictions_zero, predictions), dim =2))
    logits = torch.cat((predictions_zero, predictions), dim =2)

    loss = 0

    if labels is not None:

        labels = torch.stack([labels[i, :sep_index_max.item()] for i in range(labels.size(0))])
        # print(self.class_weight)
        loss_fct = CrossEntropyLoss(weight = self.class_weight if split.item() == 0 else torch.FloatTensor([1] * logits.size(2)).cuda())
        active_logits = logits.view(-1, logits.size(2))
        active_labels = labels.view(-1)
        loss = loss_fct(active_logits, active_labels)

    return logits, loss



class BertTaggerMultiClassDescription(BertPreTrainedModel):

  def __init__(self, config):

    super().__init__(config)

    self.bert = BertModel(config)#from_pretrained(config.finetuning_task['model_name'], output_hidden_states = config.output_hidden_states, output_attentions = config.output_attentions)
    self.num_labels = config.finetuning_task['num_labels']
    self.drop = torch.nn.Dropout(config.finetuning_task['dropout_prob'])
    self.bert_output_size = config.hidden_size
    self.linear = torch.nn.Linear(self.bert_output_size, self.num_labels) #Number of classes 2
    self.softmax = torch.nn.LogSoftmax(dim=2)
    self.finetune = config.finetuning_task['transformer_finetune']
    self.use_symbols = False
    self.dataset = config.finetuning_task['dataset']
    self.class_weight = [0.1 if self.dataset == 'medmentions' else 0.01]  + [1] * (self.num_labels - 1) # 0.01, 2 OntoNotes 0.01 medMentions
    self.class_weight =torch.FloatTensor(self.class_weight).cuda()

    # self.device = 'cpu' if config.finetuning_task['no_cuda'] else 'cuda:0'
    self.description_mode = config.finetuning_task['description_mode']
    self.dataset = config.finetuning_task['dataset']
    self.filter_classes = config.finetuning_task['filter_classes']

    self.init_weights()

  def forward(self, *args, input_ids = RequiredParam, attention_mask= RequiredParam, token_type_ids= RequiredParam, sep_index = RequiredParam, seq_mask = RequiredParam, split = RequiredParam, labels = None, **kwargs):
    # seq_transformer = batch_size x max_seq_length (padded) : sentence
    sep_index_max = torch.max(sep_index)
    predictions = []
    #sel = random.randint(0, input_ids.size(0) - 1)
    for i in range(input_ids.size(0)):
        if not self.finetune:
            with torch.no_grad():
                words_out = self.bert(input_ids = input_ids[i], attention_mask = attention_mask[i], token_type_ids = token_type_ids[i])[0]
        else:
            words_out  = self.bert(input_ids = input_ids[i], attention_mask = attention_mask[i], token_type_ids = token_type_ids[i])[0]

        words_out = torch.stack([words_out[i, :sep_index_max.item(), :] for i in range(words_out.size(0))])

        pooled_out = self.drop(words_out)
        logits = self.linear(pooled_out)
        predictions.append(logits[:,:, 1])

    #print(torch.stack(predictions).size())
    logits = self.softmax(torch.stack(predictions).transpose(0,1).transpose(1,2))
    #logits = self.softmax(torch.stack(predictions).transpose(0,3).squeeze(0))

    #print(logits.size())

    loss = 0
    if labels is not None:
        labels = torch.stack([labels[i, :sep_index_max.item()] for i in range(labels.size(0))])
        loss_fct = CrossEntropyLoss(weight = self.class_weight if split.item() == 0 else torch.FloatTensor([1] * logits.size(2)).cuda())
        active_logits = logits.view(-1, input_ids.size(0))
        active_labels = labels.view(-1)
        loss = loss_fct(active_logits, active_labels)

    return logits, loss


class BertTagger(BertPreTrainedModel):

  def __init__(self, config):

    super().__init__(config)

    self.bert = BertModel(config)#from_pretrained(config.finetuning_task['model_name'], output_hidden_states = config.output_hidden_states, output_attentions = config.output_attentions)
    self.num_labels = config.finetuning_task['num_labels']
    self.drop = torch.nn.Dropout(config.finetuning_task['dropout_prob'])
    self.bert_output_size = config.hidden_size
    self.linear = torch.nn.Linear(self.bert_output_size, self.num_labels) #Number of classes 2
    self.softmax = torch.nn.LogSoftmax(dim=2)
    self.finetune = config.finetuning_task['transformer_finetune']
    self.dataset = config.finetuning_task['dataset']
    self.class_weight = [0.1 if self.dataset == 'medmentions' else 0.01, 1] # 0.01, 2
    self.class_weight =torch.FloatTensor(self.class_weight).cuda()


    self.init_weights()

  def forward(self, *args, input_ids = RequiredParam, attention_mask= RequiredParam, token_type_ids= RequiredParam, labels=None, **kwargs):
    # seq_transformer = batch_size x max_seq_length (padded) : sentence
    if not self.finetune:
        with torch.no_grad():
            word_out = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0]
    else:
        word_out = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0]

    pooled_out = self.drop(word_out)
    logits = self.softmax(self.linear(word_out))

    loss = 0
    if labels is not None:
        loss_fct = CrossEntropyLoss(weight = self.class_weight)
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_loss2 = token_type_ids.view(-1) == 0
            #print(torch.mul(active_loss, active_loss2))
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                torch.mul(active_loss, active_loss2), labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    return logits, loss
