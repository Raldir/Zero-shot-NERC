""" The trainer for the transformer and LSTM architectures"""

import torch
import argparse
import glob
import os
import random
import timeit
import sys
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import timeit
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.evaluation import evaluator_tagger
from models.transformers_ner import load_bert_model
from torch.optim import Adam

import logging
logger = logging.getLogger(__name__)
import numpy as np
import gensim
from models.batch_processor import BatchProcessor
from models.adamw import mAdamW


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


class Trainer(object):

    def __init__(self, args, dataloader, dataloader_dev = None):
        self.args = args
        self.model_type = args.model_type
        self.dataset = dataloader
        self.dataset_dev = dataloader_dev

        set_seed(self.args) # fixed seed to replicate results. During evaluation this line is commented out to measure scores across different seeds.

        #Load the respective model
        self.tokenizer = dataloader.tokenizer
        self.model = load_bert_model(args, len(self.dataset.entity_labels))

        self.device = args.device
        self.model.to(self.device)
        self.processor = BatchProcessor(args, self.model, self.dataset)
        if dataloader_dev != None:
            self.processor_dev = BatchProcessor(args, self.model, self.dataset_dev)

    def _init_fn(self):
        np.random.seed(self.args.seed)


    def train(self):
        """ Train the model """

        tb_writer = SummaryWriter(os.path.join(self.args.output_dir, 'events'))

        #reproductibility. Only model parameters are completely randomized. Rest uses fixed seed.
        set_seed(self.args)

        self.args.train_batch_size = self.args.per_gpu_train_batch_size
        train_sampler = RandomSampler(self.dataset)
        train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, collate_fn = self.processor.collator, worker_init_fn=self._init_fn)


        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        #Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr= self.args.learning_rate, eps=self.args.adam_epsilon)

        #optimizer = mAdamW(optimizer_grouped_parameters, lr= self.args.learning_rate, eps=self.args.adam_epsilon)

        #optimizer = Adam(optimizer_grouped_parameters, lr=self.args.learning_rate)

        #Add scheduler
        if self.args.scheduler == 'linear_scheduler':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        elif self.args.scheduler == 'plateau_scheduler':
            scheduler = ReduceLROnPlateau(optimizer, 'max', patience = 5, verbose = True)
        else:
            logger.info("No scheduler detected...")

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.args.output_dir, self.args.checkpoint, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.args.output_dir, self.args.checkpoint, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            logger.info("***** Loading optimizer and scheduler state from previous training *****")
            optimizer.load_state_dict(torch.load(os.path.join(self.args.output_dir, self.args.checkpoint, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.output_dir, self.args.checkpoint, "scheduler.pt")))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        num_patience_steps = 3
        early_stopping_patience = num_patience_steps
        validation_score = 0.0
        best_step = 0

        # Check if continuing training from a checkpoint
        if os.path.exists(self.args.output_dir):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = os.path.join(self.args.output_dir, self.args.checkpoint).split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // self.args.gradient_accumulation_steps)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(self.args.num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0]
        )
        # Added here for reproductibility
        set_seed(self.args)

        for ep in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=self.args.local_rank not in [-1, 0])
            preds = []
            all_labels = []

            if ep == 0:
                all_labels, preds, preds_prob, validation_loss = self.predict()
                acc = self.log_results(tb_writer, all_labels, preds, preds_prob, global_step)
                tb_writer.add_scalar("val_loss", validation_loss, global_step)

            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = self.processor.batch_to_dict(batch)

                if 'MRC' in self.args.model_type:
                    del inputs['labels']

                __, loss = self.model(**inputs)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()

                    if self.args.scheduler == 'linear_scheduler':
                        scheduler.step()  # Update learning rate schedule


                    self.model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if self.args.local_rank in [-1, 0] and  self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        all_labels, preds, preds_prob, validation_loss = self.predict()
                        acc = self.log_results(tb_writer, all_labels, preds, preds_prob, global_step)

                        if self.args.scheduler == 'plateau_scheduler':
                            scheduler.step(validation_loss)  # Update scheduler based on validation loss

                        logger.info("Writing into Summary...")
                        #tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.args.logging_steps, global_step)
                        print("Current loss", (tr_loss - logging_loss) / self.args.logging_steps, global_step)
                        tb_writer.add_scalar("val_loss", validation_loss, global_step)
                        logging_loss = tr_loss
                        #Only save model if validation score is higher
                        if self.args.early_stopping:
                            if acc > validation_score:
                                validation_score = acc
                                early_stopping_patience = num_patience_steps
                                best_step = global_step
                                self.save_model(optimizer, scheduler)

                            else:
                                early_stopping_patience-=1
                                if early_stopping_patience == 0:
                                    # self.save_model(optimizer, scheduler)
                                    logger.info("Validation score has not increased. Early stopping...!")
                                    tb_writer.close()
                                    del self.model
                                    return global_step, tr_loss / global_step, validation_score
        logger.info("Validation score has not increased. Early stopping...!")
        tb_writer.close()

        return global_step, tr_loss / global_step, validation_score

    #Save model
    def save_model(self, optimizer, scheduler):
        if self.args.model != 'transformer':
            return
        output_dir = os.path.join(self.args.output_dir, "checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        #torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))

        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    #Model prediction (either on dev if exists or on test)
    def predict(self):
        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size

        eval_sampler = SequentialSampler(self.dataset_dev if self.dataset_dev !=None else self.dataset)
        eval_dataloader = DataLoader(self.dataset_dev if self.dataset_dev !=None else self.dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, collate_fn = self.processor.collator if self.dataset_dev == None else self.processor_dev.collator)

          # Eval!
        logger.info("  Num examples = %d", len(self.dataset_dev) if self.dataset_dev !=None else len(self.dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        preds = []
        preds_prob = []
        start_time = timeit.default_timer()
        all_labels = []
        val_loss = 0.0

        count = 0
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            # count+=1
            # if count >10:
            #     break
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = self.processor.batch_to_dict(batch)
                all_labels += inputs['labels'].tolist()


                if 'MRC' in self.args.model_type:
                    del inputs['start_positions']
                    del inputs['end_positions']
                    del inputs['labels']
                elif self.dataset_dev == None:
                    del inputs['labels']

                # if self.dataset_dev == None:
                #     del inputs['labels']

                if 'MRC' in self.args.model_type:
                    start_logits, end_logits, span_logits = self.model(**inputs)
                    batch_size = start_logits.size(0)
                    seq_len = start_logits.size(-1)
                    preds_processed = []
                    for b in range(batch_size):
                        preds_cur = [0] * seq_len
                        for i in range(seq_len):
                            if start_logits[b][i] == 1:
                                preds_cur[i] = 1
                                for j in range(i, seq_len):
                                    if end_logits[b][j] == 1:
                                        for l in range(i, j+1):
                                            preds_cur[l] = 1
                                        break
                        preds_processed.append(preds_cur)
                    preds += preds_processed
                else:
                    outputs, loss = self.model(**inputs)

                if self.dataset_dev != None and not 'MRC'  in self.args.model_type:
                    val_loss += loss.item()

                if 'MRC' in self.args.model_type:
                    continue
                elif 'multiclass' in self.args.mode:
                    if 'tagger' in self.args.mode:
                        outputs = torch.argmax(outputs, dim=2)
                        preds += outputs.detach().cpu().numpy().tolist()
                    else:
                        outputs = torch.argmax(outputs, dim=1)
                        preds += outputs.detach().cpu().numpy().tolist()
                else:
                    if 'tagger' in self.args.mode:
                        outputs_max = torch.argmax(outputs, dim=2)
                        preds_prob += outputs.detach().cpu().numpy().tolist()
                        preds += outputs_max.detach().cpu().numpy().tolist()
                    else:
                        outputs_max = torch.argmax(outputs, dim=1)
                        preds_prob += [el[1] for el in outputs.detach().cpu().numpy().tolist()]
                        preds += outputs_max.detach().cpu().numpy().tolist()

                        # if len(preds) == 0:
                        #     preds.append(outputs.detach().cpu().numpy())
                        # else:
                        #     preds[0] = np.append(preds[0], outputs.detach().cpu().numpy(), axis=0)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(self.dataset))

        #     # if self.model_type in ['BertEntityOnlySigmoid']:
        #     #     preds_max = np.round(preds_prob)
        #     #     preds_prob = [ele for ele in preds_prob]

        return all_labels, preds, preds_prob, val_loss / (step + 1)


    #Log results in the SummaryWriter
    def log_results(self, tb_writer, labels, preds, preds_prob, ep):
        if 'tagger' in self.args.mode:
            entity_labels_dev = self.dataset_dev.entity_labels
            sep_indicies = [f.sep_index for f in self.dataset_dev.features]
            if 'multiclass' in self.args.mode:
                if 'few_shot' in self.args.mode:
                    entity_labels = self.dataset.entity_labels
                    evaluator = evaluator_tagger(self.args, labels, preds, entity_labels_dev, sep_indicies)
                else:
                    texts = [f.text_list[0] for f in self.dataset_dev.features]
                    evaluator = evaluator_tagger(self.args, labels, preds, entity_labels_dev, sep_indicies, texts)
            else:
                description_types_dev = [f.description_type for f in self.dataset_dev.features]
                texts = [f.text for f in self.dataset_dev.features]
                evaluator = evaluator_tagger(self.args, labels, preds, entity_labels_dev, sep_indicies, texts, preds_prob, description_types_dev)
        else:
            entity_labels_dev = self.dataset_dev.entity_labels
            if 'multiclass' in self.args.mode:
                # if self.args.model == 'lstm-crf':
                #     entity_labels = self.dataset.entity_labels
                #     evaluator = evaluator_classification(self.args, labels, preds, entity_labels)
                # else:
                evaluator = evaluator_classification(self.args, labels, preds, entity_labels_dev)
            else:
                description_types_dev = [f.description_type for f in self.dataset_dev.features]
                entity_num  = [f.entity_num for f in self.dataset_dev.features]
                evaluator = evaluator_classification(self.args, labels, preds, entity_labels_dev, preds_prob, description_types_dev, entity_num, use_probs = self.dataset_dev.add_negative_samples)

        scores = evaluator.scores
        logger.info("After Epoch {}, score is {}".format(ep, evaluator.scores['f1_macro_no_neg']))
        tb_writer.add_scalar("f1", scores['f1'], ep)
        tb_writer.add_scalar("recall", scores['recall'], ep)
        tb_writer.add_scalar("precision", scores['precision'], ep)
        tb_writer.add_scalar("f1_macro", scores['f1_macro'], ep)
        tb_writer.add_scalar("f1_no_neg", scores['f1_no_neg'], ep )
        tb_writer.add_scalar("f1_macro_no_neg", scores['f1_macro_no_neg'], ep )
        #tb_writer.add_scalar("f1_chunk_macro_no_neg", scores_chunk['f1_macro_no_neg'], ep )
        tb_writer.add_scalar("recall_no_neg", scores['recall_no_neg'], ep )
        tb_writer.add_scalar("precision_no_neg", scores['precision_no_neg'], ep)
        tb_writer.add_scalar("accuracy", scores['acc'], ep)
        # tb_writer.add_text("Classification_report", scores['report_results_formatted'], ep)
        #
        if 'tagger' in self.args.mode:
            scores_chunk = evaluator.scores_chunk
            tb_writer.add_scalar("f1_chunk",  scores_chunk['f1_macro_no_neg'], ep)
            tb_writer.add_scalar("recall_no_neg_chunk", scores_chunk['recall_no_neg'], ep)
            tb_writer.add_scalar("precision_no_neg_chunk",  scores_chunk['precision_no_neg'], ep)
        if 'multiclass' in self.args.mode and "CRF" not in self.args.model_type:
            if 'tagger' in self.args.mode:
                labels, preds = evaluator.flat_labels(labels, preds)
                labels, preds = evaluator.id_to_entity(labels,preds)
            else:
                labels, preds = evaluator.id_to_entity(labels,preds)

        return evaluator.scores['f1_macro_no_neg']
