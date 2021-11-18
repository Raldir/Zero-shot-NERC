Main components of the source code for the paper [Leveraging Type Descriptions for Zero-shot Named Entity Recognition and Classification](https://aclanthology.org/2021.acl-long.120/), published in ACL2021.

If you use the code in this repository, e.g. as a baseline in your experiment or simply want to refer to this work, we kindly ask you to use the following citation:

```
@inproceedings{aly-etal-2021-leveraging,
    title = "Leveraging Type Descriptions for Zero-shot Named Entity Recognition and Classification",
    author = "Aly, Rami  and
      Vlachos, Andreas  and
      McDonald, Ryan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.120",
    doi = "10.18653/v1/2021.acl-long.120",
    pages = "1516--1528",
}
```

## Installation:

For running the entire codebase the libraries sklearn, gensim, spacy, nltk, pytorch, and huggingface's transformer library need to be installed. To simply run the main experiments, it should be sufficient to install sklearn, pytorch, and the huggingface library. Code has been tested only on python3.7. Code is bound to be running on cuda, so GPU is needed.

## Dataset:
The scripts for converting both `Medmentions` and `Ontonotes` to their zero-shot version are provided and can be applied to any dataset. The only requirement is that the input is provided in IOB format.  Call the script `generate_datasets.py` with the argument `zero-shot` and the respective dataset and the input/output paths. Due to not owning the datasets (nor entire descriptions) themselves, the zero-shot versions of the datasets are not provided here. The OntoNotes dataset has to be obtained through LDC and then [converted](https://cemantix.org/data/ontonotes.html). MedMentions dataset is available [here](https://github.com/chanzuckerberg/MedMentions). Sample descriptions as a reference are found in `data/entity_descriptions`.

## Models:

### SMXM

There are three SMXM models for the full NERC task: `BertTaggerMultiClassDescription` (correpsonds to approach i 'Description-based'), `BertTaggerMultiClassIndependent` (corresponds to approach 'Independent'), and `BertTaggerMultiClass` (corresponds to approach 'class-aware').
Example command for OntoNotes:

```
python3.7 run.py --config_mode BertTaggerMultiClass --overwrites [mode:tagger_multiclass_filtered_classes,entity_descriptions_mode:annotation_guidelines,per_gpu_train_batch_size:7] --config_files [config03]
```

The pre-trained SMXM Model for OntoNotes can be downloaded [here]( https://drive.google.com/file/d/1PGEyBsuc6n085j9kZ5TtkAV7hC5mggdd/view?usp=sharing). To run the model in inference on the test data call:

```
python3.7 test.py --split conll-2012-test --mode tagger_multiclass_filtered_classes --model transformer --dataset ontonotes --output_dir ../dumpe/BertTaggerMultiClass_config03_mode_tagger_multiclass_filtered_classes__entity_descriptions_mode_annotation_guidelines__per_gpu_train_batch_size_7/ --entity_descriptions_mode annotation_guidelines --max_sequence_length 300 --max_description_length 150 --mask_entity_partially --mask_probability 0.7 --model_type BertTaggerMultiClass --checkpoint checkpoin
```

### BEM

To run this baseline, it is required to first fine-tune a BERT model on SNLI. This can be done easily using scripts found in huggingface's repository. The path to that model needs to be set in `models/transformers_ner`.
```
python3.7 run.py --config_mode BertTagger --overwrites [mode:tagger_filtered_classes,dataset:ontonotes,entity_descriptions_mode:entailment,per_gpu_train_batch_size:20,description:bert-large] --config_files [config19]
```

### MRC for NERC

```
python3.7 run.py --config_mode BertTaggerMRC --overwrites [mode:tagger_filtered_classes,dataset:ontonotes,entity_descriptions_mode:annotation_guidelines,per_gpu_train_batch_size:20,description:bert-large] --config_files [config19]
```

## Contact:

Feel free to reach out over email at rmya2@cam.ac.uk
