import csv
import torch
import json
import re
import os
import pickle
import random
import numpy as np
from transformers import BertTokenizerFast
from pathlib import Path
from copy import deepcopy



class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, seq_labels, ner_labels):
        self.encodings = encodings
        self.seq_labels = seq_labels
        self.ner_labels = ner_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['seq_labels'] = torch.tensor(self.seq_labels[idx])
        item['ner_labels'] = torch.tensor(self.ner_labels[idx])
        return item

    def __len__(self):
        return len(self.seq_labels)


def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


# general token-level labels
def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        max_len = len(doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)])
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels[:max_len]
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def load_seq_data_from_tsv(path):
    file = open(path, "r", encoding="utf-8-sig")
    lines = list(csv.reader(file, delimiter="\t", quotechar=None))[1:]

    texts = []
    labels = []
    for line in lines:
        texts.append(line[0])
        labels.append(int(line[1]))

    return texts, labels

def load_seq_data_from_json(path, MAX_LEN):
    with open(path, "r") as f:
        data = json.load(f)
    
    texts = []
    labels = []
    for item in data:
        text = item['title'] + " " + item['text']
        text = " ".join(text.split()[:MAX_LEN])
        texts.append(text)
        labels.append(0)
    
    return texts, labels


def load_and_cache_dataset(DATA_PATH, BERT_MODEL='bert-base-uncased', MAX_LEN=256, num_labels=12):
    # tokenization
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    # load ner data
    train_ner_texts, train_ner_tags = read_wnut(os.path.join(DATA_PATH, 'ner_train.txt'))
    test_ner_texts, test_ner_tags = read_wnut(os.path.join(DATA_PATH, 'ner_dev.txt'))

    tags = deepcopy(train_ner_tags)
    tags.extend(test_ner_tags)

    unique_tags = list(set(tag for doc in train_ner_tags for tag in doc))
    tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
    print(tag2id)
    id2tag = {id: tag for tag, id in tag2id.items()}

    # get sequence label from ner label
    train_seq_labels = []
    test_seq_labels = []

    for train_tag in train_ner_tags:
        tag_set = set(train_tag)
        current_label = np.zeros([num_labels])
        if len(tag_set) == 1:
            current_label[tag2id['O']] = 1
        else:
            tag_set.remove('O')
            for tag in tag_set:
                current_label[tag2id[tag]] = 1
        train_seq_labels.append(list(current_label))

    for test_tag in test_ner_tags:
        tag_set = set(test_tag)
        current_label = np.zeros([num_labels])
        if len(tag_set) == 1:
            current_label[tag2id['O']] = 1
        else:
            tag_set.remove('O')
            for tag in tag_set:
                current_label[tag2id[tag]] = 1
        test_seq_labels.append(list(current_label))

    train_encodings = tokenizer(train_ner_texts, is_pretokenized=True, return_offsets_mapping=True, padding=True,
                                truncation=True, max_length=MAX_LEN)
    test_encodings = tokenizer(test_ner_texts, is_pretokenized=True, return_offsets_mapping=True, padding=True,
                               truncation=True, max_length=MAX_LEN)

    train_ner_labels = encode_tags(train_ner_tags, train_encodings, tag2id)
    test_ner_labels = encode_tags(test_ner_tags, test_encodings, tag2id)

    train_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")


    data_to_save = (
    train_encodings, train_seq_labels, train_ner_labels, test_encodings, test_seq_labels, test_ner_labels)
    with open('data/cached_train_test_{}'.format(MAX_LEN), 'wb') as f:
        pickle.dump(data_to_save, f)


def load_and_cache_predict_dataset(DATA_PATH, BERT_MODEL, MAX_LEN=256):
    # tokenization
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)

    if DATA_PATH.endswith(".tsv"):
        predict_text, predict_seq_label = load_seq_data_from_tsv(DATA_PATH)
    elif DATA_PATH.endswith(".json"):
        predict_text, predict_seq_label = load_seq_data_from_json(DATA_PATH, MAX_LEN)
    else:
        raise ValueError("Evaluation data should either be a tsv file or a json file")

    predict_encodings = tokenizer(predict_text, padding=True, truncation=True, max_length=MAX_LEN)

    predict_ner_label = np.zeros([len(predict_text), MAX_LEN])

    data_to_save = (predict_encodings, predict_seq_label, predict_ner_label)
    with open('data/cached_predict_{}_{}'.format(DATA_PATH.split('/')[-1].replace(".","_"), MAX_LEN), 'wb') as f:
        pickle.dump(data_to_save, f)