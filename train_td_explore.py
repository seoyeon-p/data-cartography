import argparse
import csv
import os
import sys
import json

import pickle
import random
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW, AutoModel, AutoTokenizer,get_linear_schedule_with_warmup
from torch.distributions.distribution import Distribution
from tqdm import tqdm
from torch.distributions import Categorical

from itertools import cycle
from loss import *
import random
random.seed(0)

csv.field_size_limit(sys.maxsize)
#torch.manual_seed(0)

#os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
parser.add_argument('--task', type=str, help='task name (SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG)')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--aug_train_path',type=str)
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--label_smoothing', type=float, default=-1., help='label smoothing \\alpha')
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--warmup_steps',type=int, default=0)
parser.add_argument('--gradient_accumulation_steps',default=1)
parser.add_argument('--adv_mixup',action='store_true')
parser.add_argument('--labeled_train_path', type=str, help='labeled train dataset path for consistency learning')
parser.add_argument('--unlabeled_train_path', type=str, help='unlabeled train dataset path for consistency learning')
parser.add_argument('--consistency_learning', action='store_true')
parser.add_argument('--partial_supervised',action='store_true')
parser.add_argument('--mutual_exclusive',action='store_true')
parser.add_argument('--augment_unlabeled', action='store_true')
parser.add_argument('--noisy_label',action='store_true')
parser.add_argument('--mixup',action='store_true')
parser.add_argument('--fixmatch',action='store_true')
parser.add_argument('--training_dynamics',action='store_true')
parser.add_argument('--cl',action='store_true')
args = parser.parse_args()
print(args)


assert args.task in ('SNLI', 'MNLI', 'QQP','imdb','TwitterPPDB', 'SWAG', 'HellaSWAG','FEVER', 'SST','CoLA','TREC')
assert args.model in ('bert-base-uncased', 'roberta-base', 'bert-large-uncased')


if args.task in ('SNLI', 'MNLI','FEVER'):
    n_classes = 3
elif args.task in ('QQP', 'TwitterPPDB','CoLA'):
    n_classes = 2
elif args.task in ('SWAG', 'HellaSWAG'):
    n_classes = 1
elif args.task in ('SST','imdb'):
    n_classes = 5
elif args.task in ('TREC'):
    n_classes = 6


classes = []
class TRECProcessor:
    def __init__(self):
        self.label_map = [0,1,2,3,4,5]
    def load_samples(self,path):
        samples = []
        file = open(path,"r")
        data = file.read().strip().split("\n")
        desc = f'loading \'{path}\''
        for row in tqdm(data,desc=desc):
            row = row.split("\t")
            label = int(row[1])
            sentence = row[2]
            guid = int(row[0])
            samples.append((sentence,label,guid))
            classes.append(label)
        samples = random.sample(samples,len(samples))
        return samples


class imdbProcessor:
    def __init__(self):
        self.label_map = [0,1,2,3,4]
    def load_samples(self,path):
        samples = []
        file = open(path,"r")
        data = file.read().strip().split("\n")
        desc = f'loading \'{path}\''
        for row in tqdm(data,desc=desc):
            row = row.split("\t")
            label = int(row[1])
            sentence = row[2]
            guid = int(row[0])
            samples.append((sentence,label,guid))
        return samples


def cuda(tensor):
    """Places tensor on CUDA device."""

    return tensor.to(args.device)


def load(dataset, batch_size, shuffle):
    """Creates data loader with dataset and iterator options."""

    return DataLoader(dataset, batch_size, shuffle=shuffle)


def adamw_params(model):
    """Prepares pre-trained model parameters for AdamW optimizer."""

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return params


def encode_sentence(sentence):
    """
    Encodes pair inputs for pre-trained models using the template
    [CLS] sentence1 [SEP] sentence2 [SEP]. Used for SNLI, MNLI, QQP, and TwitterPPDB.
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = tokenizer.encode(
        sentence, max_length=args.max_seq_length
    )
    input_ids = inputs
    #if args.model == 'bert-base-uncased' or args.model == 'bert-large-uncased':
    #    segment_ids = inputs['token_type_ids']
    #else:
    segment_ids = [0]*len(inputs)
    attention_mask = [1] * len(inputs) #inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length
    return (
        cuda(torch.tensor(input_ids).long()),
        cuda(torch.tensor(segment_ids).long()),
        cuda(torch.tensor(attention_mask).long()),
    )



def encode_pair_inputs(sentence1, sentence2):
    """
    Encodes pair inputs for pre-trained models using the template
    [CLS] sentence1 [SEP] sentence2 [SEP]. Used for SNLI, MNLI, QQP, and TwitterPPDB.
    Returns input_ids, segment_ids, and attention_mask.
    """

    inputs = tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=args.max_seq_length
    )
    input_ids = inputs['input_ids']
    if args.model == 'bert-base-uncased' or args.model == 'bert-large-uncased':
        segment_ids = inputs['token_type_ids']
    else:
        segment_ids = [0]*len(inputs['input_ids'])
    attention_mask = inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length
    for input_elem in (input_ids, segment_ids, attention_mask):
        assert len(input_elem) == args.max_seq_length
    return (
        cuda(torch.tensor(input_ids).long()),#.cuda(),
        cuda(torch.tensor(segment_ids).long()),#.cuda(),
        cuda(torch.tensor(attention_mask).long()),#.cuda(),
    )


def encode_mc_inputs(context, start_ending, endings):
    """
    Encodes multiple choice inputs for pre-trained models using the template
    [CLS] context [SEP] ending_i [SEP] where 0 <= i < len(endings). Used for
    SWAG and HellaSWAG. Returns input_ids, segment_ids, and attention_masks.
    """

    #context_tokens = tokenizer.tokenize(context)
    #start_ending_tokens = tokenizer.tokenize(start_ending)
    all_input_ids = []
    all_segment_ids = []
    all_attention_masks = []
    for ending in endings:
        #ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
        inputs = tokenizer.encode_plus(
            context, start_ending+" " + ending, add_special_tokens=True, max_length=args.max_seq_length
        )
        input_ids = inputs['input_ids']
        if args.model == 'bert-base-uncased' or args.model == 'bert-large-uncased':
            segment_ids = inputs['token_type_ids']
        else:
            segment_ids = [0] * len(inputs['input_ids'])
        attention_mask = inputs['attention_mask']
        padding_length = args.max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        segment_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        for input_elem in (input_ids, segment_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        all_input_ids.append(input_ids)
        all_segment_ids.append(segment_ids)
        all_attention_masks.append(attention_mask)
    return (
        cuda(torch.tensor(all_input_ids).long()), #.cuda(),
        cuda(torch.tensor(all_segment_ids).long()), #.cuda(),
        cuda(torch.tensor(all_attention_masks).long()), #.cuda(),
    )


def encode_label(label):
    """Wraps label in tensor."""

    return cuda(torch.tensor(label).long())#.cuda()

class CoLAProcessor:
    def __init__(self):
        self.label_map = [0,1]
    def load_samples(self,path):
        samples = []
        file = open(path,"r")
        data = file.read().strip().split("\n")
        desc = f'loading \'{path}\''
        classes = []
        for row in tqdm(data,desc=desc):
            row = row.split("\t")
            label = int(row[1])
            sentence = row[2]
            guid = int(row[0])
            classes.append(label)
            samples.append((sentence,label,guid))
        return samples

class SSTProcessor:
    def __init__(self):
        self.label_map = [0,1,2,3,4]
    def load_samples(self,path):
        samples = []
        file = open(path,"r")
        data = file.read().strip().split("\n")
        desc = f'loading \'{path}\''
        for row in tqdm(data,desc=desc):
            row = row.split("\t")
            label = int(row[1])
            sentence = row[2]
            guid = int(row[0])
            samples.append((sentence,label,guid))
        return samples

class SNLIProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[1]
                    sentence1 = row[4]
                    sentence2 = row[7]
                    label = row[2]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        
        return samples


class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path):
        samples = []
        ids = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = int(row[0])
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = row[-1]
                    ids.append(guid)
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass
        return samples


class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path):
        samples = []
        aug_samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                #print(row)
                try:
                    guid = int(row[0])
                    sentence1 = row[3]
                    sentence2 = row[4]
                    label = row[5]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass

        return samples


class PAWS_QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path):
        samples = []
        aug_samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                #print(row)
                try:
                    guid = row[0]
                    sentence1 = row[1]
                    sentence2 = row[2]
                    label = row[3]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label, guid))
                except:
                    pass

        return samples


class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3 
    
    def load_samples(self, path):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples.append((sentence1, sentence2, label, []))
                except:
                    pass
        return samples




class FEVERProcessor:
    """Data loader for QQP."""

    def __init__(self):
        self.label_map = {"REFUTES":0, "SUPPORTS":1, "NOT ENOUGH INFO":2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path):
        samples = []
        # guid_check = []
        # label_check = []
        #new_file = open("./data/FEVER/sym_test_v1.txt","w")
        #new_file = open("./data/FEVER/train.jsonl","w")
        #new_data = []
        df = pd.read_json(path, lines=True)
        for i, (_, line) in enumerate(df.iterrows()):
            #line['id'] = line['id']#int(i)
            #new_data.append(dict(line))
            guid = line['id']#i#int(line['id'])
            sentence1 = line['evidence']
            sentence2 = line['claim']
            try:
                label = line['label']
            except:
                label = line['gold_label']
            label = self.label_map[label]
            label = int(label)
            # label_check.append(label)
            # guid_check.append(guid)
            samples.append((sentence1, sentence2, label, guid))
        # print(len(list(set(guid_check))))
        # print(len(guid_check))
        # print(len(list(set(label_check))))
        #new_data = json()
        #new_data = json.dumps(new_data)
        # with open("./data/FEVER/train.jsonl","w") as f:
        #     for entry in new_data:
        #         print(type(entry))
        #         json.dump(entry,f)
        #         f.write("\n")
        return samples

class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path):
        samples = []
        guids = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = int(row[0])
                    guids.append(guid)
                    context = row[4]
                    start_ending = row[5]
                    endings = row[7:11]
                    label = int(row[-1])
                    samples.append((context, start_ending, endings, label, guid))
                except:
                    pass
        return samples


class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path):
        samples = []
        with open(path) as f:
            desc = f'loading \'{path}\''
            for line in f:
                try:
                    line = line.rstrip()
                    input_dict = json.loads(line)
                    context = input_dict['ctx_a']
                    start_ending = input_dict['ctx_b']
                    endings = input_dict['endings']
                    label = input_dict['label']
                    samples.append((context, start_ending, endings, label, []))
                except:
                    pass
        return samples



def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor']()


def select_aug_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}AugProcessor']()

class TextDataset(Dataset):
    """
    Task-specific dataset wrapper. Used for storing, retrieving, encoding,
    caching, and batching samples.
    """

    def __init__(self, path, processor, num_instances=None, augment=False,train=True):
        self.samples = processor.load_samples(path)
        self.unlabeled = False
        self.cache = {}
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]
            if args.task in ('SNLI', 'MNLI', 'QQP', 'MRPC', 'TwitterPPDB','PAWS_QQP','FEVER'):
                sentence1, sentence2, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_pair_inputs(
                    sentence1, sentence2
                )
                packed_inputs = (sentence1, sentence2)
            elif args.task in ('SWAG', 'HellaSWAG'):
                context, ending_start, endings, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_mc_inputs(
                    context, ending_start, endings
                )
            elif args.task in ('SST','CoLA','TREC'):
                sentence, label, guid = sample
                input_ids, segment_ids, attention_mask = encode_sentence(sentence)
            label_id = encode_label(label)
            res = ((input_ids, segment_ids, attention_mask), label_id, guid)
            self.cache[i] = res
        return res



class Model(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model)
        self.classifier = nn.Linear(768, n_classes)
        if args.task in ('SWAG','HellaSWAG'):
            self.n_choices = -1

    def forward(self, input_ids, segment_ids, attention_mask):
        # On SWAG and HellaSWAG, collapse the batch size and
        # choice size dimension to process everything at once
        if args.task in ('SWAG', 'HellaSWAG'):
            n_choices = 4#input_ids.size(1)
            self.n_choices = n_choices
            input_ids = input_ids.view(-1, input_ids.size(-1))
            segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        transformer_params = {
            'input_ids': input_ids,
            'token_type_ids': (
                segment_ids if args.model == 'bert-base-uncased' else None
            ),
            'attention_mask': attention_mask,
        }
        transformer_outputs = self.model(**transformer_params)
        if args.task in ('SWAG', 'HellaSWAG'):
            pooled_output = transformer_outputs[1]
            logits = self.classifier(pooled_output)
            logits = logits.view(-1, n_choices)
            return logits, pooled_output
        else:
            cls_output = transformer_outputs[0][:, 0]
            logits = self.classifier(cls_output)
            return logits, transformer_outputs[0]

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss. Adapted from https://bit.ly/2T6kfz7. If 0 < smoothing < 1,
    this smoothes the standard cross-entropy loss.
    """

    def __init__(self, smoothing):
        super().__init__()
        _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
        self.confidence = 1. - smoothing
        smoothing_value = smoothing / (_n_classes - 1)
        one_hot = cuda(torch.full((_n_classes,), smoothing_value))
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, output, target):
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(F.log_softmax(output, 1), model_prob, reduction='sum')


def smoothing_label(target, label, smoothing):
    _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
    confidence = 1. - smoothing
    smoothing_value = smoothing / (_n_classes - 1)
    one_hot = cuda(torch.full((_n_classes,), smoothing_value))
    model_prob = one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), confidence)
    return model_prob



if args.task in ('SNLI','MNLI','FEVER'):
    temperature = 0.05
elif args.task in ('QQP','TwitterPPDB'):
    temperature = 0.1
elif args.task in ('SWAG','HellaSWAG','TREC'):
    temperature = 0.01
def train(d1,d2=None,aug=None,epoch=0):
    """Fine-tunes pre-trained model on training set."""

    model.train()
    train_loader = tqdm(load(d1, args.batch_size, True))
    train_loss = 0.
    optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
    if args.training_dynamics:
        epoch_file_name = open("./feature_outputs/training_dynamics/"+args.task+"_dynamics_epoch_"+str(epoch)+".jsonl","w")
    for i, dataset in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels, guids = dataset
        #guid = guids.tolist()
        #loss = criterion(model(*inputs), label)
        logits, hidden1 = model(inputs[0],inputs[1],inputs[2])
        #print(aug_inputs)
        #print(guid)
        if args.task in ('SWAG'):
            hidden1 = hidden1.view(logits.shape[0],4,-1)
        # hidden1 = torch.mean(hidden1,dim=1)
        if args.training_dynamics:
            td_df = pd.DataFrame({"guid": guids.tolist(), f"logits_epoch_{epoch}": logits.tolist(),"gold": labels.tolist()})
            td_df.to_json(epoch_file_name, lines=True, orient="records")
        
        loss = criterion(logits,labels)
        
        train_loss += loss.item()
        train_loader.set_description(f'train loss = {(train_loss / (i+1)):.6f}')
        loss.backward()
        if args.max_grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    return train_loss / len(train_loader)


def evaluate(dataset):
    """Evaluates pre-trained model on development set."""

    model.eval()
    eval_loss = 0.
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    for i, dataset in enumerate(eval_loader):
        with torch.no_grad():
            inputs, labels, _  = dataset
            #loss = criterion(model(*inputs), label)
            logits, outputs = model(inputs[0],inputs[1],inputs[2])
            
            # if args.task in ('SNLI','QQP'):
            #     output = model.classifier(output[0][:, 0])
            # elif args.task in ('SWAG'):
            #     output = model.classifier(output[1])
            #     output = output.view(-1,model.n_choices)
            loss = criterion(logits,labels)
        eval_loss += loss.item()
        eval_loader.set_description(f'eval loss = {(eval_loss / (i+1)):.6f}')
    return eval_loss / len(eval_loader)


model = cuda(Model())
#model = Model().cuda()
#model = nn.DataParallel(model)
processor = select_processor()
tokenizer = AutoTokenizer.from_pretrained(args.model)

if args.label_smoothing == -1:
    criterion = nn.CrossEntropyLoss()
    #criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
else:
    criterion = LabelSmoothingLoss(args.label_smoothing)

# if args.train_path:
#     train_dataset = TextDataset(args.train_path, processor)
#     print(f'train samples = {len(train_dataset)}')

if args.task in ('SNLI','QQP','SWAG','MNLI','FEVER','SST','CoLA','TREC'):
    train_dataset = TextDataset(args.train_path,processor)
    print(f'train samples = {len(train_dataset)}')
    #print(f'Augmented train samples = {len(aug_dataset)}')
if args.dev_path:
    dev_dataset = TextDataset(args.dev_path, processor,train=False)
    print(f'dev samples = {len(dev_dataset)}')
if args.test_path:
    test_dataset = TextDataset(args.test_path, processor,train=False)
    print(f'test samples = {len(test_dataset)}')


if args.do_train:
    print()
    print('*** training ***')
    best_loss = float('inf')
    for epoch in range(0, args.epochs):
        #train_loss = train(d1=train_dataset, epoch=epoch)
        train_loss = train(d1=train_dataset,epoch=epoch)
        #train_loss = (train_loss + aug_train_loss)/2
        #train_loss = aug_train_loss
        eval_loss = evaluate(dev_dataset)
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), args.ckpt_path)
        print(
            f'epoch = {epoch} | '
            f'train loss = {train_loss:.6f} | '
            f'eval loss = {eval_loss:.6f}'
        )

if args.do_evaluate:
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError(f'\'{args.ckpt_path}\' does not exist')
    
    print()
    print('*** evaluating ***')

    output_dicts = []
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    test_loader = tqdm(load(test_dataset, args.batch_size, False))

    for i, data in enumerate(test_loader):
        with torch.no_grad():
            inputs, label, _ = data
            logits, outputs = model(inputs[0],inputs[1],inputs[2])
            for j in range(logits.size(0)):
                if args.task in ('imdb'):
                    pos,neg = 0, 0
                    for k in range(0,len(logits[j])):

                        if k == 0 or k==1 :
                            neg += logits[j][k].data.tolist()
                        else:
                            pos += logits[j][k].data.tolist()
                    new_logits = torch.tensor([neg/2,pos/3])
                else:
                    new_logits = logits[j]
            for j in range(logits.size(0)):
                probs = F.softmax(new_logits, -1)
                output_dict = {
                    'index': args.batch_size * i + j,
                    'true': label[j].item(),
                    'pred': new_logits.argmax().item(),
                    'conf': probs.max().item(),
                    'logits': new_logits.cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)

    print(f'writing outputs to \'{args.output_path}\'')

    with open(args.output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')

    y_true = [output_dict['true'] for output_dict in output_dicts]
    y_pred = [output_dict['pred'] for output_dict in output_dicts]
    y_conf = [output_dict['conf'] for output_dict in output_dicts]

    accuracy = accuracy_score(y_true, y_pred) * 100.
    f1 = f1_score(y_true, y_pred, average='macro') * 100.
    confidence = np.mean(y_conf) * 100.

    results_dict = {
        'accuracy': accuracy_score(y_true, y_pred) * 100.,
        'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
        'confidence': np.mean(y_conf) * 100.,
    }
    for k, v in results_dict.items():
        print(f'{k} = {v}')
