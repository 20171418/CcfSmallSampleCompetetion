import numpy as np
import pandas as pd
import time
import math
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoModel, AdamW, AutoConfig, AutoTokenizer
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import os
import json

import random


class CFG:
    input_path = './data'
    model_path = './roberta_data'
    scheduler = 'cosine'
    batch_scheduler = True
    num_cycles = 0.5  # 1.5
    num_warmup_steps = 0
    max_input_length = 512
    epochs = 20
    encoder_lr = 20e-6
    decoder_lr = 20e-6
    min_lr = 0.5e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0
    num_fold = 5
    batch_size = 8
    seed = 1006
    OUTPUT_DIR = './model_weight/'
    num_workers = 2
    device = 'cuda'
    print_freq = 100


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(CFG.seed)


## 1. Read Data & EDA
def read_jsonfile(file_name):
    data = []
    with open(file_name, encoding="utf-8") as f:
        for i in f.readlines():
            data.append(json.loads(i))
    return data


train = pd.DataFrame(read_jsonfile(CFG.input_path + "/train.json"))
test = pd.DataFrame(read_jsonfile(CFG.input_path + "/testA.json"))
train['label_id'] = train['label_id'].apply(lambda x: int(x))
train.reset_index(inplace=True)

skf = StratifiedKFold(n_splits=5)
for fold, (_, val_) in enumerate(skf.split(X=train, y=train.label_id, groups=train.label_id)):
    train.loc[val_, "fold"] = int(fold)
t = train.groupby('fold')['label_id'].value_counts()


# ## 2. Build model Input and Dataset
class TrainDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.title = df['title'].values
        self.assignee = df['assignee'].values
        self.abstract = df['abstract'].values
        self.label = df['label_id'].values
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token

    def __len__(self):
        return len(self.title)

    def __getitem__(self, item):
        label = int(self.label[item])
        title = self.title[item]

        abstract = self.abstract[item]

        input_text = title + self.sep_token + abstract

        inputs = self.tokenizer(input_text, truncation=True, max_length=500)

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    # batch是一个列表，其中每个元素都是数据集返回的字典
    # 首先，我们将input_ids、attention_mask和图片分开
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels, dim=0)
    # 返回一个新的字典，其中包含了处理后的input_ids、attention_mask和图片
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
dataset = TrainDataset(train, tokenizer)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class Custom_Bert_Mean(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(CFG.model_path)
        config.output_hidden_states = True
        self.base = AutoModel.from_pretrained(CFG.model_path, config=config)
        dim = config.hidden_size
        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(p=0.2)
        self.cls = nn.Linear(dim, 36)

    def forward(self, input_ids, attention_mask, labels=None):
        base_output = self.base(input_ids=input_ids,
                                attention_mask=attention_mask,
                                )

        output_last = base_output.hidden_states[-1]  # b, s ,h

        # tensor平均
        output = self.pooler(output_last, attention_mask)
        output = self.dropout(output)
        output = self.cls(output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, labels)

        return SequenceClassifierOutput(logits=output, loss=loss)


def get_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def train_fn(train_loader, model, optimizer, epoch, device, scheduler):
    model.train()
    total_loss = 0
    total_samples = 0

    for step, batch in enumerate(train_loader):
        label = batch['labels'].to(device)
        mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        batch_size = label.size(0)
        output = model(input_ids, mask, labels=label)
        loss = output.loss
        total_loss += loss.item()
        total_samples += batch_size
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if step % CFG.print_freq == 0 or step == len(train_loader) - 1:
            print(f'epoch:{epoch}, step:{step}, loss:{loss.item()}')

    return total_loss / total_samples


def valid_fn(valid_loader, model, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    preds = []
    labels = []

    for step, batch in enumerate(valid_loader):
        label = batch['labels'].to(device)
        mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        batch_size = label.size(0)

        with torch.no_grad():
            output = model(input_ids, mask, labels=label)

        loss = output.loss
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        y_preds = output.logits.argmax(dim=-1)
        preds.append(y_preds.to('cpu').numpy())
        labels.append(label.to('cpu').numpy())

    predictions = np.concatenate(preds)
    labels = np.concatenate(labels)

    return total_loss / total_samples, predictions, labels


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
    ]
    return optimizer_parameters


def train_loop(folds, train):
    print('start training...')
    for fold in range(folds):
        print(f'fold {fold} start training...')
        model = Custom_Bert_Mean()
        model.to(CFG.device)

        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=CFG.encoder_lr,
                                                    decoder_lr=CFG.decoder_lr,
                                                    weight_decay=CFG.weight_decay)
        optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

        tr_data = train[train['fold'] != fold].reset_index(drop=True)
        va_data = train[train['fold'] == fold].reset_index(drop=True)
        tr_dataset = TrainDataset(tr_data, tokenizer)
        va_dataset = TrainDataset(va_data, tokenizer)

        best_score = 0
        for epoch in range(CFG.epochs):
            train_loader = DataLoader(tr_dataset, batch_size=CFG.batch_size, shuffle=True, collate_fn=collate_fn)
            valid_loader = DataLoader(va_dataset, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn)

            avg_loss_train = train_fn(train_loader, model, optimizer, epoch, CFG.device, CFG.scheduler)
            avg_loss_valid, predictions, labels = valid_fn(valid_loader, model, CFG.device)
            score = get_score(labels, predictions)
            print(f'epoch {epoch} train loss: {avg_loss_train:.4f}')
            print(f'epoch {epoch} valid loss: {avg_loss_valid:.4f}')
            print(f'epoch {epoch} f1 score: {score:.4f}')

            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), f'./{CFG.OUTPUT_DIR}fold{fold}_best_score.bin')


if __name__ == '__main__':
    train_loop(CFG.num_fold, train)
