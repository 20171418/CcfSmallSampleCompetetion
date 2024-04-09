import numpy as np
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import  AutoModel,AutoConfig,AutoTokenizer
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional  as F
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
    device='cuda'
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
    with open(file_name) as f:
        for i in f.readlines():
            data.append(json.loads(i))
    return data


test_df = pd.DataFrame(read_jsonfile(CFG.input_path + "/testA.json"))




# ## 2. Build model Input and Dataset
class TrainDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.title = df['title'].values
        self.assignee = df['assignee'].values
        self.abstract = df['abstract'].values

        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token

    def __len__(self):
        return len(self.title)

    def __getitem__(self, item):

        title = self.title[item]

        abstract = self.abstract[item]

        input_text = title + self.sep_token + abstract

        inputs = self.tokenizer(input_text, truncation=True, max_length=500)

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),

        }
def collate_fn(batch):
    # batch是一个列表，其中每个元素都是数据集返回的字典
    # 首先，我们将input_ids、attention_mask和图片分开
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask']for item in batch]



    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # 返回一个新的字典，其中包含了处理后的input_ids、attention_mask和图片
    return {'input_ids': input_ids, 'attention_mask': attention_mask}
tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)

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

def infer_5folds(test_loader, model, device):
    model.to(device)
    model.eval()
    preds = []
    probs = []
    for step, batch in tqdm(enumerate(test_loader)):
        # return {'input_ids': input_ids, 'attention_mask': attention_mask}
        mask = batch['attention_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask)
        logits = F.softmax(output.logits, dim=-1)

        probs.append(logits.detach().cpu().numpy())

    predictions = np.concatenate(probs,axis=0)


    return predictions

def cal_result():
    res = []
    for fold in range(5):
        saved_path = f'./{CFG.OUTPUT_DIR}fold{fold}_best_score.bin'
        model=Custom_Bert_Mean()
        model.load_state_dict(torch.load(saved_path))
        test_dataset = TrainDataset(test_df, tokenizer)
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=CFG.batch_size * 2,
                                     collate_fn=collate_fn,
                                    shuffle=False,
                                    num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        result_1fold = infer_5folds(test_dataloader, model, CFG.device)
        res.append(result_1fold)

    res_1 = np.array(res)
    print(res_1.shape)
    res_2 = np.mean(res_1,axis=0)
    print(res_2.shape)
    res_3 = np.argmax(res_2,axis=-1)
    print(res_3.shape)
    test_df['label'] = res_3

    test = test_df[['id', 'label']]
    test.to_csv('submit_A.csv', index=None)

if __name__ == '__main__':
    cal_result()