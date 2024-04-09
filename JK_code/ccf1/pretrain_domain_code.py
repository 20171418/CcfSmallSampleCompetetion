


import json
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, \
     TrainingArguments, AutoModelForMaskedLM
import pandas as pd


def read_jsonfile(file_name):
    data = []
    with open(file_name, encoding='utf-8') as f:
        data = json.loads(f.read(), strict=False)
    return data

pydf = pd.DataFrame(read_jsonfile("./data/发明专利数据.json"))

pydf = pydf[['pat_name', 'pat_applicant', 'pat_summary']]
pydf = pydf.dropna()
pydf['source'] = pydf['pat_name'] +  pydf['pat_applicant'] + pydf['pat_summary']

tokenizer = AutoTokenizer.from_pretrained('./roberta_data')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


class PDataset(Dataset):

    def __init__(self, df, tokenizer):
        super().__init__()
        self.df = df.reset_index(drop=True)
        # maxlen allowed by model config
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        row = self.df.iloc[index]
        doc = row.source
        inputs = {}
        try:
          doc_id = tokenizer(doc, truncation=True, max_length=125)
          doc_id = data_collator([doc_id])
          inputs['input_ids'] = doc_id['input_ids'][0].tolist()
          inputs['labels'] = doc_id['labels'][0].tolist()
       
          if 'token_type_ids' in inputs:
            inputs['token_type_ids'] = [0] * len(inputs['input_ids'])
        except:
          print('*'*20)
          print(doc)
          print('*'*20)
          
        return inputs

    def __len__(self):
        return self.df.shape[0]


mask_id = tokenizer.mask_token_id
def data_collator_p(batch):
    max_length = max([len(i['input_ids']) for i in batch])
    input_id, token_type, labels = [], [], []
    for i in batch:
        input_id.append(i['input_ids'] + [mask_id]*(max_length-len(i['input_ids'])))
        labels.append(i['labels'] + [-100] * (max_length - len(i['labels'])))

    output={}
    output['input_ids'] = torch.as_tensor(input_id, dtype=torch.long)

    output['labels'] = torch.as_tensor(labels, dtype=torch.long)
    return output

training_args = TrainingArguments(
    output_dir='./domain_weight/train',
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=10,
    save_total_limit=1,
    save_strategy='epoch',
    learning_rate=2e-5,
    logging_strategy="steps",
    logging_steps=5,
    report_to="tensorboard",  # 更改此处为 "tensorboard"
    # fp16=True,
    gradient_accumulation_steps=4,
)

dataset = PDataset(pydf, tokenizer)

model = AutoModelForMaskedLM.from_pretrained('./roberta_data')
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator_p,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model('./domain_weight')
