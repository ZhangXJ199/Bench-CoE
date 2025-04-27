import pdb
import glob
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_length=512):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.labels[idx]
        # 在这里对每个文本进行tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def encode_text(x):
    encoding = tokenizer.encode_plus(
        x,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    return pd.Series([encoding['input_ids'].flatten(), encoding['attention_mask'].flatten()])


def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    progress = tqdm(data_loader, total=len(data_loader), desc="Training", leave=False)
    for d in progress:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress.set_postfix({'loss': np.mean(losses)})

    return correct_predictions.double() / n_examples, losses


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    correct_predictions = 0
    losses = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            losses.append(loss.item())
            _, preds = torch.max(outputs.logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
    return correct_predictions.double() / n_examples, losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载 mmlu 数据集的 parquet 文件
data_path = '../data/TIGER-Lab/MMLU-Pro/data/'

file_list = glob.glob(data_path + "test*.parquet")
df_list = [pd.read_parquet(file) for file in file_list]
df_train = pd.concat(df_list, ignore_index=True)

file_list = glob.glob(data_path + "validation*.parquet")
df_list = [pd.read_parquet(file) for file in file_list]
df_val = pd.concat(df_list, ignore_index=True)

subject_mapping = {'biology': 0,
                   'business': 1,
                   'chemistry': 2,
                   'computer science': 3,
                   'economics': 4,
                   'engineering': 5,
                   'health': 6,
                   'history': 7,
                   'law': 8,
                   'math': 9,
                   'philosophy': 10,
                   'physics': 11,
                   'psychology': 12,
                   'other': 13
}


df_train['label_num'] = df_train['category'].map(subject_mapping)
df_val['label_num'] = df_val['category'].map(subject_mapping)

print(df_train['label_num'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = TextDataset(text=df_train['question'], labels=df_train['label_num'], tokenizer=tokenizer, max_length=512)
val_dataset = TextDataset(text=df_val['question'], labels=df_val['label_num'], tokenizer=tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True, num_labels=14)

model = model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

train_losses = []
val_losses = []
EPOCHS = 10

best_val_acc = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device,
                                        n_examples=len(train_dataset))
    for loss in train_loss:
        train_losses.append(loss)
    print(f'Train loss {np.mean(train_losses)} Train accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, val_loader, loss_fn, device, n_examples=len(val_dataset))
    for loss in val_loss:
        val_losses.append(loss)
    print(f'Val loss {np.mean(val_loss)} Validation accuracy {val_acc}')

    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained('./bert_subject_mmlu_pro')
        print("Saved improved model with Validation accuracy:", val_acc)

plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('Train loss')
plt.xlabel('Index')
plt.ylabel('loss')
plt.savefig('train_loss.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(val_losses)
plt.title('Val loss')
plt.xlabel('Index')
plt.ylabel('loss')
plt.savefig('val_loss.png')
plt.close()


