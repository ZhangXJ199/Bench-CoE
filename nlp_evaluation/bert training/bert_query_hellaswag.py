import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class TextDataset(Dataset):
    def __init__(self, questions, cor_models, tokenizer, max_length=512):
        self.questions = questions
        self.cor_models = cor_models
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        label = self.cor_models[idx]

        encoding = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    for d in tqdm(data_loader, desc="Training", leave=False):
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(outputs.squeeze(-1), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(outputs.squeeze(-1), labels)
            losses.append(loss.item())
            predictions.append(outputs.max().item())

    return np.mean(losses), predictions

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True, num_labels=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

with open('merged_results_hellaswag.json', 'r') as f:
    dataset = json.load(f)

train_dataset = TextDataset(
    questions=[data["input_sentence"] for data in dataset],
    cor_models=[data["cor_models"] for data in dataset],
    tokenizer=tokenizer,
    max_length=512
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_fn = torch.nn.MSELoss().to(device)

EPOCHS = 20
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    print(f"Train loss: {train_loss}")

model.save_pretrained('./bert_query_hellaswag')
