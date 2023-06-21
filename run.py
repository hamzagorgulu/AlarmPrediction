import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from helpers import *
import pickle
import matplotlib.pyplot as plt

# read alarm text
with open("datasets/tupras_alarm.txt" ,"r") as f:
    alarms = f.read()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

# hyperparameters
BATCH_SIZE = 64 
BLOCK_SIZE = 256 
EMBEDDING_SIZE = 128
N_HEAD = 8
N_LAYER = 4
DROPOUT = 0.3
LR = 3e-4
EVAL_ITERS = 200
EVAL_INTERVAL = 500
MAX_ITERS = 500

torch.manual_seed(42)

# all alarms occurred in dataset
unique_alarms = list(set(alarms))
VOCAB_SIZE = len(unique_alarms)

# dicts for converting 
str_to_int = { alarm:i for i,alarm in enumerate(unique_alarms) }
int_to_str = { i:alarm for i,alarm in enumerate(unique_alarms) }

# encoder and decoder
encoder = lambda string: [str_to_int[alarm] for alarm in string]
decoder = lambda integer: ''.join([int_to_str[i] for i in integer])

# Train and test splits
encoded_data = torch.tensor(encoder(alarms), dtype=torch.long)
split = int(0.8*len(encoded_data)) 
split2 = int(0.9*len(encoded_data)) 
train_data = encoded_data[:split]
val_data = encoded_data[split:split2]
test_data = encoded_data[split2:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def test():
    # Load the model and records
    model = DecoderOnly().to(device)
    with open("records", "rb") as f:
        records = pickle.load(f)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations
    with torch.no_grad():
        # Calculate the loss for the test data
        losses = torch.zeros(EVAL_ITERS)
        accuracies = torch.zeros(EVAL_ITERS)
        precisions = torch.zeros(EVAL_ITERS)
        recalls = torch.zeros(EVAL_ITERS)
        f1_scores = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch("test")
            logits, loss, accuracy, precision, recall, f1_score = model(X, Y)
            losses[k] = loss.item()
            accuracies[k] = accuracy.item()
            precisions[k] = precision.item()
            recalls[k] = recall.item()
            f1_scores[k] = f1_score.item()

        test_loss = losses.mean()
        test_accuracy = accuracies.mean()
        test_precision = precisions.mean()
        test_recall = recalls.mean()
        test_f1 = f1_scores.mean()

    # Print the test results
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")
    

def plot_results(records):
    train_losses = [record['train'] for record in records]
    val_losses = [record['val'] for record in records]
    iterations = range(0, len(records) * EVAL_INTERVAL, EVAL_INTERVAL)

    # Plotting the loss curves
    plt.plot(iterations, train_losses, label='Train Loss')
    plt.plot(iterations, val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


@torch.no_grad()
def calculate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS) 
        accuracies = torch.zeros(EVAL_ITERS)
        precisions = torch.zeros(EVAL_ITERS)
        recalls = torch.zeros(EVAL_ITERS)
        f1_scores = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss, accuracy, precision, recall, f1_score = model(X, Y)
            losses[k] = loss.item()
            accuracies[k] = accuracy.item()
            precisions[k] = precision.item()
            recalls[k] = recall.item()
            f1_scores[k] = f1_score.item()
        out[split] = losses.mean()
        out[split+"_acc"] = accuracies.mean()
        out[split+"_prec"] = precisions.mean()
        out[split+"_recall"] = recalls.mean()
        out[split+"_f1"] = f1_scores.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = torch.matmul(wei, v)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, EMBEDDING_SIZE):
        super().__init__()
        self.linear1 = nn.Linear(EMBEDDING_SIZE, 4 * EMBEDDING_SIZE)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x

class Block(nn.Module):

    def __init__(self, EMBEDDING_SIZE, N_HEAD):
        super().__init__()
        head_size = EMBEDDING_SIZE // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)
        self.ffwd = FeedForward(EMBEDDING_SIZE)
        self.ln1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.ln2 = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderOnly(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[Block(EMBEDDING_SIZE, N_HEAD=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(EMBEDDING_SIZE) # final layer norm
        self.lm_head = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            accuracy = accuracy_score(torch.argmax(logits, dim=1).cpu(), targets.cpu())
            precision = precision_score(torch.argmax(logits, dim=1).cpu(), targets.cpu(), average = "macro", labels=np.unique(targets.cpu()), zero_division=0.0)
            recall = recall_score(torch.argmax(logits, dim=1).cpu(), targets.cpu(), average = "macro", labels=np.unique(targets.cpu()), zero_division=0.0)
            f1 = f1_score(torch.argmax(logits, dim=1).cpu(), targets.cpu(), average = "macro", labels=np.unique(targets.cpu()), zero_division=0.0)

        return logits, loss, accuracy, precision, recall, f1

model = DecoderOnly().to(device)

print("Training starts now...")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

records = []
for iter in range(MAX_ITERS):

    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        out = calculate_loss()
        print(f"step {iter}: train loss {out['train']:.4f}, train_acc {out['train_acc']:.4f}, train_prec. {out['train_prec']:.4f}, train_recall {out['train_recall']:.4f}, train_f1 {out['train_f1']:.4f}, val loss {out['val']:.4f} ,  val acc {out['val_acc']:.4f}, val prec. {out['val_prec']:.4f}, val recall {out['val_recall']:.4f}, val f1 {out['val_f1']:.4f}")
        records.append(out)

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss, accuracy, precision, recall, f1 = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

with open("records","wb") as f:
    pickle.dump(records, f)
    
# get the results on test set
test()

# Load the records
with open("records", "rb") as f:
    records = pickle.load(f)

# Plot the results
plot_results(records)
