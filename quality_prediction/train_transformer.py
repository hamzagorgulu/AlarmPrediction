import torch
from torch import nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.optim as optim
from math import floor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is: {device}")

def get_paired_data():
    input_path = r"C:\Users\hamza\Desktop\AlarmPrediction\process_inputs.npy"
    target_path = r"C:\Users\hamza\Desktop\AlarmPrediction\quality_targets.npy"

    f = open(input_path, "rb")
    inputs = np.load(f)

    f = open(target_path, "rb")
    targets = np.load(f)

    paired_data = [[inputs[i], targets[i]] for i in range(len(targets))]
    print("Input-target pairs are loaded.")
    return paired_data

paired_data = get_paired_data()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            # Even dimension
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            # Odd dimension, extend div_term by 1
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Define the Transformer Model
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, nhead=4, dim_feedforward=512, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = input_dim
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout, max_len=5000)

        self.encoder_layers = TransformerEncoderLayer(self.d_model, nhead=nhead,
                                                      dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, features]
        src += self.pos_encoder(src) # add positional encoding to the src itself
        transformer_output = self.transformer_encoder(src)  # some say the embedding should not be done for time series dataset, needs to be investigated
        output = self.decoder(transformer_output[-1])
        return output

# Instantiate the Transformer model
input_dim = paired_data[0][0].shape[-1]  # Assuming paired_data[0][0] is [seq_len, features]
output_dim = 1
model = TransformerTimeSeries(input_dim, output_dim, nhead = 1, num_layers=4, dim_feedforward=512, dropout=0.1)
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Train the model
epochs = 100  # Adjust the number of epochs as needed
total_num_of_data = len(paired_data)
num_of_train_data = floor(total_num_of_data * 0.9)
train_paired_data = paired_data[:num_of_train_data]
test_paired_data = paired_data[num_of_train_data:]
print(f"train pairs len: {len(train_paired_data)}")
print(f"test pairs len: {len(test_paired_data)}")

test_loss_lst = []
for epoch in range(epochs):
    running_train_loss = 0
    running_test_loss = 0
    for idx, (input, output) in enumerate(tqdm(train_paired_data)):
        model.train()
        X = torch.from_numpy(input).to(torch.float32).unsqueeze(0).to(device) # mini batch
        assert X.isnan().unique().item() == False, "The input contains nan values"
        y = torch.from_numpy(output).to(torch.float32).unsqueeze(0).to(device)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y) # both is float quality values

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training loss to TensorBoard
        writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_paired_data) + idx)
        running_train_loss += loss.item()

        if idx % 200 == 0:
            print(f'Train: Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, target: {y.item()}, pred: {outputs.item()}')
    
    # Log average training loss for the epoch
    writer.add_scalar('Loss/Average_Train', running_train_loss / len(train_paired_data), epoch)

    for idx, (input, output) in enumerate(test_paired_data):
        X = torch.from_numpy(input).to(torch.float32).unsqueeze(0).to(device)
        assert X.isnan().unique().item() == False, "The input contains nan values"
        y = torch.from_numpy(output).to(torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y) # both is float quality values
            running_test_loss += loss.item()

        # Log test loss for each step
        writer.add_scalar('Loss/Test', loss.item(), epoch * len(test_paired_data) + idx)

        if idx % 5 == 0:
            print(f'Test: Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, target: {y.item()}, pred: {outputs.item()}')
    
    # Log average test loss for the epoch
    writer.add_scalar('Loss/Average_Test', running_test_loss / len(test_paired_data), epoch)

    test_loss_lst.append(running_test_loss)
    print(f"test loss for epochs: {test_loss_lst} ")
    if (epoch + 1) % 1 == 0:
        model_path = 'model_weights_transformer.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Pytorch stated dict is saved at {model_path}")

# Close the writer when you are done
writer.close()
breakpoint()


