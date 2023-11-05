import torch
from torch import nn
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.optim as optim
from math import floor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import itertools
from torch.utils.data import DataLoader, TensorDataset
import json


writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is: {device}")

def get_paired_data():
    input_path = "process_inputs.npy"
    target_path = "quality_targets.npy"

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


# Train the model
epochs = 100  # Adjust the number of epochs as needed
batch_size = 50
total_num_of_data = len(paired_data)
num_of_train_data = floor(total_num_of_data * 0.9)
train_paired_data = paired_data[:num_of_train_data]
test_paired_data = paired_data[num_of_train_data:]
print(f"train pairs len: {len(train_paired_data)}")
print(f"test pairs len: {len(test_paired_data)}")

# create pytorch datalaoders
# Convert the paired data to tensors
train_inputs = torch.tensor([pair[0] for pair in train_paired_data]).float()
train_targets = torch.tensor([pair[1] for pair in train_paired_data]).float()
test_inputs = torch.tensor([pair[0] for pair in test_paired_data]).float()
test_targets = torch.tensor([pair[1] for pair in test_paired_data]).float()

# Create TensorDataset
train_dataset = TensorDataset(train_inputs, train_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
breakpoint()
# Define the values for the grid search
num_layers_list = [2, 4, 8, 16]
dim_feedforward_list = [256, 512, 1024]
dropout_list = [0.1, 0.2, 0.3]
learning_rate_list = [1e-3, 1e-4, 1e-5, 1e-6]

test_loss_lst = []
best_test_loss = float('inf')
best_model_weights = None
# Store the results with hyperparameters
results = {}

# Perform grid search
for num_layers, dim_feedforward, dropout, learning_rate in itertools.product(num_layers_list, dim_feedforward_list, dropout_list, learning_rate_list):
    print(f"Running for: num_layers={num_layers}, dim_feedforward={dim_feedforward}, dropout={dropout}, learning_rate={learning_rate}")
    
    # Instantiate the Transformer model
    model = TransformerTimeSeries(input_dim=input_dim, output_dim=1, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout)
    model.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        running_train_loss = 0
        running_test_loss = 0
        for batch_idx, (batch_input, batch_output) in enumerate(train_loader):
            model.train()
            batch_input = batch_input.to(device)
            batch_output = batch_output.to(device)
            #X = torch.from_numpy(input).to(torch.float32).unsqueeze(0).to(device) # mini batch
            #assert X.isnan().unique().item() == False, "The input contains nan values"
            #y = torch.from_numpy(output).to(torch.float32).unsqueeze(0).to(device)

            # Forward pass
            outputs = model(batch_input)
            loss = criterion(outputs, batch_output) # both is float quality values

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training loss to TensorBoard
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_paired_data) + batch_idx)
            running_train_loss += loss.item()
            
        # Log average training loss for the epoch
        writer.add_scalar('Loss/Average_Train', running_train_loss / len(train_paired_data), epoch)
        print(f'Train: Epoch [{epoch+1}/{epochs}], Loss: {running_train_loss / len(train_paired_data):.4f}')

        for batch_idx, (batch_input, batch_output) in enumerate(test_loader):
            batch_input = batch_input.to(device)
            batch_output = batch_output.to(device)
            #X = torch.from_numpy(input).to(torch.float32).unsqueeze(0).to(device)
            #assert X.isnan().unique().item() == False, "The input contains nan values"
            #y = torch.from_numpy(output).to(torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(batch_input)
                loss = criterion(outputs, batch_output) # both is float quality values
                running_test_loss += loss.item()

            # Log test loss for each step
            writer.add_scalar('Loss/Test', loss.item(), epoch * len(test_paired_data) + batch_idx)
        
        # Log average test loss for the epoch
        writer.add_scalar('Loss/Average_Test', running_test_loss / len(test_paired_data), epoch)
        print(f'Test: Epoch [{epoch+1}/{epochs}], Loss: {running_test_loss / len(test_paired_data):.4f}')

        if running_test_loss < best_test_loss:
            best_test_loss = running_test_loss
            best_model_weights = model.state_dict()
            best_hyperparameters = {
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "learning_rate": learning_rate
        }
    # Save the best model's weights based on test loss
    if best_model_weights is not None:
        best_model_path = f'best_model_weights_transformer_num_layers_{best_hyperparameters["num_layers"]}_dim_{best_hyperparameters["dim_feedforward"]}_dropout_{best_hyperparameters["dropout"]}_lr_{best_hyperparameters["learning_rate"]}.pth'
        torch.save(best_model_weights, best_model_path)
        print(f"PyTorch best model's weights are saved at {best_model_path}")

        
        results[(num_layers, dim_feedforward, dropout, learning_rate)] = {
            "hyperparameters": {
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "learning_rate": learning_rate
            },
            "results": {
                "train_loss": running_train_loss / len(train_loader),
                "test_loss": running_test_loss / len(test_loader)
            }
        }

with open('grid_search_results.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)

# Close the writer when you are done
writer.close()
breakpoint()


