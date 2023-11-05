import pandas as pd
from helpers.analyze_df import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import normalize
from math import floor
import numpy as np

def get_paired_data():
    input_path = r"C:\Users\hamza\Desktop\AlarmPrediction\process_inputs.npy"
    target_path = r"C:\Users\hamza\Desktop\AlarmPrediction\quality_targets.npy"

    f = open(input_path, "rb")
    inputs = np.load(f)

    f = open(target_path, "rb")
    targets = np.load(f)

    paired_data = [[inputs[i], targets[i]] for i in range(len(targets))]
    return paired_data

paired_data = get_paired_data()

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 500)  # Adjust based on your input data
        self.layer2 = nn.Linear(500, 500)  # Adjust based on the type of problem
        self.layer3 = nn.Linear(500, 50)
        self.layer4 = nn.Linear(50, output_dim)

    def forward(self, x):
        x = normalize(x)
        x = x.flatten().unsqueeze(0)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        return x

# Instantiate the model
input_dim = paired_data[0][0].shape[0] * paired_data[0][0].shape[1]  # flatten all input
output_dim = 1
model = NeuralNetwork(input_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjust the learning rate as needed

# Train the model
epochs = 10  # Adjust the number of epochs as needed
total_num_of_data = len(paired_data)
num_of_train_data = floor(total_num_of_data * 0.9)
train_paired_data = paired_data[:num_of_train_data]
test_paired_data = paired_data[num_of_train_data:]
print(f"train pairs len: {len(train_paired_data)}")
print(f"test pairs len: {len(test_paired_data)}")

test_loss_lst = []
for epoch in range(epochs):
    total_test_loss = 0
    for idx, (input, output) in enumerate(train_paired_data):
        model.train()
        X = torch.from_numpy(input).to(torch.float32)
        assert X.isnan().unique().item() == False, "The input contains nan values"
        y = torch.from_numpy(output).to(torch.float32)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(0)) # both is float quality values

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 200 == 0:
            print(f'Train: Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, target: {y.item()}, pred: {outputs.item()}')
    
    for idx, (input, output) in enumerate(test_paired_data):
        X = torch.from_numpy(input).to(torch.float32)
        assert X.isnan().unique().item() == False, "The input contains nan values"
        y = torch.from_numpy(output).to(torch.float32)

        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y.unsqueeze(0)) # both is float quality values
            total_test_loss += loss

        if idx % 5 == 0:
            print(f'Test: Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, target: {y.item()}, pred: {outputs.item()}')

    test_loss_lst.append(total_test_loss)
    print(f"total test loss lst: {test_loss_lst} ")
    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), 'model_weights.pth')

breakpoint()
