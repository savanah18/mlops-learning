# Create a training pipeline using MLP model from .model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MLP
from dataset import MNISTDataSet
import einops

# Define the hyperparameters
input_dim = 28*28
hidden_dim = 32
output_dim = 10
lr = 0.01
epochs = 1

# Create a simple 1 layer MLP
model = MLP(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load the dataset
dataset = MNISTDataSet(root='MNIST', download=True, train=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
# integrate gpu device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.train()
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        # Get the inputs and labels and load to device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # flatten the inputs using einops
        inputs = einops.rearrange(inputs, 'b c h w -> b (c h w)')
        # convert labels to one hot encoding and convert to float
        labels = torch.nn.functional.one_hot(labels, num_classes=output_dim).float()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')