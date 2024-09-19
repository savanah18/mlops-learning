# Import Dataset and DataLoader libraries
from dataset import MNISTDataSet
from torch.utils.data import DataLoader
from model import CNN
import torch
from tqdm import tqdm

if __name__ == "__main__":
    train_dataset = MNISTDataSet(root='./data', train=True, download=True)
    test_dataset = MNISTDataSet(root='./data', train=False, download=True)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN()
    model.to(device)
    
    # training hyperparameters
    lr = 1e-3
    epochs = 10

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ready for train
    model.train()

    for epoch in range(epochs):

        for x, y in tqdm(train_data_loader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            # zero the gradients
            optimizer.zero_grad()

            # backpropagate
            loss.backward()

            # update the parameters
            optimizer.step()

            # print the loss
        print(f'Epoch {epoch}, Loss {loss.item()}')