import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PointPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=50, output_size=2):
        super(PointPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

## Training
def main():
    # Hyperparameters
    input_size = 6  # 3 points * 2 coordinates each
    hidden_size = 50
    output_size = 2  # Predicting next (x, y)

    # Initialize the model
    model = PointPredictor(input_size, hidden_size, output_size).to(device=device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_path = os.path.join('..','..','..','data','NHANES2','point_predictor_data')

    X_train = np.load(os.path.join(data_path,'train_input.npy'))
    y_train = np.load(os.path.join(data_path,'train_output.npy'))

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

    X_val = np.load(os.path.join(data_path,'val_input.npy'))
    y_val = np.load(os.path.join(data_path,'val_output.npy'))

    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

    # Training loop (assuming X_train and y_train are prepared as before)
    for epoch in range(10000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.view(-1, input_size))  # Flatten input to match input_size
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(X_val.view(-1, input_size))
                loss = criterion(outputs, y_val)
            print(f'Epoch [{epoch+1}/10000], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(),os.path.join('..','..','weights','point_predictor.pth'))

if __name__ == '__main__':
    main()