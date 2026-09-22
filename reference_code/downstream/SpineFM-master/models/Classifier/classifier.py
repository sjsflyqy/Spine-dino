import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from dataset import *
def main():
    def calculate_accuracy(loader, model):
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # No need to compute gradients during validation
            for images, targets in loader:
                images, labels = torch.Tensor(np.array(images)).to(device), torch.Tensor(np.array([target['labels'] for target in targets])).to(device=device,dtype=torch.uint8)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy
    root = os.path.join(os.getcwd(),'..')
    data_path = os.path.join(root,'data','NHANES2','Vertebrae')

    train_loader = get_data_loader(dataset='ResNet_training',data_path=data_path,mode='Training',b=32)
    val_loader = get_data_loader(dataset='ResNet_training',data_path=data_path,mode='Validation',b=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pre-trained ResNet model
    ResNet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights)

    # Reconfigure the final layer for four class classification
    in_features = ResNet.fc.in_features
    ResNet.fc = torch.nn.Linear(in_features,4)

    ResNet.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ResNet.parameters(), lr=0.001)

    num_epochs = 15
    best_accuracy = 0
    for epoch in range(num_epochs):

        running_loss = 0
        ResNet.train()

        for i, (images, targets) in enumerate(train_loader):
            # Move inputs and labels to the device
            images, labels = torch.Tensor(np.array(images)).to(device), torch.Tensor(np.array([target['labels'] for target in targets])).to(device=device,dtype=torch.uint8)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ResNet(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        
        val_accuracy = calculate_accuracy(val_loader, ResNet)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(ResNet.state_dict(),os.path.join('weights','resnet_4_class_224px.pth'))

    

if __name__ == '__main__':
    main()