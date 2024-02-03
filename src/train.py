import torch
import torch.optim as optim
import torch.nn as nn
from dataset_loader import load_data
from model import create_resnet50

def train_model(data_dir, num_classes, num_epochs=25, batch_size=8, learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    dataloader = load_data(data_dir, batch_size)
    model = create_resnet50(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch : {epoch}")
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print("loss : ", loss.item())
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs} completed.')

    return model

if __name__ == "__main__":
    model = train_model(data_dir='/Users/ajitkumarsingh/Desktop/cattle-breed-classifier-webapp/Cattle_Resized', num_classes=26)
    torch.save(model.state_dict(), 'cattle_breed_classifier.pth')
