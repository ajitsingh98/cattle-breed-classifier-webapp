import torch
from dataset_loader import load_data
from model import create_resnet50

def evaluate_model(model_path, data_dir, num_classes, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloader = load_data(data_dir, batch_size)
    model = create_resnet50(num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

if __name__ == "__main__":
    evaluate_model(model_path='cattle_breed_classifier.pth', data_dir='cattle_image', num_classes=30)
