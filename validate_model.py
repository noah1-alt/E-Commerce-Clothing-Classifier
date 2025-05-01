import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define CNN (same as yours)
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.elu(self.conv1(x)))
        x = self.pool(self.elu(self.conv2(x)))
        x = self.pool(self.elu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
model = CNN()
model.load_state_dict(torch.load('model/fashion_mnist_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# Load Fashion MNIST test set
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate
correct = 0
total = 0
bag_count = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        bag_count += (predicted == 8).sum().item()

accuracy = 100 * correct / total
bag_percentage = 100 * bag_count / total
print(f'Test Accuracy: {accuracy:.2f}%')
print(f'Percentage of "Bag" predictions: {bag_percentage:.2f}%')