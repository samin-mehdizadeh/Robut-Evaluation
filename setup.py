## Code for loading models and datasets
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset
##Model
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, image_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.feature_size = self._compute_feature_size(image_size)
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _compute_feature_size(self, image_size):

        x = torch.zeros(1, self.conv1.in_channels, image_size, image_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel() 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.feature_size)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to select dataset
def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        input_channels = 1
        image_size = 28
        num_classes = 10

    elif dataset_name == 'emnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
        train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
        input_channels = 1
        image_size = 28
        num_classes = 47  

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        input_channels = 3
        image_size = 32
        num_classes = 10

    elif dataset_name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        input_channels = 3
        image_size = 32
        num_classes = 10

    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_dataset = datasets.ImageNet(root='/content/data', split='val', transform=transform)
        train_dataset = None
        input_channels = 3
        image_size = 224
        num_classes = 1000 

    else:
        raise ValueError("Invalid dataset name. Choose from 'mnist', 'emnist', 'cifar10', 'svhn', or 'imagenet'.")

    return train_dataset, test_dataset, input_channels, image_size, num_classes

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, label