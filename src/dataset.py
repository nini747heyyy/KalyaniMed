import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

def get_dataloaders(batch_size=64, size=128):
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Data Augmentations and Normalization
    transform_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = DataClass(split='train', transform=transform_train, download=True, size=size)
    val_dataset = DataClass(split='val', transform=transform_test, download=True, size=size)
    test_dataset = DataClass(split='test', transform=transform_test, download=True, size=size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(info['label'])
    return train_loader, val_loader, test_loader, num_classes