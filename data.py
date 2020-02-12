from torchvision import datasets, transforms
import os

def get_dataset(path='anime_dataset'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return datasets.ImageFolder(os.path.join('.', path), transform)

