import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([transforms.ToTensor()])

from utils.config_parser import load_config
config = load_config("config.yaml")
num_clients = config["num_clients"]

def get_local_loader(client_id):
    dataset = datasets.MNIST(root="./data/datasets", train=True, download=True, transform=transform)
    part_size = len(dataset) // num_clients
    parts = random_split(dataset, [part_size] * num_clients)
    return DataLoader(parts[client_id], batch_size=32, shuffle=True)

def get_test_loader():
    test_dataset = datasets.MNIST(root="./data/datasets", train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=64, shuffle=False)
