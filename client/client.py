import torch
from torch import nn, optim
from models.cnn import CNN
from data.data_loader import get_local_loader

class Client:
    def __init__(self, client_id, config, logger, device):
        self.id = client_id
        self.config = config
        self.logger = logger
        self.device = device
        self.model = CNN().to(self.device)
        self.train_loader = get_local_loader(client_id)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config["learning_rate"])

    def set_weights(self, weights):
        self.model.load_state_dict({k: v.to(self.device) for k,v in weights.items()})

    def train_local(self):
        self.model.train()
        for epoch in range(self.config["local_epochs"]):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
        # 返回 CPU 权重，方便聚合
        return {"weights": {k: v.cpu() for k,v in self.model.state_dict().items()},
                "num_samples": len(self.train_loader.dataset)}
