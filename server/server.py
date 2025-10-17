import torch
from models.cnn import CNN
from server.aggregator import fed_avg
from data.data_loader import get_test_loader

class Server:
    def __init__(self, config, logger, device):
        self.config = config
        self.logger = logger
        self.device = device
        self.global_model = CNN().to(self.device)  # 放到 GPU
        self.test_loader = get_test_loader()

    def get_global_model(self):
        # 直接返回 CPU tensor 的 state_dict，便于聚合
        return {k: v.cpu() for k,v in self.global_model.state_dict().items()}

    def aggregate(self, updates):
        if not updates:
            return  # 避免空列表报错
        new_weights = fed_avg(updates)
        self.global_model.load_state_dict({k: v.to(self.device) for k,v in new_weights.items()})

    def evaluate(self):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.global_model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total
