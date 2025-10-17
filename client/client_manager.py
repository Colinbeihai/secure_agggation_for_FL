from client.client import Client
import random

class ClientManager:
    def __init__(self, config, logger, device):
        self.clients = [Client(i, config, logger, device) for i in range(config["num_clients"])]
        self.logger = logger

    def distribute_model(self, global_weights):
        for c in self.clients:
            c.set_weights(global_weights)

    def train_all(self, drop_prob=0.1):
        updates = []
        for client in self.clients:
            if random.random() > drop_prob:
                self.logger.log(f"Client {client.id} training locally...")
                update = client.train_local()
                updates.append(update)
            else:
                self.logger.log(f"Client {client.id} dropped out this round.")
        return updates

