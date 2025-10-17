def accuracy(pred, label):
    return (pred.argmax(dim=1) == label).float().mean().item()
