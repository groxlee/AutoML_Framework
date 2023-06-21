from testbox import BaseBox, TuneBox
import torch
import torch.nn as nn
import torch.nn.functional as F

import ray
from ray import tune

# network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#########################
config = {
    "dataset": "cifar10",
    "train_batch_size": 4,
    "test_batch_size": 4,
    "train_shuffle": True,
    "test_shuffle": False,
    "train_num_workers": 2,
    "test_num_workers": 2,
    "loss_function": "CrossEntropy",
    "optim": "SGD",
    "optim_lr": 0.001,
    "momentum": 0.9,
    "epochs": 5,
}
def main():
    model = Net()
    testbox = BaseBox(model=model, config=config)
    testbox.start()
    print("----------- main done -------------")
# def main():
#     model = Net()
#     testbox = TuneBox(model=model, config=config)
#     testbox.start()
#     print("----------- main done -------------")


if __name__ == "__main__":
    main()

