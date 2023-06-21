import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import ray
from ray import air, tune
from ray.air import session
from ray.train.torch import TorchCheckpoint
from ray.tune.schedulers import AsyncHyperBandScheduler


class DataLoader:
    def __init__(self, dataName):
        self.dataName = dataName
        self.dataset = self.__checkDataset()
    
    def __checkDataset(self):
        if self.dataName == "cifar10":
            return datasets.CIFAR10
        else:
            assert False, "Dataset Not Found!!!"
            

    def __getTransform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform

    def getSet(self, ifTrain=True):
        transform = self.__getTransform()
        set = self.dataset(
            "data",
            train=ifTrain,
            download=True,
            transform=transform
        )
        return set

    def get(self, type="train", batchSize=1, shuffle=False, numWorker=0):
        set = self.getSet(True if type=="train" else False)
        loader = torch.utils.data.DataLoader(
            set,
            batch_size=batchSize,
            shuffle=shuffle,
            num_workers=numWorker
        )
        return loader

class LossFunction:
    def __init__(self):
        self.__options = {
            "CrossEntropy": nn.CrossEntropyLoss(),
            "L1": nn.L1Loss(),
            "MSE": nn.MSELoss(),
            "NLL": nn.NLLLoss(),
        }

    def get(self, name):
        function = self.__options.get(name, "Loss Function NOT Found!!!")
        return function


# ##########################
# config = {
#     "dataset": "cifar10",
#     "train_batch_size": 4,
#     "test_batch_size": 4,
#     "train_shuffle": True,
#     "test_shuffle": False,
#     "train_num_workers": 2,
#     "test_num_workers": 2,
#     "loss_function": "CrossEntropy",
#     "optim": "SGD",
#     "optim_lr": 0.001,
#     "momentum": 0.9,
#     "epochs": 5,
# }
# ##########################

class BaseBox:
    def __init__(self, model, config):
        self.config = config
        self.load_data()
        self.load_model(model)
        print("ready to start!")


    def load_data(self):
        print("Loading dataset...")
        data_loader = DataLoader(self.config["dataset"])
        self.train_loader = data_loader.get(
            "train",
            batchSize = self.config["train_batch_size"],
            shuffle =   self.config["train_shuffle"],
            numWorker = self.config["train_num_workers"]
        )
        self.test_loader = data_loader.get(
            "test",
            batchSize = self.config["test_batch_size"],
            shuffle =   self.config["test_shuffle"],
            numWorker = self.config["test_num_workers"]
        )
        print("Loading dataset... Done")
    
    def load_model(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        print("Load model Done.")

    def load_loss_function(self, name):
        loss_function = LossFunction()
        self.lossfunction = loss_function.get(name)
        print("Loss Function Load.")
        
    def load_optimizer(self, name, params, lr, momentum=0):
        ### add code here ###
        __options = {
            "SGD": optim.SGD(params, lr=lr, momentum=momentum),
            # "Adam": optim.Adam(params, lr=lr),
            # "AdamW": optim.AdamW(params, lr=lr),
        }
        self.optimizer = __options.get(name, "Optimizer NOT Found!!!")

    def train(self, epoch):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (input, label) in enumerate(self.train_loader):
            input, label = input.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.lossfunction(output, label)
            loss.backward()
            self.optimizer.step()
            # print statistics
            running_loss += loss.item()
            if (batch_idx + 1) % 3000 == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx + 1:5d}]\tLoss: {running_loss / 3000:.3f}'
                )
                running_loss = 0.0
    
    def eval(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for input, label in self.test_loader:
                input, label = input.to(self.device), label.to(self.device)
                output = self.model(input)
                test_loss += self.lossfunction(output, label).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
            
        test_loss /= len(self.test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset))
        )
        # Empty cache after evaluation
        torch.cuda.empty_cache()
        return correct / len(self.test_loader.dataset)
    
    def start(self):
        print("------------- start -------------")
        self.load_loss_function(self.config["loss_function"])
        self.load_optimizer(
            name = self.config["optim"],
            params=self.model.parameters(),
            lr=self.config["optim_lr"],
            momentum=self.config["momentum"]
        )
        self.epochs = self.config["epochs"]
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.eval()


class TuneBox(BaseBox):
    def __init__(self, model, config):
        super().__init__(model, config)
    
    ### need to modify
    def set_hyperparameter(self):
        self.tune_config = {
            "lr": tune.loguniform(1e-4, 1e-2), 
            "momentum": tune.uniform(0.1, 0.9),
        }
    
    def tune_train(self, config):
        self.load_loss_function(self.config["loss_function"])
        self.load_optimizer(
            name = self.config["optim"],
            params=self.model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"]
        )
        self.epochs = self.config["epochs"]
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            acc = self.eval()
            session.report({"mean_accuracy": acc})
    
    def start(self):
        print("------------- Ray tune start -------------")

        ray.init()
        ### need to modify, add more type of sched
        self.sched = AsyncHyperBandScheduler()
        self.set_hyperparameter()
        tuner = tune.Tuner(
            self.tune_train,
            param_space=self.tune_config,
            tune_config=tune.TuneConfig(
                num_samples=10,
                metric="mean_accuracy",
                mode="max",
                scheduler=self.sched,
            ),
            run_config=air.RunConfig(
                name="testbox_cifar10_1",
                local_dir="G:/TestProject/results",
                stop = {"mean_accuracy": 0.8},
            ),
        )
        self.results = tuner.fit()
        print("------------- Ray tune end -------------")
        print("Best config is:", self.results.get_best_result().config)

