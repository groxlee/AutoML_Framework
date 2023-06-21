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
from fuel.BaseTorch import BaseTorch

class TuneTorch(BaseTorch):
    def __init__(self, model, config, checkpoint_settings):
        # super().__init__(model, config, checkpoint_settings)
        self.config = config
        self.load_model(model)
        self.checkpoint_settings = checkpoint_settings
    
    ### need to modify
    def set_hyperparameter(self):
        self.tune_config = {
            "lr": tune.loguniform(1e-4, 1e-2), 
            "momentum": tune.uniform(0.5, 0.9),
        }
    
    def tune_train(self, config):
        self.load_data()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"]
        )
        self.epochs = self.config["hyperparameters"]["num_epoch"]
        for epoch in range(1, self.epochs + 1):
            self.train(epoch, optimizer, criterion)
            acc = self.eval(criterion)
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
                name=self.checkpoint_settings.run_id,
                local_dir=self.checkpoint_settings.results_dir,
                stop = {"mean_accuracy": 0.8},
            ),
        )
        self.results = tuner.fit()
        print("------------- Ray tune end -------------")
        print("Best config is:", self.results.get_best_result().config)