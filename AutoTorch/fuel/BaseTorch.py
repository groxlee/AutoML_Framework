import os
# Solve the key board interrupt exception after joining tensorboard
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

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

# class LossFunction:
#     def __init__(self):
#         self.__options = {
#             "CrossEntropy": nn.CrossEntropyLoss(),
#             "L1": nn.L1Loss(),
#             "MSE": nn.MSELoss(),
#             "NLL": nn.NLLLoss(),
#         }

#     def get(self, name):
#         function = self.__options.get(name, "Loss Function NOT Found!!!")
#         return function


class BaseTorch:
    def __init__(self, model, config, checkpoint_settings):
        self.config = config
        self.checkpoint_settings = checkpoint_settings
        self.load_data()
        self.load_model(model)
        self.writer = SummaryWriter(self.checkpoint_settings.write_path)
        print("ready to start!")


    def load_data(self):
        print("Loading dataset...")
        data_loader = DataLoader("cifar10")
        self.train_loader = data_loader.get(
            "train",
            batchSize = 4,
            shuffle =   True,
            numWorker = 2
        )
        self.test_loader = data_loader.get(
            "test",
            batchSize = 4,
            shuffle =   False,
            numWorker = 2
        )
        print("Loading dataset... Done")
    
    def load_model(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        print("Load model Done.")

    def train(self, epoch, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (input, label) in enumerate(self.train_loader):
            input, label = input.to(self.device), label.to(self.device)
            optimizer.zero_grad()
            #input shape:  [4, 3, 32, 32]
            output = self.model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # Tensorboard
            if self.config["tensorboard"]["loss"]:
                self.writer.add_scalar('Training loss', loss.item(), epoch*len(self.train_loader)+batch_idx)
            if self.config["tensorboard"]["learning_rate"]:
                temp_lr = optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning Rate', temp_lr, epoch*len(self.train_loader)+batch_idx)
            if (batch_idx + 1) % 3000 == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx + 1:5d}]\tLoss: {running_loss / 3000:.3f}'
                )
                running_loss = 0.0
            if (batch_idx + 1) % self.checkpoint_settings.interval == 0:
                run_logs_dir = self.checkpoint_settings.run_logs_dir
                os.makedirs(run_logs_dir, exist_ok=True)
                checkpoint_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "critetion": criterion.state_dict(),
                }
                torch.save(checkpoint_dict, os.path.join(run_logs_dir, f'checkpoint_{batch_idx+1}.pth'))
                print(f'Checkpoint Save... Done')
            

     
    def eval(self, criterion):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for input, label in self.test_loader:
                input, label = input.to(self.device), label.to(self.device)
                output = self.model(input)
                test_loss += criterion(output, label).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
            
        test_loss /= len(self.test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset))
        )
        accuracy = correct / len(self.test_loader.dataset)
        # Tensorboard
        if self.config["tensorboard"]["accuracy"]:
            self.writer.add_scalar('Accuracy', accuracy, self.epochs)
        # Empty cache after evaluation
        torch.cuda.empty_cache()
        return accuracy
    
    def start(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config["hyperparameters"]["learning_rate"],
            momentum=self.config["hyperparameters"]["momentum"]
        )
        self.epochs = self.config["hyperparameters"]["num_epoch"]
        start_epoch = 1
        if self.checkpoint_settings.resume:
            print("resume from checkpoint")
            checkpoint_path = os.path.join(self.checkpoint_settings.write_path, 'checkpoint.pth')
            checkpoint_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint_dict["model"])
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
            criterion.load_state_dict(checkpoint_dict["critetion"])
            start_epoch = checkpoint_dict["epoch"]
            print("Load checkpoint... Done")
        try:
            for epoch in range(start_epoch, self.epochs + 1):
                self.train(epoch, optimizer,criterion)
                self.eval(criterion)
                
            run_id = self.checkpoint_settings.run_id
            write_path = self.checkpoint_settings.write_path
            dummy_input = torch.randn(1, 3, 32, 32, device=self.device)
            torch.onnx.export(self.model, dummy_input, f'{write_path}/{run_id}.onnx')
        except (KeyboardInterrupt) as ex:
            print(f'Training was interrupted. Please wait while the model is generated.')
            run_id = self.checkpoint_settings.run_id
            write_path = self.checkpoint_settings.write_path
            dummy_input = torch.randn(1, 3, 32, 32, device=self.device)
            torch.onnx.export(self.model, dummy_input, f'{write_path}/{run_id}.onnx')
            checkpoint_dict = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "critetion": criterion.state_dict(),
            }
            torch.save(checkpoint_dict, os.path.join(write_path, 'checkpoint.pth'))
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
