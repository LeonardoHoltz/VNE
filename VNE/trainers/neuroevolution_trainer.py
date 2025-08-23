# translation:
# epoch = generation
# forward = 

import torch
import torch.nn as nn
import torch.optim as optim

from utils.builder import instantiate

class DefaultTrainer():
    def __init__(self, cfg):
        self.device = self.set_device(cfg.device)
        self.model = self.build_model(cfg.model, self.device)
        self.criterion = self.build_criterion(cfg.criterion)
        self.optimizer = self.build_optimizer(cfg.optimizer)
        self.train_loader, self.test_loader = self.build_dataset(cfg.dataset)
        self.epochs = cfg.epochs
    
    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
    
    def run_epoch(self):
        for epoch in range(1, self.epochs + 1):
            self.run_step(epoch)
    
    def run_step(self, epoch):
        total_loss, correct = 0, 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # forward
            output = self.model(data)
            loss = self.criterion(output, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward() # I believe NEAT dont need this
            self.optimizer.step()

            # log metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        acc = 100. * correct / len(self.train_loader.dataset)
        print(f"Train Epoch {epoch}: Loss={total_loss/len(self.train_loader):.4f}, Acc={acc:.2f}%")
        
    def before_step(self):
        pass
    
    def after_step(self):
        pass
    
    def before_train(self):
        self.model.train()
    
    def before_epoch(self):
        pass
    
    def after_epoch(self):
        pass
    
    def after_train(self):
        pass
    
    def set_device(self, device_cfg):
        if device_cfg == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device("cpu")
    
    def build_model(self, model_cfg, device):
        return instantiate(model_cfg).to(device)
    
    def build_criterion(self, criterion_cfg):
        return instantiate(criterion_cfg)
    
    def build_optimizer(self, optimizer_cfg):
        return instantiate(optimizer_cfg)
    
    def build_dataset(self, dataset_cfg):
        dataset = instantiate(dataset_cfg)
        return dataset.train_loader, dataset.test_loader

