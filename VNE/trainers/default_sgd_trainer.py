import torch
from tqdm import tqdm
import logging

from VNE.utils.builder import instantiate

class DefaultSGDTrainer():
    def __init__(self, cfg):
        self.device = self.set_device(cfg.device)
        self.model = self.build_model(cfg.model, self.device)
        self.criterion = self.build_criterion(cfg.criterion)
        self.optimizer = self.build_optimizer(cfg.optimizer, self.model.parameters())
        self.train_loader, self.test_loader = self.build_dataset(cfg.dataset)
        self.epochs = cfg.epochs
        
        logging.basicConfig(level=logging.INFO,
                            format='[%(levelname)s] %(message)s)'
        )
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        self.before_train()
        for cur_epoch in range(1, self.epochs+1):
            self.before_epoch(cur_epoch)
            self.run_epoch()
            self.after_epoch()
        self.after_train()
    
    def run_epoch(self):
        with tqdm(total=len(self.train_loader), disable=False) as pbar:
            for batch_idx, data in enumerate(self.train_loader):
                loss, pred = self.run_step(data)
                pbar.set_description(f"Loss: {loss.item()}")
                pbar.update(1)
    
    def run_step(self, data):
        X, y = data
        X, y = X.to(self.device), y.to(self.device)

        # forward
        output = self.model(X)
        loss = self.criterion(output, y)

        # backward
        self.optimizer.zero_grad()
        loss.backward() # I believe NEAT dont need this
        self.optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        
        return loss, pred
        
    def before_step(self):
        pass
    
    def after_step(self):
        pass
    
    def before_train(self):
        self.model.train()
    
    def before_epoch(self, cur_epoch):
        self.logger.info(f"Epoch [{cur_epoch}/{self.epochs}]:")
    
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
    
    def build_optimizer(self, optimizer_cfg, params):
        optim_factory = instantiate(optimizer_cfg, _partial_=True)
        return optim_factory(params)
    
    def build_dataset(self, dataset_cfg):
        dataset = instantiate(dataset_cfg)
        return dataset.train_loader, dataset.test_loader

