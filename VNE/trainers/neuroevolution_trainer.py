import torch
from evotorch.neuroevolution import SupervisedNE
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation
from evotorch.logging import StdOutLogger, PandasLogger
from tqdm import tqdm
import logging

from VNE.utils.builder import instantiate

class NeuroEvolutionTrainer():
    def __init__(self, cfg):
        self.device = self.set_device(cfg.device)
        self.model = self.build_model(cfg.model, self.device)
        self.criterion = self.build_criterion(cfg.criterion)
        self.train_dataset, self.test_dataset = self.build_dataset(cfg.dataset)
        self.generations = cfg.generations
        
        self.problem = SupervisedNE(
            dataset=self.train_dataset,
            network=self.model,
            loss_func=self.criterion,
            minibatch_size=cfg.dataset.batch_size,
            common_minibatch=True,
            num_gpus_per_actor='max',
            num_actors=1,
            subbatch_size=50,
            device=self.device,
        )
        
        self.searcher = GeneticAlgorithm(
            problem=self.problem,
            operators=[
                SimulatedBinaryCrossOver(
                    self.problem,
                    tournament_size=cfg.population // 4,
                    cross_over_rate=0.9,
                    eta=2,
                ),
                GaussianMutation(
                    self.problem,
                    stdev=0.1,
                    mutation_probability=0.1,
                ),
            ],
            popsize=cfg.population,
            elitist=True,
        )        
        
    
    def train(self):
        self.before_train()
        self.searcher.run(self.generations)
        self.pandas_logger.to_dataframe().mean_eval.plot()
        self.after_train()
    
    def before_train(self):
        self.stdout_logger = StdOutLogger(self.searcher, interval = 1)
        self.pandas_logger = PandasLogger(self.searcher, interval = 1)
    
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
    
    def build_dataset(self, dataset_cfg):
        dataset = instantiate(dataset_cfg)
        return dataset.train_dataset, dataset.test_dataset

