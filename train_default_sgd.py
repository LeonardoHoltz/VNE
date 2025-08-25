import hydra
from omegaconf import DictConfig, OmegaConf
from VNE.utils.builder import instantiate

@hydra.main(version_base=None, config_path="VNE/conf", config_name="default_sgd")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    trainer = build_trainer(cfg)
    trainer.train()

def build_trainer(cfg):
    train_factory = instantiate(cfg.trainer, _partial_=True)
    return train_factory(cfg)

if __name__ == "__main__":
    main()
