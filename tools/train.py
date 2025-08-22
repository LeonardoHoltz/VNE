import hydra
from omegaconf import DictConfig, OmegaConf
from utils.builder import instantiate

@hydra.main(version_base=None, config_path="../conf", config_name="main")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    trainer = instantiate(cfg.trainer, cfg=cfg)
    trainer.train()

if __name__ == "__main__":
    main()
