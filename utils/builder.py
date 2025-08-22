# Instantiate objects based on a hydra configuration

import hydra

def instantiate(cfg, **kwargs):
    return hydra.utils.instantiate(cfg, **kwargs)

def call(cfg, **kwargs):
    return hydra.utils.call(cfg, **kwargs)