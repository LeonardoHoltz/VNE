# Instantiate objects based on a hydra configuration

import hydra

def instantiate(build_cfg, **kwargs):
    return hydra.utils.instantiate(build_cfg, **kwargs)

def call(call_cfg, **kwargs):
    return hydra.utils.call(call_cfg, **kwargs)