from . import config

def get_inference_module():
    from . import inference
    return inference

def get_training_module():
    from . import training
    return training

__all__ = ['config', 'get_inference_module', 'get_training_module']