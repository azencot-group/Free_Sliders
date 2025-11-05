from typing import Literal, Optional
import yaml
from pydantic import BaseModel

class PretrainedModelConfig(BaseModel):
    name_or_path: str
    v2: bool = False
    v_pred: bool = False
    clip_skip: Optional[int] = None
    precision: str = "fp16"
    height: int = 512
    width: int = 512
    max_steps: int = 50

class NetworkConfig(BaseModel):
    rank: int = 4
    alpha: float = 1.0
    training_method: str = "noxattn"

class EvalConfig(BaseModel):
    type: str = "diffusion"
    batch_size: int = 1

class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 200
    output_path: str = "./outputs"
    precision: str = "bfloat16"

class LoggingConfig(BaseModel):
    use_wandb: bool = False
    verbose: bool = False

class OtherConfig(BaseModel):
    use_xformers: bool = False

class RootConfig(BaseModel):
    prompts_file: str
    pretrained_model: PretrainedModelConfig
    network: NetworkConfig
    eval: Optional[EvalConfig]
    save: Optional[SaveConfig]
    logging: Optional[LoggingConfig]
    other: Optional[OtherConfig]


def load_config_from_yaml(config_path: str) -> RootConfig:
    import os
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.eval is None:
        root.eval = EvalConfig()

    if root.save is None:
        root.save = SaveConfig()

    if root.logging is None:
        root.logging = LoggingConfig()

    if root.other is None:
        root.other = OtherConfig()

    return root
