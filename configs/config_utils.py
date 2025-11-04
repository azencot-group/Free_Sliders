from typing import Literal, Optional
import yaml
from pydantic import BaseModel

class PretrainedModelConfig(BaseModel):
    name_or_path: str
    v2: bool = False
    v_pred: bool = False
    clip_skip: Optional[int] = None

class NetworkConfig(BaseModel):
    rank: int = 4
    alpha: float = 1.0

class TrainConfig(BaseModel):
    iterations: int = 500
    lr: float = 1e-4
    optimizer: str = "adamw"
    optimizer_args: str = ""
    lr_scheduler: str = "constant"
    max_denoising_steps: int = 50

class EvalConfig(BaseModel):
    type: str = "diffusion"
    batch_size: int = 1

class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 200
    generated_videos_path: str = "./generated_videos"

class LoggingConfig(BaseModel):
    use_wandb: bool = False
    verbose: bool = False

class OtherConfig(BaseModel):
    use_xformers: bool = False

class RootConfig(BaseModel):
    static_prompts_file: str
    dynamic_prompts_file: str
    pretrained_model: PretrainedModelConfig
    network: NetworkConfig
    train: Optional[TrainConfig]
    eval: Optional[EvalConfig]
    save: Optional[SaveConfig]
    logging: Optional[LoggingConfig]
    other: Optional[OtherConfig]


def load_config_from_yaml(config_path: str) -> RootConfig:
    import os
    print(os.getcwd())
    print(os.path.abspath(config_path))
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.train is None:
        root.train = TrainConfig()

    if root.eval is None:
        root.eval = EvalConfig()

    if root.save is None:
        root.save = SaveConfig()

    if root.logging is None:
        root.logging = LoggingConfig()

    if root.other is None:
        root.other = OtherConfig()

    return root
