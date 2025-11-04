import warnings
from pathlib import Path

from .base_logger import BaseLogger
from typing import Dict, Any, List
import numpy as np
from PIL import Image


def is_basic(x):
    return isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


def convert_no_basic_to_str(sub_dict: Dict[str, Any]):
    return {k: v if is_basic(v)
    else str(v) if not isinstance(v, dict) else convert_no_basic_to_str(v)
            for k, v in sub_dict.items()}


def convert_no_basic_to_str_from_any(p: Any):
    if is_basic(p):
        return p
    elif isinstance(p, dict):
        return convert_no_basic_to_str(p)
    else:
        return str(p)


class WandbLogger(BaseLogger):

    def __init__(self, project=None, *args, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)
        if self.rank != 0:
            return
        import wandb
        self.wandb = wandb
        local_dir_api_project = ['wandb', 'neptune', 'logger', '']
        if project is None:
            for folder in local_dir_api_project:
                local_path_api_project = Path(folder) / 'project.txt'
                if local_path_api_project.exists():
                    project = local_path_api_project.read_text().strip()
                    if folder == 'neptune':
                        project = project.split('/')[-1]
                    break
        if project is None:
            warnings.warn('''Please create a file at neptune/project.txt with your Neptune project name''')
            raise FileNotFoundError('project file name not found')
        self.run = wandb.init(
            # set the wandb project where this run will be logged
            project=project,

        )



    def stop(self):
        if self.rank == 0:
            self.run.finish()

    def log(self, name: str, data: Any, step=None):
        if self.rank == 0:
            self.wandb.log({name: data})

    def _log_fig(self, name: str, fig: Any):
        if self.rank == 0:
            if isinstance(fig, np.ndarray):
                if fig.dtype != np.uint8:
                    fig = fig * 255
                    fig = fig.astype(np.uint8)
                fig = Image.fromarray(fig)
            self.wandb.log({name: self.wandb.Image(fig)})

    def log_hparams(self, params: Dict[str, Any]):
        if self.rank == 0:
            params = convert_no_basic_to_str(params)
            if isinstance(params, dict):
                self.wandb.config.update(params)
            else:
                self.wandb.config.update({'hparams': params})


    def log_params(self, params: Dict[str, Any]):
        if self.rank == 0:
            params = convert_no_basic_to_str(params)
            if isinstance(params, dict):
                self.wandb.config.update(params)
            else:
                self.wandb.config.update({'params': params})

    def add_tags(self, tags: List[str]):
        if self.rank == 0:
            self.run.tags = self.run.tags + tuple(tags)

    def log_name_params(self, name: str, params: Any):
        if self.rank == 0:
            params = convert_no_basic_to_str_from_any(params)
            self.wandb.config.update({name: params}, allow_val_change=True)
