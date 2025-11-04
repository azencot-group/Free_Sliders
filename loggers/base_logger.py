from abc import ABC, abstractmethod
from typing import Dict, Any, List

class Itemlogger():

    def __init__(self, name, logger):
        self.name = name
        self.logger : BaseLogger = logger

    def log(self, data):
        self.logger.log(self.name, data)

    def add(self, params):
        self.logger.add(self.name, params)

    def log_fig(self, fig):
        self.logger.log_fig(self.name, fig)

    def log_dict(self, data):
        self.logger.log_dict(self.name, data)

class BaseLogger(ABC):

    def __init__(self, no_plot: bool = False, rank: int = 0, *args, **kwargs):
        super(BaseLogger, self).__init__()
        self.no_plot = no_plot
        self.rank = rank

    def __enter__(self):
        return self

    @abstractmethod
    def stop(self):
        pass

    def __exit__(self, type, value, traceback):
        self.stop()

    @abstractmethod
    def log(self, name: str, data: Any, step=None):
        pass

    def __getitem__(self, key):
        """Allows access to the logger using the item syntax."""
        return Itemlogger(key, self)

    def __setitem__(self, key, value):
        self[key].add(value)

    def log_dict(self, name: str, data: Dict[str, Any], step=None):
        for k, v in data.items():
            self.log(f'{name}/{k}', v.item(), step)

    def log_fig(self, name: str, fig: Any):
        if self.no_plot:
            return
        self._log_fig(name, fig)

    @abstractmethod
    def _log_fig(self, name: str, fig: Any):
        pass

    @abstractmethod
    def log_hparams(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def add_tags(self, tags: List[str]):
        pass

    @abstractmethod
    def log_name_params(self, name : str, params: Any):
        pass

    def add(self, name: str, params: Any):
        """Adds a new logger item."""
        self.log_name_params(name, params)

    def log_audio(self, name : str, path : str):
        pass
