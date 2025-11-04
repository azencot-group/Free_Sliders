import yaml
import copy
from pydantic import BaseModel, model_validator
import torch
from typing import Optional


class PromptSettings(BaseModel):  # yaml のやつ
    neutral: str
    positive: str = None   # if None, neutral will be used
    negative: str = None  # if None, neutral will be used
    guidance_scale: float = 1.0  # default is 1.0
    resolution: int = 512  # default is 512 @TODO change?
    dynamic_resolution: bool = False  # default is False @TODO need this?
    batch_size: int = 1  # default is 1
    tag: str = None
    variations: Optional[list[dict[str, str]]] = []  # list of biased (neutral, positive, negative) tuples

    @model_validator(mode='before')
    def fill_prompts(self):
        keys = self.keys()
        if "neutral" not in keys:
            raise ValueError("neutral prompt must be specified")
        if "positive" not in keys:
            self["positive"] = self["neutral"]
        if "negative" not in keys:
            self["negative"] = self["neutral"]

        return self


class PromptEmbedsPair:
    neutral: torch.FloatTensor      # base prompt
    positive: torch.FloatTensor     # positive concept
    negative: torch.FloatTensor     # negative concept

    variations: Optional[list[tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]] # biased variations

    guidance_scale: float
    resolution: int
    dynamic_resolution: bool
    batch_size: int
    dynamic_crops: bool

    loss_fn: torch.nn.Module

    def __init__(
            self,
            loss_fn: torch.nn.Module,
            neutral: torch.FloatTensor,
            positive: torch.FloatTensor,
            negative: torch.FloatTensor,
            settings: PromptSettings,
            variations: Optional[list[tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]],
    ) -> None:
        self.loss_fn = loss_fn
        self.neutral = neutral
        self.positive = positive
        self.negative = negative

        self.variations = variations  # list of biased (neutral, positive, negative) tuples

        self.guidance_scale = settings.guidance_scale
        self.resolution = settings.resolution
        self.dynamic_resolution = settings.dynamic_resolution
        self.batch_size = settings.batch_size


    def _scale(
            self,
            neutral_latents: torch.FloatTensor,
            positive_latents: torch.FloatTensor,
            negative_latents: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.loss_fn(
            neutral_latents,
            neutral_latents
            + self.guidance_scale * (positive_latents - negative_latents)
        )


class PromptEmbedsCache:  # 使いまわしたいので
    prompts: dict[str, torch.FloatTensor] = {}

    def __setitem__(self, __name: str, __value: torch.FloatTensor) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str) -> Optional[torch.FloatTensor]:
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None


def load_prompts_from_yaml(path, attributes=None):
    if attributes is None:
        attributes = []
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    # print(prompts)
    if len(prompts) == 0:
        raise ValueError("prompts file is empty")
    if len(attributes) != 0:
        newprompts = []
        for i in range(len(prompts)):
            for att in attributes:
                copy_ = copy.deepcopy(prompts[i])
                copy_['neutral'] = att + ' ' + copy_['neutral']
                copy_['positive'] = att + ' ' + copy_['positive']
                copy_['negative'] = att + ' ' + copy_['negative']
                newprompts.append(copy_)
    else:
        newprompts = copy.deepcopy(prompts)

    # print(newprompts)
    # print(len(prompts), len(newprompts))
    prompt_settings = [PromptSettings(**prompt) for prompt in newprompts]

    return prompt_settings
