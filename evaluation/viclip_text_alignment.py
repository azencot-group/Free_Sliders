
import logging
from pathlib import Path
import clip as _clip_pkg
import torch

from evaluation.base_evaluator import BaseEvaluator
from evaluation.viclip_utils import SimpleTokenizer, ViCLIP, clip_image_transform

logger = logging.getLogger(__name__)

class ViclipTextAlignment(BaseEvaluator):
    # The OpenAI CLIP package ships the BPE vocab; resolve it from there
    # rather than requiring a local copy under evaluation/metric_models/.
    pretrained_tokenizer = (
        Path(_clip_pkg.__file__).parent / 'bpe_simple_vocab_16e6.txt.gz').resolve().absolute()
    pretrained_checkpoint = (
        Path(__file__) / '../metric_models/ViClip-InternVid-10M-FLT.pth').resolve().absolute()

    def __init__(
        self,
        index_file: Path, 
        edit_video_dir: Path,
        reference_video_dir: Path,
        edit_prompt: str,
        device: torch.device,
        pretrained_tokenizer: str = None,
        pretrained_checkpoint: str = None,
        stride: int = 3,  # accept 8 frames as input
    ):
        super().__init__(index_file, edit_video_dir, None, edit_prompt, device, stride=stride)
        pretrained_tokenizer = pretrained_tokenizer or self.pretrained_tokenizer
        pretrained_checkpoint = pretrained_checkpoint or self.pretrained_checkpoint

        # logger.debug(f"Loding model {pretrained_checkpoint}")
        tokenizer = SimpleTokenizer(bpe_path=pretrained_tokenizer)
        self.model = ViCLIP(tokenizer=tokenizer, pretrain=pretrained_checkpoint)
        self.model.to(self.device)
        self.model.eval()
        # logger.debug(f"Model {self.model.__class__.__name__} loaded")

        self.image_transform = clip_image_transform(224)

    def range(self):
        return 0, 1

    def preprocess(self, sample):
        text = sample['edit_prompt']
        video = sample['edit_video']
        
        text_inputs = text
        
        frames = []
        for frame in video:
            frames.append(self.image_transform(frame))
        video_inputs = torch.stack(frames).to(self.device)[None] # (1, T, C, H, W)
        # logger.debug(f"video_inputs shape: {video_inputs.shape}")

        return text_inputs, video_inputs

    @torch.no_grad()
    def evaluate(self, args) -> float:
        text_inputs, video_inputs = args

        text_embs: torch.Tensor = self.model.encode_text(text_inputs).float()
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # logger.debug(f"text_embs shape: {text_embs.shape}")

        video_embs: torch.Tensor = self.model.encode_vision(video_inputs, test=True).float()
        video_embs = video_embs / torch.norm(video_embs, dim=-1, keepdim=True)
        # logger.debug(f"video_embs shape: {video_embs.shape}")

        score = (text_embs @ video_embs.T).cpu().squeeze().item()

        return score