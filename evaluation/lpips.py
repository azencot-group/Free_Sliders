# @Ref     :   https://github.com/christophschuhmann/improved-aesthetic-predictor
import logging
from pathlib import Path
from torchvision import transforms

import numpy as np
import torch
from evaluation.lpips_utils import LPIPS

from evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

class Lpips(BaseEvaluator):
    def __init__(
        self,
        index_file: Path, 
        edit_video_dir: Path,
        reference_video_dir: Path,
        edit_prompt: str,
        device: torch.device,
        pretrained_model_name: str = None,
    ):
        super().__init__(index_file, edit_video_dir, reference_video_dir, edit_prompt, device)
        self.loss_fn = LPIPS(net='alex', spatial=False).to(self.device)  # Non-spatial LPIPS
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Resize((64, 64)),  # Resize if needed
        ])

    def range(self):
        return 0, 1  # LPIPS scores range from 0 (identical) to 1 (very different)

    def preprocess(self, sample):
        edit_video = torch.stack([self.transform(frame) for frame in sample['edit_video']]).to(self.device)
        reference_video = torch.stack([self.transform(frame) for frame in sample['reference_video']]).to(self.device)

        # Convert to LPIPS input format (BCHW) and normalize to [-1,1]
        edit_video = edit_video * 2 - 1
        reference_video = reference_video * 2 - 1

        return {
            'video_id': sample['video_id'],
            'edit_id': sample['edit_id'],
            'edit_video': edit_video,
            'reference_video': reference_video,
        }

    @torch.no_grad()
    def evaluate(self, args) -> float:
        """Compute LPIPS score per frame and return the average."""
        edit_video = args['edit_video']
        reference_video = args['reference_video']

        lpips_scores = []
        for i in range(edit_video.shape[0]):  # Loop over frames
            frame1 = edit_video[i].unsqueeze(0)
            frame2 = reference_video[i].unsqueeze(0)
            lpips_score = self.loss_fn(frame1, frame2).mean().item()
            lpips_scores.append(lpips_score)

        return np.mean(lpips_scores)  # Return average LPIPS across frames
