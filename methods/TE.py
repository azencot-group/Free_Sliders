from typing_extensions import Optional
import random
import torch
from pathlib import Path
import pickle
import os
import sys

# Add the parent directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import flush
from loggers.base_logger import BaseLogger


class TE:
    def __init__(self, base_model):
        self.base_model = base_model

    def __call__(self, args, text_embeddings, prompts, seed=42, logger : Optional[BaseLogger]=None):
        assert logger is not None

        results = None
        for tag, prompt_list in prompts.items():
            if tag in text_embeddings:

                for i, prompt in enumerate(prompt_list):
                    text_embeds = text_embeddings[tag][i]
            
                    random.seed(seed)
                    seeds = [random.randint(1, 5000) for _ in range(args.num_images)]
                    
                    for seed in seeds:

                        # File path: one file per seed
                        dir_name = f"{args.BaseModel}_latents/EmbedsInter/{prompt.tag}/{args.guidance}/{args.guidance2}/{args.t}"
                        os.makedirs(dir_name, exist_ok=True)
                        save_path = os.path.join(dir_name, f"{seed}.pkl")

                        latents = self.base_model.get_initial_latents(1, seed=seed).to(
                                                                self.base_model.device, dtype=torch.float32)  # @TODO add dtype
                        results = self.run_interpolate_embeds(
                                        latents=latents,
                                        prompt_pair=text_embeds,  # Should contain neutral, positive, and negative prompts
                                        prompts=prompt,  # Textual representation of these prompts
                                        timestep_to=args.t,  # Timestep where intervention occurs
                                        timesteps=50,  # Total diffusion steps @TODO from config
                                        scales=args.scales,  # Scaling factor for positive-negative difference
                                        logger=logger,
                                        guidance1=args.guidance,  # Guidance scale for classifier-free guidance
                                        guidance2=args.guidance2,
                                        tag=prompt.tag,
                                        seed=seed,
                                        prompt_list=prompt_list,
                                        save_path=save_path,
                                    )
        return results


    def run_interpolate_embeds(self, latents, prompt_pair, prompts, timestep_to, timesteps,
                 scales, logger, guidance1=6.0, guidance2=1.0, tag="static", seed=42,
                   prompt_list=None, normalize_noise=False, save_path=None):
        """
        Perform the text embeddings manipulation diffusion experiment:
        1. Diffuse using the neutral prompt until a specific timestep t.
        2. At timestep t, modify the noise at each step using as text embeddings:
            neutral_prompt_embeds + scale * (positive_prompt_embeds - negative_prompt_embeds).
        3. Continue diffusion using the modified text embeddings.

        Args:
        - latents: Initial noise latents.
        - prompt_pair: Object containing neutral, positive, and negative prompts.
        - prompts: Object with corresponding textual prompts for logging.
        - timestep_to: Timestep where noise modification occurs.
        - timesteps: Total number of diffusion steps.
        - scale: Scaling factor for the positive-negative difference.
        - logger: Neptune run object for logging.
        - guidance: Guidance scale for classifier-free guidance.
        """
        if scales is None:
            scales = [0, 1, 2, 3]

        # Load existing data if present
        if save_path is not None and os.path.exists(save_path):
            with open(save_path, "rb") as f:
                scale_latents = pickle.load(f)
        else:
            scale_latents = {}

        missing_scales = [scale for scale in scales if scale not in scale_latents]
        if missing_scales:
            print(f"Running missing scales: {missing_scales}")

            self.base_model.noise_scheduler.set_timesteps(self.base_model.max_steps, device=self.base_model.device)
            generator1 = torch.Generator().manual_seed(seed)
        
            with torch.no_grad():
                latents = latents * self.base_model.noise_scheduler.init_noise_sigma

                # Step 1: Diffuse using the neutral prompt until the intervention point.
                positive_embeddings = prompt_pair.positive
                negative_embeddings = prompt_pair.negative
                neutral_embeddings = prompt_pair.neutral

                latents, generator1 = self.base_model.diffusion(latents,
                                                    neutral_embeddings,
                                                    0,
                                                    timestep_to,
                                                    guidance1,
                                                    process_str="Diffusing until splitting point",
                                                    generator=generator1,
                                                    )

                base_latents = latents

                ns = self.base_model.noise_scheduler
                import copy

                for scale in missing_scales:
                    generator2 = copy.deepcopy(generator1)

                    latents = base_latents
                    noise_scheduler_cp = copy.deepcopy(ns)


                    modified_text_embeds = neutral_embeddings + scale * (positive_embeddings - negative_embeddings).mean(0)[None]

                    latents, generator2 = self.base_model.diffusion(latents,
                                                        modified_text_embeds,
                                                        timestep_to,
                                                        50,
                                                        guidance2,
                                                        process_str="Diffusing from splitting point",
                                                        scheduler=noise_scheduler_cp,
                                                        normalize_noise=normalize_noise,
                                                        generator=generator2,
                                                        )

                    scale_latents[scale] = latents


            # Save back
            with open(save_path, "wb") as f:
                pickle.dump(scale_latents, f)
            print(f"Saved latents to {save_path}")

        # Step 3: Visualization and metrics computation.
        results = self.base_model.log_and_evaluate({scale: scale_latents[scale] for scale in scales},
                                        prompts, 
                                        logger, 
                                        tag,
                                            t=timestep_to, 
                                            experiment="TE",
                                            prompt_list=prompt_list, 
                                            save_path=save_path,
                                        # compute_metrics_results=False,
                                        )

        flush()
        return results
