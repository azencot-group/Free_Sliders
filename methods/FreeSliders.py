from typing_extensions import Optional
from tqdm import tqdm
import random
from diffusers import LMSDiscreteScheduler
import pickle
import os
import torch

from loggers.base_logger import BaseLogger

class FreeSliders:
    def __init__(self, base_model):
        self.base_model = base_model

    def __call__(self, args, text_embeddings, prompts, seed=42, logger : Optional[BaseLogger]=None):
        assert logger is not None

        for concept_type, prompt_list in prompts.items():
            if concept_type in text_embeddings:

                for i, prompt in enumerate(prompt_list):
                    tag = prompt.tag
                    text_embeds = text_embeddings[concept_type][i]

                    print(f"\nComputing FreeSliders for - "
                          f"Tag: {tag}\n"
                          f"Neutral Prompt: {prompt.neutral}\n"
                          f"Positive Prompt: {prompt.positive}\n"
                          f"Negative Prompt: {prompt.negative}")

                    random.seed(seed)
                    seeds = [random.randint(1, 5000) for _ in range(args.num_images)]

                    for seed in seeds:

                        # File path: save latents per seed
                        dir_name = f"{args.BaseModel}_sd_latents/FreeSliders/{tag}/{args.guidance}/{args.guidance2}/{args.t}"
                        os.makedirs(dir_name, exist_ok=True)
                        save_path = os.path.join(dir_name, f"{seed}.pkl")

                        latents = self.base_model.get_initial_latents(seed=seed).to(
                                                                self.base_model.device, dtype=torch.float32)
                        self.run_freesliders(
                                        latents=latents,
                                        prompt_pair=text_embeds,  # Should contain neutral, positive, and negative prompts
                                        prompts=prompt,  # Textual representation these prompts
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

    def run_freesliders(self, latents, prompt_pair, prompts, timestep_to=15, timesteps=50,
                 scales=[0, 1, 2, 3], logger=None, guidance1=6.0, guidance2=1.0, tag="", seed=42,
                 prompt_list=None, save_path=None):
        """
        Perform the FreeSliders experiment:
        1. Diffuse using the neutral prompt until a specific timestep t.
        2. At timestep t, modify the noise at each step using:
            neutral_noise_prediction + scale * (positive_noise_prediction - negative_noise_prediction).
        3. Continue diffusion using the modified noise.

        Args:
        - latents: Initial noise latents.
        - prompt_pair: Object containing neutral, positive, and negative prompts.
        - prompts: Object with corresponding textual prompts for logging.
        - timestep_to: Timestep where noise modification occurs.
        - timesteps: Total number of diffusion steps.
        - scales: Scaling factors for the positive-negative difference.
        - logger: Neptune run object for logging.
        - guidance: Guidance scale for classifier-free guidance for neutral prompt.
        - guidance2: Guidance scale for classifier-free guidance for positive/negative prompts.
        - tag: Concept tag for logging purposes.
        - seed: Random seed for reproducibility.
        - prompt_list: List of all prompts for logging.
        - save_path: Path to save/load intermediate latents.
        """

        # Load existing data if present
        if save_path is not None and os.path.exists(save_path):
            with open(save_path, "rb") as f:
                scale_latents = pickle.load(f)
        else:
            scale_latents = {}


        missing_scales = [scale for scale in scales if scale not in scale_latents]
        if missing_scales:
            print(f"Running missing scales: {missing_scales}")

            self.base_model.noise_scheduler.set_timesteps(
                self.base_model.max_steps, device=self.base_model.device
            )
            generator1 = torch.Generator().manual_seed(seed)

            latents = latents * self.base_model.noise_scheduler.init_noise_sigma

            positive_embeddings = prompt_pair.positive
            negative_embeddings = prompt_pair.negative
            neutral_embeddings = prompt_pair.neutral

            with torch.no_grad():

                # Step 1: Diffuse using the neutral prompt until the intervention point.
                latents, generator1 = self.base_model.diffusion(latents,
                                                    neutral_embeddings,
                                                    0,
                                                    timestep_to,
                                                    guidance1,
                                                    process_str="Diffusing until splitting point",
                                                    generator=generator1)

                base_latents = latents
                ns = self.base_model.noise_scheduler
                import copy

                # Step 2: Compute the modified noise prediction for each scale.
                for scale in missing_scales:
                    latents = base_latents
                    generator2 = copy.deepcopy(generator1)
                    noise_scheduler_cp = copy.deepcopy(ns)
                    for j, t in enumerate(tqdm(noise_scheduler_cp.timesteps[timestep_to:timesteps],
                                               desc=f"Diffusing with scale {scale}"
                                               ), start=timestep_to):
                        neutral_noise = self.base_model.predict_noise(
                                            t,
                                           latents,
                                           neutral_embeddings,
                                           guidance1,
                                            scheduler=noise_scheduler_cp,
                                            do_classifier_free_guidance=True)

                        positive_noise = self.base_model.predict_noise(
                                            t,
                                           torch.cat([latents] * prompt_pair.batch_size),
                                           positive_embeddings,
                                           guidance2,
                                            scheduler=noise_scheduler_cp,
                                            do_classifier_free_guidance=False)
                        negative_noise = self.base_model.predict_noise(
                                            t,
                                           torch.cat([latents] * prompt_pair.batch_size),
                                           negative_embeddings,
                                           guidance2,
                                            scheduler=noise_scheduler_cp,
                                            do_classifier_free_guidance=False)

                        # neutral_noise, positive_noise, negative_noise = self.base_model.predict_noise_triplet(t, latents, neutral_embeddings, positive_embeddings,
                        #                       negative_embeddings,
                        #                       guidance1, guidance2, scheduler=noise_scheduler_cp)

                        modified_noise = neutral_noise + scale * (positive_noise - negative_noise).mean(0)[None]

                        # Step forward in diffusion
                        if isinstance(self.base_model.noise_scheduler, LMSDiscreteScheduler):
                            latents = noise_scheduler_cp.step(modified_noise, t, latents).prev_sample

                    scale_latents[scale] = latents


            # Save back
            with open(save_path, "wb") as f:
                pickle.dump(scale_latents, f)
            print(f"Saved latents to {save_path}")

        # Step 3: Visualization and metrics computation
        return self.base_model.log_and_evaluate(
                                    {scale: scale_latents[scale] for scale in scales},
                                    prompts,
                                    logger,
                                    tag,
                                    t=timestep_to,
                                    experiment="FreeSliders",
                                    prompt_list=prompt_list,
                                    save_path=save_path,
                                    # compute_metrics_results=False,
                                    seed=seed,
                                    )

