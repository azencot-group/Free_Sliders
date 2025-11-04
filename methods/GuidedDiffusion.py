from typing_extensions import Optional
from diffusers import CogVideoXDPMScheduler
from tqdm import tqdm
import random
from base_models.BaseModelCogVideoX import BaseModel
import torch
import neptune
from loggers.base_logger import BaseLogger
from utils.utils import concat_embeddings, plot_latents, export_video, flush, select_variation_batch
from diffusers import LMSDiscreteScheduler
import pickle
import os
import torch
import json
from pathlib import Path

class GuidedDiffusion:
    def __init__(self, base_model):
        self.base_model = base_model

    def __call__(self, args, text_embeddings, prompts, logger : Optional[BaseLogger]=None):
        assert logger is not None
        print("Guided Diffusion Experiment")
        logger["sys/tags"].add("Guided")
        if args.normalize_noise:
            logger["sys/tags"].add("normalize_noise")
        if args.explore_ranges:
            logger["sys/tags"].add("explore_ranges")
        if args.optimal_scales and args.BaseModel == "T2I":
            logger["sys/tags"].add("optimal_scales")
            out_path = Path(f"guided_{args.optimal_scales_per_concept}_{args.BaseModel}.json")
            out_path = out_path.with_name(out_path.name)
            optimal_scales_data = json.loads(Path(out_path).read_text())

        all_concepts_data = {}

        for concept_type, prompt_list in prompts.items():
            if concept_type in text_embeddings:

                for i, prompt in enumerate(prompt_list):
                    tag = prompt.tag
                    text_embeds = text_embeddings[concept_type][i]

                    all_concepts_data[tag] = {}

                    if args.optimal_scales and args.BaseModel == "T2I":
                        pos_ratio = '1.0'
                        neg_ratio = '1.0'
                        intermediate_scale_num = '7'
                        args.scales = optimal_scales_data["guided"][tag][pos_ratio][neg_ratio][intermediate_scale_num]["optimal_scales"]

                    if args.optimal_scales and args.BaseModel == "T2V":
                        logger["sys/tags"].add("optimal_scales")
                        out_path = Path(f"scores_and_ranges/guided_optimal_ranges_{args.BaseModel}_{prompt.tag}.json")
                        out_path = out_path.with_name(out_path.name)
                        optimal_scales_data = json.loads(Path(out_path).read_text())
                        pos_ratio = next(iter(optimal_scales_data["guided"][prompt.tag]))
                        neg_ratio = next(iter(optimal_scales_data["guided"][prompt.tag][pos_ratio]))
                        intermediate_scale_num = next(iter(optimal_scales_data["guided"][prompt.tag][pos_ratio][neg_ratio]))
                        args.scales = optimal_scales_data["guided"][prompt.tag][pos_ratio][neg_ratio][intermediate_scale_num]["optimal_scales"]

                    print(f"\nComputing Guided Diffusion Sliders for - "
                          f"Tag: {tag}\n"
                          f"Neutral Prompt: {prompt.neutral}\n"
                          f"Positive Prompt: {prompt.positive}\n"
                          f"Negative Prompt: {prompt.negative}")
                    random.seed(42)
                    seeds = [random.randint(1, 5000) for _ in range(args.num_images)]

                    for seed in seeds:
                        all_concepts_data[tag][seed] = {}
                        dir_name = f"{args.BaseModel}_sd1-5_latents/Guided/{tag}/{args.guidance}/{args.guidance2}/{args.t}"
                        os.makedirs(dir_name, exist_ok=True)
                        # File path: one file per seed
                        save_path = os.path.join(dir_name, f"{seed}.pkl")

                        latents = self.base_model.get_initial_latents(seed=seed).to(
                                                                self.base_model.device, dtype=torch.float32)  # @TODO add dtype
                        self.run_guided(
                                        latents=latents,
                                        prompt_pair=text_embeds,  # Should contain neutral, positive, and negative prompts
                                        prompts=prompt,  # Textual representation of these prompts
                                        timestep_to=args.t,  # Timestep where intervention occurs
                                        timesteps=50,  # Total diffusion steps
                                        scales=args.scales_to_explore if args.explore_ranges else args.scales,  # Scaling factor for positive-negative difference
                                        logger=logger,
                                        guidance1=args.guidance,  # Guidance scale for classifier-free guidance
                                        guidance2=args.guidance2,
                                        tag=prompt.tag,
                                        seed=seed,
                                        prompt_list=prompt_list,
                                        save_latents= args.save_latents,
                                        save_path=save_path,
                                        latent_data=all_concepts_data[tag][seed],
                                        normalize_noise=args.normalize_noise,
                                        explore_ranges=args.explore_ranges,  # Explore ranges for the concept
                                    )

    def run_guided(self, latents, prompt_pair, prompts, timestep_to, timesteps,
                 scales, logger, guidance1=6.0, guidance2=1.0, tag="", seed=42,
                   random_bias=True, prompt_list=None, save_latents=False, save_path=None, latent_data=None, normalize_noise=False, explore_ranges=False):
        """
        Perform the guided diffusion experiment:
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
        - scale: Scaling factor for the positive-negative difference.
        - logger: Neptune run object for logging.
        - guidance: Guidance scale for classifier-free guidance.
        """
        if scales is None:
            scales = [0, 1, 2, 3]
        self.base_model.noise_scheduler.set_timesteps(
            self.base_model.max_steps, device=self.base_model.device
        )
        generator1 = torch.Generator().manual_seed(seed)
        # Load existing data if present
        if save_path is not None and os.path.exists(save_path):
            with open(save_path, "rb") as f:
                scale_latents = pickle.load(f)
        else:
            scale_latents = {}

        missing_scales = [scale for scale in scales if scale not in scale_latents]
        if missing_scales:
            print(f"Running missing scales: {missing_scales}")

            latents = latents * self.base_model.noise_scheduler.init_noise_sigma

            # Step 1: Diffuse using the neutral prompt until the intervention point.
            positive_embeddings, negative_embeddings = select_variation_batch(prompt_pair)
            neutral_embeddings = prompt_pair.neutral

            with torch.no_grad():

                latents, generator1 = self.base_model.diffusion(latents,
                                                    neutral_embeddings,
                                                    0,
                                                    timestep_to,
                                                    guidance1,
                                                    process_str="Diffusing until splitting point",
                                                    generator=generator1)

                # Step 2: Compute the modified noise prediction.
                old_pred_original_sample = None

                base_latents = latents

                ns = self.base_model.noise_scheduler
                import copy

                for scale in missing_scales:
                    generator2 = copy.deepcopy(generator1)

                    latents = base_latents
                    noise_scheduler_cp = copy.deepcopy(ns)
                    latent_data[scale] = {}
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

                        # Modify noise prediction: neutral + scale * (positive - negative)
                        if not normalize_noise:
                            modified_noise = neutral_noise + scale * (positive_noise - negative_noise).mean(0)[None]
                        else:
                            # normalize the direction of the noise
                            direction = positive_noise - negative_noise
                            direction = direction / (
                                        torch.linalg.vector_norm(direction, dim=(1, 2, 3), keepdim=True) + 1e-6)
                            modified_noise = neutral_noise + scale * direction

                        # Step forward in diffusion
                        extra_step_kwargs = {'eta': 0.0, 'generator': generator2}
                        if isinstance(self.base_model.noise_scheduler, LMSDiscreteScheduler):
                            latents = noise_scheduler_cp.step(modified_noise, t, latents).prev_sample
                        elif isinstance(self.base_model.noise_scheduler, CogVideoXDPMScheduler):
                            latents, old_pred_original_sample = noise_scheduler_cp.step(
                                modified_noise,
                                old_pred_original_sample,
                                t,
                                (self.base_model.noise_scheduler.timesteps[0:50])[j - 1] if j > 0 else None,
                                latents,
                                **extra_step_kwargs,
                                return_dict=False,
                            )
                        else:
                            latents = noise_scheduler_cp.step(modified_noise, t, latents, **extra_step_kwargs, return_dict=False)[0]

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
                                    experiment="Guided",
                                    prompt_list=prompt_list,
                                    explore_ranges=explore_ranges,
                                    save_path=save_path,
                                    compute_metrics_results=False,
                                    seed=seed,
                                    )

    def find_ranges_per_concept(self, args, concept, text_embeddings, prompts, logger : Optional[BaseLogger]=None):
        assert logger is not None

        print(f"Guided Diffusion Find Ranges Concept {concept}")
        logger["sys/tags"].add("Guided")
        if args.normalize_noise:
            logger["sys/tags"].add("normalize_noise")
        if args.explore_ranges:
            logger["sys/tags"].add("explore_ranges")

        all_seeds_data = {}
        for concept_type, prompt_list in prompts.items():
            if concept_type in text_embeddings:
                for i, prompt in enumerate(prompt_list):
                    tag = prompt.tag
                    if tag != concept:
                        continue
                    text_embeds = text_embeddings[concept_type][i]

                    random.seed(24)
                    seeds = [random.randint(1, 5000) for _ in range(args.num_images)]

                    for seed in seeds:
                        dir_name = f"{args.BaseModel}_latents/Guided/{tag}/{args.guidance}/{args.guidance2}/{args.t}"
                        os.makedirs(dir_name, exist_ok=True)
                        # File path: one file per seed
                        save_path = os.path.join(dir_name, f"{seed}.pkl")

                        all_seeds_data[seed] = {}
                        latents = self.base_model.get_initial_latents(1, seed=seed).to(
                            self.base_model.device, dtype=torch.float32)  # @TODO add dtype
                        slider_imgs, contextual_clip, positive_clip, negative_clip, lpips, img_slider = self.run_guided(
                            latents=latents,
                            prompt_pair=text_embeds,  # Should contain neutral, positive, and negative prompts
                            prompts=prompt,  # Textual representation of these prompts
                            timestep_to=args.t,  # Timestep where intervention occurs
                            timesteps=50,  # Total diffusion steps
                            scales=args.scales_to_explore if args.explore_ranges else args.scales,
                            logger=logger,
                            guidance1=args.guidance,  # Guidance scale for classifier-free guidance
                            guidance2=args.guidance2,
                            tag=prompt.tag,
                            seed=seed,
                            prompt_list=prompt_list,
                            save_path=save_path,
                            latent_data=all_seeds_data[seed],
                            normalize_noise=args.normalize_noise,
                            explore_ranges=args.explore_ranges,  # Explore ranges for the concept
                        )
                        all_seeds_data[seed] = {"slider_imgs": slider_imgs,
                                                "contextual_clip": contextual_clip,
                                                "positive_clip": positive_clip,
                                                "negative_clip": negative_clip,
                                                "lpips": lpips,
                                                "img_slider": img_slider}
            return all_seeds_data
