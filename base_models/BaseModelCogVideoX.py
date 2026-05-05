import gc
import math
import os

import numpy as np
import pandas as pd
import torch
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from tqdm import tqdm

from evaluation.new_slider_metrics import compute_overall_video_scores
from utils.prompt_utils import PromptEmbedsCache, PromptEmbedsPair
from utils.utils import (
    flush,
    get_device,
    plot_latents,
    export_video,
    compute_metrics,
)


class BaseModel:
    """CogVideoX wrapper that mirrors the BaseModel_StableDiffusion / BaseModel_StableAudio surface."""

    def __init__(self, config, run_id=1):
        super().__init__()

        self.run_id = run_id
        self.batch_size = 1

        self.device = get_device()
        # CogVideoX-2b expects bfloat16 on CUDA. MPS lacks bfloat16 support in some
        # ops; fall back to float32 there.
        if self.device.type == "cuda":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

        video_cfg = getattr(config, "video", None)
        self.model_path = (
            video_cfg.pretrained_model if video_cfg is not None else "THUDM/CogVideoX-2b"
        )
        self.height = video_cfg.height if video_cfg is not None else 480
        self.width = video_cfg.width if video_cfg is not None else 720
        self.num_frames = video_cfg.num_frames if video_cfg is not None else 24
        self.max_steps = video_cfg.max_steps if video_cfg is not None else 50

        pipeline = CogVideoXPipeline.from_pretrained(
            self.model_path, torch_dtype=self.weight_dtype
        )
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

        # CPU offload only meaningful on CUDA. On MPS / CPU, place all submodules
        # explicitly on the device — without offloading they would otherwise stay
        # on CPU and mismatch the transformer at forward time.
        if self.device.type == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.device)
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()

        self.vae = pipeline.vae
        self.transformer = pipeline.transformer
        self.noise_scheduler = pipeline.scheduler
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.pipeline = pipeline

        self.transformer.to(self.device, dtype=self.weight_dtype)
        self.transformer.requires_grad_(False)
        self.transformer.eval()

        self.vid_output_path = config.save.output_path

        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor = 8
        self.transformer_in_channels = 16

        self._prime_patch_embed_cache()

    def _prime_patch_embed_cache(self):
        """Precompute positional embeddings for our latent shape on CPU.

        CogVideoXPatchEmbed recomputes pos embeddings on every forward when the
        input shape doesn't match the model's defaults. The recompute uses
        float64, which MPS does not support, so it crashes there. Computing
        once on CPU and caching the result fixes both the crash and the
        per-step recompute cost.

        Forward compares `sample_height/width` against the post-VAE latent
        H/W (= raw_h // vae_scale_factor) and `sample_frames` against
        `(latent_frames - 1) * temporal_compression_ratio + 1`.
        """
        patch_embed = self.transformer.patch_embed
        if not hasattr(patch_embed, "_get_positional_embeddings"):
            return
        latent_h = self.height // self.vae_scale_factor
        latent_w = self.width // self.vae_scale_factor
        latent_frames = (self.num_frames - 1) // self.vae_scale_factor_temporal + 1
        pre_time_compression_frames = (
            latent_frames - 1
        ) * patch_embed.temporal_compression_ratio + 1
        joint = patch_embed._get_positional_embeddings(
            latent_h, latent_w, pre_time_compression_frames, device=torch.device("cpu")
        )
        patch_embed.pos_embedding = joint.to(self.device, dtype=self.weight_dtype)
        patch_embed.sample_height = latent_h
        patch_embed.sample_width = latent_w
        patch_embed.sample_frames = pre_time_compression_frames

    def encode_prompts(self, prompts):
        criteria = torch.nn.MSELoss()
        cache = PromptEmbedsCache()
        prompt_pairs: list[PromptEmbedsPair] = []

        with torch.no_grad():
            for settings in prompts:
                for prompt in [settings.neutral, settings.positive, settings.negative]:
                    if cache[prompt] is None:
                        text_tokens = self.tokenizer(
                            [prompt],
                            padding="max_length",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt",
                        ).input_ids
                        text_embeddings = self.text_encoder(
                            text_tokens.to(self.text_encoder.device)
                        )[0]

                        uncond_input = self.tokenizer(
                            [""],
                            padding="max_length",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt",
                        ).input_ids
                        uncond_embeddings = self.text_encoder(
                            uncond_input.to(self.text_encoder.device)
                        )[0]

                        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                        cache[prompt] = text_embeddings

                prompt_pairs.append(
                    PromptEmbedsPair(
                        criteria,
                        cache[settings.neutral],
                        cache[settings.positive],
                        cache[settings.negative],
                        settings,
                    )
                )
        return prompt_pairs

    def get_initial_latents(self, batch_size=1, seed=42):
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(
            (
                batch_size,
                (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
                self.transformer_in_channels,
                self.height // self.vae_scale_factor,
                self.width // self.vae_scale_factor,
            ),
            generator=generator,
            device="cpu",
        )
        latents = noise * self.noise_scheduler.init_noise_sigma
        return latents

    def delete_and_flush(self):
        del self.tokenizer, self.text_encoder
        flush()

    def diffusion(
        self,
        latents,
        text_embeddings,
        start_timesteps=0,
        total_timesteps=50,
        guidance_scale=9.0,
        do_classifier_free_guidance=True,
        use_dynamic_cfg=True,
        num_inference_steps=50,
        process_str="Processing Diffusion",
        attention_kwargs=None,
        scheduler=None,
        normalize_noise=False,
        generator=None,
    ):
        if scheduler is None:
            scheduler = self.noise_scheduler

        old_pred_original_sample = None
        for i, t in enumerate(
            tqdm(
                scheduler.timesteps[start_timesteps:total_timesteps],
                desc=process_str,
                leave=False,
                dynamic_ncols=True,
            )
        ):
            noise_pred = self.predict_noise(
                t,
                latents,
                text_embeddings,
                guidance_scale,
                do_classifier_free_guidance,
                use_dynamic_cfg,
                attention_kwargs=attention_kwargs,
                scheduler=scheduler,
            )
            extra_step_kwargs = {"eta": 0.0, "generator": None}

            if not isinstance(scheduler, CogVideoXDPMScheduler):
                latents = scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
            else:
                latents, old_pred_original_sample = scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    (scheduler.timesteps[start_timesteps:total_timesteps])[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(text_embeddings.dtype)

        return latents, generator

    def predict_noise(
        self,
        t,
        latents,
        text_embeddings,
        guidance_scale,
        do_classifier_free_guidance=True,
        use_dynamic_cfg=True,
        attention_kwargs=None,
        num_inference_steps=50,
        scheduler=None,
    ):
        if scheduler is None:
            scheduler = self.noise_scheduler

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        timestep = t.expand(latent_model_input.shape[0])

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(self.weight_dtype),
            encoder_hidden_states=text_embeddings,
            timestep=timestep,
            image_rotary_emb=None,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()

        if use_dynamic_cfg:
            _guidance_scale = 1 + guidance_scale * (
                (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
            )
            guidance_scale = _guidance_scale

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if do_classifier_free_guidance:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise_pred

        return noise_pred_text

    def decode_latents(self, latents):
        vae = self.vae.to(self.device)
        latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        latents = (1 / vae.config.scaling_factor) * latents
        video = vae.decode(latents).sample

        outputs = []
        for batch_idx in range(video.shape[0]):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)
            batch_output = torch.stack(
                [(batch_vid[i] / 2 + 0.5).clamp(0, 1) for i in range(batch_vid.shape[0])]
            )
            batch_output = batch_output.cpu().permute(0, 2, 3, 1).float().numpy()
            outputs.append(batch_output)

        return np.stack(outputs)

    def log_and_evaluate(
        self,
        mapped_latents,
        prompts,
        logger,
        tag,
        t=None,
        experiment="Misc",
        compute_metrics_results=True,
        prompt_list=None,
        save_path="",
        seed=None,
    ):
        """Decode latents to videos, save to disk, and (optionally) compute alignment metrics."""

        prompts_str = [
            f"neutral: {prompts.neutral}",
            f"positive: {prompts.positive}",
            f"negative: {prompts.negative}",
        ]

        for key in mapped_latents.keys():
            mapped_latents[key] = mapped_latents[key].to(self.weight_dtype)

        if not compute_metrics_results:
            print("Plotting gifs without metrics...")
            plot_latents(mapped_latents, prompts_str, self, logger, tag, t, experiment, metric_scores_df=None)
            export_video(self, mapped_latents, experiment, tag, fps=12, save_path=save_path)
            return None, None, None, None, None, None

        print("Exporting videos...")
        vid_paths = export_video(self, mapped_latents, experiment, tag, fps=12, save_path=save_path)

        print("Computing metrics...")
        try:
            pos_results, neg_results = compute_metrics(vid_paths, prompts, logger, tag, save_path=save_path)
        except Exception as e:
            print(f"Skipping video metrics — evaluator setup failed: {e}")
            return None, None, None, None, None, None

        lpips_vector = {r["scale"]: r["lpips"]["score"][0] for r in pos_results}

        clip_text_alignment_positive = {r["scale"]: r["clip_text_alignment"]["score"][0] for r in pos_results}
        clip_text_alignment_negative = {r["scale"]: r["clip_text_alignment"]["score"][0] for r in neg_results}
        viclip_text_alignment_positive = {r["scale"]: r["viclip_text_alignment"]["score"][0] for r in pos_results}
        viclip_text_alignment_negative = {r["scale"]: r["viclip_text_alignment"]["score"][0] for r in neg_results}

        derived_metrics = compute_overall_video_scores(
            list(mapped_latents.keys()),
            clip_text_alignment_positive,
            viclip_text_alignment_positive,
            clip_text_alignment_negative,
            viclip_text_alignment_negative,
            lpips_vector,
            motion_scores=None,
            tag=tag,
            logger=logger,
        )

        contextual_clip_text_alignment = []
        contextual_viclip_text_alignment = []
        for i in mapped_latents.keys():
            if i >= 0:
                contextual_clip_text_alignment.append(clip_text_alignment_positive[i])
                contextual_viclip_text_alignment.append(viclip_text_alignment_positive[i])
            else:
                contextual_clip_text_alignment.append(clip_text_alignment_negative[i])
                contextual_viclip_text_alignment.append(viclip_text_alignment_negative[i])

        df = pd.DataFrame.from_dict(derived_metrics, orient="index").T
        print(df)

        print("Plotting gifs with metrics...")
        plot_latents(mapped_latents, prompts_str, self, logger, tag, t, experiment, metric_scores_df=df)

        contextual_clip = {"static": contextual_clip_text_alignment, "dynamic": contextual_viclip_text_alignment}
        positive_clip = {"static": clip_text_alignment_positive, "dynamic": viclip_text_alignment_positive}
        negative_clip = {"static": clip_text_alignment_negative, "dynamic": viclip_text_alignment_negative}
        lpips = {"static": lpips_vector}

        return None, contextual_clip, positive_clip, negative_clip, lpips, None
