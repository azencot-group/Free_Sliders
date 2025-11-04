import neptune
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from utils.prompt_utils import PromptEmbedsCache, PromptEmbedsPair
from utils.utils import plot_latents, export_video, flush, compute_metrics, image_sliders, plot_scores, \
    display_one_slider_and_graphs, display_one_slider_and_overall_scores
from evaluation.new_slider_metrics import calculate_lpips, calculate_clip, compute_overall_img_scores
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from PIL import Image
from methods.lora import LoRANetwork, UNET_TARGET_REPLACE_MODULE_CONV, DEFAULT_TARGET_REPLACE
import os

class BaseModel:
    def __init__(self, config, cache_dir='/cs/azencot_fsas/text2video', run_id=1):
        super().__init__()

        self.run_id = run_id
        self.batch_size = 1
        # Parse weight precision
        self.weight_dtype = torch.float32   # @TODO take from config
        self.device = torch.device("cuda")

        # Load the CogVideoX model
        # self.model_path = "CompVis/stable-diffusion-v1-4"   # @TODO take from config?
        self.model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        model_path = self.model_path
        self.revision = None
        revision = self.revision

        # Load scheduler, tokenizer and models.
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer",
                                                  revision=revision)

        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                                     revision=revision)
        text_encoder.requires_grad_(False)
        text_encoder.to(self.device, dtype=self.weight_dtype)

        self.text_encoder = text_encoder

        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=revision)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=self.weight_dtype)

        self.vae = vae

        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=revision)
        unet.requires_grad_(False)
        unet.to(self.device, dtype=self.weight_dtype)

        self.transformer = unet # TODO change later

        self.noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               num_train_timesteps=1000)

        # Store additional attributes
        self.height = 512   # @TODO generalize
        self.width = 512    # @TODO generalize

        self.vid_output_path = config.save.generated_videos_path

        self.max_steps = config.train.max_denoising_steps

        self.network = LoRANetwork(
            self.transformer,
        ).to(self.device, dtype=self.weight_dtype)

    def encode_prompts(self, prompts):
        criteria = torch.nn.MSELoss()
        cache = PromptEmbedsCache()
        prompt_pairs: list[PromptEmbedsPair] = []

        def encode_neutral_prompt(prompt: str):
            if cache[prompt] is None:
                text_input = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

                max_length = text_input.input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    [""], # * prompt_batch_size,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                cache[prompt] = text_embeddings
            return cache[prompt]

        def encode_directed_prompt(prompt: str):
            if cache[prompt] is None:
                text_input = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

                cache[prompt] = text_embeddings
            return cache[prompt]


        with torch.no_grad():
            for settings in prompts:
                # Encode main prompts
                neutral = encode_neutral_prompt(settings.neutral)
                positive = encode_directed_prompt(settings.positive)
                negative = encode_directed_prompt(settings.negative)

                variations = []
                if hasattr(settings, "variations"):
                    for variation in settings.variations:
                        # neutral_var = encode_neutral_prompt(variation["neutral"])
                        positive_var = encode_directed_prompt(variation["positive"])
                        negative_var = encode_directed_prompt(variation["negative"])
                        variations.append((positive_var, negative_var, variation))

                prompt_pairs.append(
                    PromptEmbedsPair(
                        criteria,
                        neutral,
                        positive,
                        negative,
                        settings,
                        variations  # new field for variation triplets
                    )
                )

        return prompt_pairs

    def get_initial_latents(self, batch_size=1, seed=42):
        generator = torch.Generator().manual_seed(seed)
        rand_latents = torch.randn(
            (batch_size, self.transformer.config.in_channels, self.height // 8, self.width // 8),
            generator=generator,
        )
        return rand_latents

    def delete_and_flush(self):
        del (
            self.tokenizer,
            self.text_encoder,
        )

        flush()

    def diffusion(self, latents, text_embeddings, start_timesteps=0, total_timesteps=50, guidance_scale=9.0,
                  process_str="Processing Diffusion", scheduler=None, attention_kwargs=None, normalize_noise=False, generator=None):

        # latents_steps = []
        if scheduler is None:
            scheduler = self.noise_scheduler

        for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps], desc=process_str):
            # predict noise
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

            # predict the noise residual
            noise_pred = self.transformer(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_target = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if not normalize_noise:
                noise_pred = guided_target
            else:
                noise_pred = guided_target / torch.norm(guided_target, p='fro').item()
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

        # return latents_steps
        return latents, None

    def predict_noise(self, t, latents, text_embeddings, guidance_scale,
                      do_classifier_free_guidance=False, use_dynamic_cfg=None, attention_kwargs=None,
                      scheduler=None, no_grad = True):

        if scheduler is None:
            scheduler = self.noise_scheduler

        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents
        if not do_classifier_free_guidance and len(text_embeddings) > 1:
            text_embeddings = text_embeddings[:1]

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with self.network:
            if no_grad:
                with torch.no_grad():
                    noise_pred = self.transformer(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                    ).sample
            else:
                noise_pred = self.transformer(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred


    def predict_noise_triplet(self, t, latents, neutral_embeddings, positive_embeddings, negative_embeddings,
                              neutral_guidance_scale, variation_guidance_scale, scheduler=None):
        """
        Batched prediction of neutral (with classifier-free guidance), positive, and negative noise
        using a SINGLE forward pass of the UNet.

        Args:
            t (torch.Tensor | int): current diffusion timestep
            latents (torch.FloatTensor): latent tensor of shape (B, C, H, W)
            neutral_embeddings (torch.FloatTensor): shape (2, E) -> [uncond, neutral]
            positive_embeddings (torch.FloatTensor): shape (E,) or (1, E)
            negative_embeddings (torch.FloatTensor): shape (E,) or (1, E)
            neutral_guidance_scale (float): guidance scale for neutral CFG branch
            variation_guidance_scale (float): guidance scale for positive / negative (if you later want CFG there)
            scheduler: optional scheduler (defaults to self.noise_scheduler)

        Returns:
            (neutral_noise_cfg, positive_noise, negative_noise)
            Each of shape (B, C, H, W)
        """
        if scheduler is None:
            scheduler = self.noise_scheduler

        if neutral_embeddings.shape[0] != 2:
            raise ValueError("neutral_embeddings must contain [uncond, neutral] stacked (batch=2).")

        # Ensure positive / negative are 2D
        if positive_embeddings.dim() == 1:
            positive_embeddings = positive_embeddings.unsqueeze(0)
        if negative_embeddings.dim() == 1:
            negative_embeddings = negative_embeddings.unsqueeze(0)

        # Build latent batch:
        #   0: uncond (neutral CFG)
        #   1: neutral cond
        #   2: positive (no CFG)
        #   3: negative (no CFG)
        latent_batch = torch.cat([latents, latents, latents, latents], dim=0)
        latent_batch = scheduler.scale_model_input(latent_batch, timestep=t)

        text_batch = torch.cat([
            neutral_embeddings[0:1],   # uncond
            neutral_embeddings[1:2],   # neutral cond
            positive_embeddings,       # positive
            negative_embeddings        # negative
        ], dim=0)

        with self.network:
            with torch.no_grad():
                noise_pred = self.transformer(
                    latent_batch,
                    t,
                    encoder_hidden_states=text_batch,
                ).sample

        # Split
        neutral_uncond, neutral_cond, positive_pred, negative_pred = noise_pred.chunk(4)

        # Neutral CFG
        neutral_cfg = neutral_uncond + neutral_guidance_scale * (neutral_cond - neutral_uncond)

        # (Optional) if later you want CFG for positive / negative directions, you can extend here
        return neutral_cfg, positive_pred, negative_pred

    def plot_clip_lpips_scores(self, scales, clip_scores, lpips_scores, logger=None, tag="misc", clip_thresh_pos=25, lpips_thresh_pos=0.1, clip_thresh_neg=None, lpips_thresh_neg=None, positive_clip=None, negative_clip=None):
        """
        Plots multiple views of CLIP and LPIPS scores across scales:
        - Line plot of CLIP and LPIPS
        - Scatter plot CLIP vs LPIPS
        - Bubble plot with scale encoded
        - Composite score: CLIP - λ * LPIPS
        - Pareto frontier (optional)

        Args:
            scales (list or np.array): The scale values used in the slider.
            clip_scores (dict or list): Dict or list of CLIP scores keyed or ordered by scale.
            lpips_scores (dict or list): Dict or list of LPIPS scores keyed or ordered by scale.
            logger: Optional logger (e.g. Neptune).
            title (str): Title for the plots.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # Convert dicts to lists aligned with scales
        if isinstance(clip_scores, dict):
            clip_scores = [clip_scores[s] for s in scales]
        if isinstance(positive_clip, dict):
            positive_clip = [positive_clip[s] for s in scales]
        if isinstance(negative_clip, dict):
            negative_clip = [negative_clip[s] for s in scales]
        if isinstance(lpips_scores, dict):
            lpips_scores = [lpips_scores[s] for s in scales]

        scales = np.array(list(scales))
        clip_scores = np.array(clip_scores)
        lpips_scores = np.array(lpips_scores)
        positive_clip = np.array(positive_clip)
        negative_clip = np.array(negative_clip)

        # Normalize the scores (min-max normalization)
        clip_scores_norm = (clip_scores - np.min(clip_scores)) / (np.max(clip_scores) - np.min(clip_scores))
        lpips_scores_norm = (lpips_scores - np.min(lpips_scores)) / (np.max(lpips_scores) - np.min(lpips_scores))

        # Compute the difference (CLIP - LPIPS)
        score_diff = clip_scores_norm - lpips_scores_norm

        # ========== Line Plot ==========
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(scales, clip_scores_norm, label="CLIP Score (normalized)", color="blue", marker='o')
        ax1.plot(scales, lpips_scores_norm, label="LPIPS Score (normalized)", color="red", marker='s')
        ax1.plot(scales, score_diff, label="CLIP - LPIPS (normalized diff)", color="green", marker='^', linestyle='--')

        ax1.set_xlabel("Scale")
        ax1.set_ylabel("Normalized Score")
        ax1.set_title("Normalized CLIP & LPIPS vs Scale")
        ax1.legend()
        ax1.grid(True)
        fig1.tight_layout()

        # Save
        fig1_path = "sd/plot_line_clip_lpips_normalized.png"
        os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
        fig1.savefig(fig1_path)

        if logger is not None:
            logger[f"scores_plots/line_plot_normalized/{tag}"].log(neptune.types.File(fig1_path))

        plt.close(fig1)

        # ========== Positive and Negative CLIP Plot ==========
        if positive_clip is not None and negative_clip is not None:
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(scales, clip_scores, label="Contextual CLIP Score", color="blue", marker='o')
            ax1.plot(scales, positive_clip, label="Positive CLIP Score", color="red", marker='o')
            ax1.plot(scales, negative_clip, label="Negative CLIP Score", color="black", marker='o')

            ax1.set_xlabel("Scale")
            ax1.set_ylabel("Score")
            ax1.set_title("Different CLIP Scores")
            ax1.legend()
            ax1.grid(True)
            fig1.tight_layout()

            # Save
            fig1_path = "sd/plot_line_clips.png"
            os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
            fig1.savefig(fig1_path)

            if logger is not None:
                logger[f"scores_plots/plot_line_clips/{tag}"].log(neptune.types.File(fig1_path))

            plt.close(fig1)

        # # ========== Bubble Plot ==========
        # fig3, ax3 = plt.subplots(figsize=(7, 6))
        # scatter = ax3.scatter(lpips_scores, clip_scores, c=scales, s=40 + 10 * np.abs(scales),
        #                       cmap='viridis', edgecolor='k', alpha=0.8)
        # cbar = plt.colorbar(scatter, ax=ax3)
        # cbar.set_label("Scale")
        # ax3.set_xlabel("LPIPS Score ↓")
        # ax3.set_ylabel("CLIP Score ↑")
        # ax3.set_title("CLIP vs LPIPS with Scale Encoding")
        # ax3.grid(True)
        # fig3.tight_layout()
        # fig3_path = "sd/plot_bubble_clip_lpips.png"
        # fig3.savefig(fig3_path)
        # if logger is not None:
        #     logger[f"scores_plots/bubble_plot/{tag}"].log(neptune.types.File(fig3_path))
        # plt.close(fig3)


        # ========== Pareto Frontier ==========
        title = "Pareto Frontier: CLIP vs LPIPS"

        # Points: (lpips, clip, scale)
        points = np.array(list(zip(lpips_scores, clip_scores, scales)))
        sorted_pts = points[np.lexsort((-clip_scores, lpips_scores))]

        # Pareto frontier
        pareto = []
        max_clip = -np.inf
        for lp, cl, sc in sorted_pts:
            if cl > max_clip:
                pareto.append((lp, cl, sc))
                max_clip = cl
        pareto = np.array(pareto)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(lpips_scores, clip_scores, c=scales, cmap="viridis", s=90, edgecolors='black',
                             label="All Scales")

        for i, scale in enumerate(scales):
            ax.annotate(
                str(scale),
                (lpips_scores[i], clip_scores[i]),
                textcoords="offset points",
                xytext=(0, 5),  # slight vertical offset
                ha='center',
                fontsize=8,
                color='black'
            )

        # Pareto frontier line
        # ax.plot(pareto[:, 0], pareto[:, 1], 'r--', lw=2, label="Pareto Frontier")
        # ax.scatter(pareto[:, 0], pareto[:, 1], color='red', s=80, marker='x', label="Pareto Optimal")


        ax.set_xlabel("LPIPS ↓", fontsize=12)
        ax.set_ylabel("CLIP ↑", fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True)
        plt.colorbar(scatter, label="Scale")
        plt.tight_layout()
        plt.show()
        fig_path = "sd/plot_composite_score.png"
        fig.savefig(fig_path)
        if logger is not None:
            logger[f"scores_plots/pareto_frontier/{tag}"].log(neptune.types.File(fig_path))
        plt.close()

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(scales, clip_scores, cmap="viridis", s=90, edgecolors='black')



    def log_and_evaluate(self, slider_imgs, prompts, logger, tag, t=None, experiment="Misc", compute_metrics_results=True,
                         prompt_list=None, explore_ranges=False, save_path="", seed=0):
        """Log and evaluate the output using Neptune and visualizations."""

        # decode latents
        img_slider = {}
        for scale in slider_imgs.keys():
            latents = slider_imgs[scale]
            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]

            img_slider[scale] = pil_images[0]

        import os
        # Add a section to save each image in `img_slider[scale]` into the directory `website_outputs`
        output_dir = f"website_outputs/{tag}/seed_{seed}"
        os.makedirs(output_dir, exist_ok=True)
        for scale, image in img_slider.items():
            image_path = os.path.join(output_dir, f"scale_{scale}.png")
            image.save(image_path)

        images = image_sliders(img_slider, experiment, run_id=self.run_id)

        lpips = calculate_lpips(img_slider)
        contextual_clip, positive_clip, negative_clip = calculate_clip(img_slider, prompts)

        # if explore_ranges:
        #     self.find_ranges(slider_imgs, contextual_clip, positive_clip, negative_clip, lpips, tag, logger, img_slider)
        overall_img_scores = compute_overall_img_scores(contextual_clip, lpips,
                                                     positive_clip,
                                                     negative_clip,
                                                     tag, logger)

        display_one_slider_and_overall_scores(images, tag, logger,
                                        overall_img_scores,
                                          clip_scores=positive_clip,
                                          lpips_scores=lpips,
                                          run_id=self.run_id)

        if logger is not None:
            logger[f"sliders_and_plots/{tag}"].log(neptune.types.File(images))

        return slider_imgs, contextual_clip, positive_clip, negative_clip, lpips, img_slider

    def save_lora_weights(self, save_path, transformer_lora_layers_to_save, weight_name):
        save_path.mkdir(parents=True, exist_ok=True)
        self.network.save_weights(
            save_path /  weight_name,
            dtype=self.weight_dtype,
        )

    def load_lora_weights(self, lora_weights_path, lora_weight, adapter_name):
        if 'full' in lora_weight:
            train_method = 'full'
        elif 'noxattn' in lora_weight:
            train_method = 'noxattn'
        else:
            train_method = 'noxattn'

        network_type = "c3lier"
        if train_method == 'xattn':
            network_type = 'lierla'

        modules = DEFAULT_TARGET_REPLACE
        if network_type == "c3lier":
            modules += UNET_TARGET_REPLACE_MODULE_CONV

        rank = 4
        alpha = 1
        if 'rank4' in lora_weight:
            rank = 4
        if 'rank8' in lora_weight:
            rank = 8
        if 'alpha1' in lora_weight:
            alpha = 1.0

        self.network = LoRANetwork(
            self.transformer,
            rank=rank,
            multiplier=1.0,
            alpha=alpha,
            train_method=train_method,
        ).to(self.device, dtype=self.weight_dtype)

        full_weight_path = os.path.join(lora_weights_path, lora_weight)
        state_dict = torch.load(full_weight_path, map_location=self.device)
        self.network.load_state_dict(state_dict)


    def set_lora(self, scale):
        self.network.set_lora_slider(scale=scale)
        return None

    def unload_lora_weights(self):
        # Completely reinitialize the unet and lora network
        unet = UNet2DConditionModel.from_pretrained(self.model_path, subfolder="unet", revision=self.revision)
        unet.requires_grad_(False)
        unet.to(self.device, dtype=self.weight_dtype)

        self.transformer = None
        torch.cuda.empty_cache()
        self.transformer = unet

        # reinitialize reference to LoRA network
        self.network = None
        torch.cuda.empty_cache()

        self.network = LoRANetwork(
            self.transformer,
        ).to(self.device, dtype=self.weight_dtype)
