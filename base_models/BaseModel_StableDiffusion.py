import neptune
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from utils.prompt_utils import PromptEmbedsCache, PromptEmbedsPair
from utils.utils import flush, image_sliders, display_one_slider_and_overall_scores
from evaluation.metrics import calculate_lpips, calculate_clip, compute_overall_img_scores
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from PIL import Image
import os

class BaseModel:
    def __init__(self, config, run_id=1):
        super().__init__()

        self.run_id = run_id
        self.batch_size = 1

        # Parse weight precision
        self.weight_dtype = torch.float32
        self.device = torch.device("cuda")

        # Load the pretrained model
        self.model_path = config.pretrained_model.name_or_path

        model_path = self.model_path
        self.revision = None

        # Load scheduler, tokenizer and models.
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer",
                                                  revision=self.revision)

        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                                     revision=self.revision)
        text_encoder.requires_grad_(False)
        text_encoder.to(self.device, dtype=self.weight_dtype)
        self.text_encoder = text_encoder

        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=self.revision)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=self.weight_dtype)
        self.vae = vae

        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=self.revision)
        unet.requires_grad_(False)
        unet.to(self.device, dtype=self.weight_dtype)
        self.network = unet

        self.noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               num_train_timesteps=1000)

        # Store additional attributes
        self.height = config.pretrained_model.height
        self.width = config.pretrained_model.width

        self.vid_output_path = config.save.output_path

        self.max_steps = config.pretrained_model.max_steps

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

                prompt_pairs.append(
                    PromptEmbedsPair(
                        criteria,
                        neutral,
                        positive,
                        negative,
                        settings,
                    )
                )

        return prompt_pairs

    def get_initial_latents(self, batch_size=1, seed=42):
        generator = torch.Generator().manual_seed(seed)
        rand_latents = torch.randn(
            (batch_size, self.network.config.in_channels, self.height // 8, self.width // 8),
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

        if scheduler is None:
            scheduler = self.noise_scheduler

        for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps], desc=process_str):

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

            # predict the noise residual
            noise_pred = self.network(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

        # return latents_steps
        return latents, None

    def predict_noise(self, t, latents, text_embeddings, guidance_scale,
                      do_classifier_free_guidance=False, use_dynamic_cfg=None, attention_kwargs=None,
                      scheduler=None, no_grad = True):

        if scheduler is None:
            scheduler = self.noise_scheduler

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        if not do_classifier_free_guidance and len(text_embeddings) > 1:
            text_embeddings = text_embeddings[:1]

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        if no_grad:
            with torch.no_grad():
                noise_pred = self.network(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
        else:
            noise_pred = self.network(
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

        with torch.no_grad():
            noise_pred = self.network(
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

    def log_and_evaluate(self, slider_imgs, prompts, logger, tag, t=None, experiment="Misc", compute_metrics_results=True,
                         prompt_list=None, save_path="", seed=0):
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
        output_dir = f"outputs/{tag}/seed_{seed}"
        os.makedirs(output_dir, exist_ok=True)
        for scale, image in img_slider.items():
            image_path = os.path.join(output_dir, f"scale_{scale}.png")
            image.save(image_path)

        images = image_sliders(img_slider, experiment, run_id=self.run_id)

        lpips = calculate_lpips(img_slider)
        contextual_clip, positive_clip, negative_clip = calculate_clip(img_slider, prompts)

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
