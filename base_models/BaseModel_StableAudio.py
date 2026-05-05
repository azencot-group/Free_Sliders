import math
import os
from typing import Optional

import torch
from torch import nn
import torchaudio
from einops import rearrange
from tqdm import tqdm

from utils.prompt_utils import PromptEmbedsCache, PromptEmbedsPair
from utils.utils import flush, get_device

from .stable_audio_tools.models.pretrained import get_pretrained_model
from .stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


class VDenoiser(nn.Module):
    """A v-diffusion-pytorch model wrapper for k-diffusion-style sampling."""

    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = 1.0

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigma):
        return sigma.atan() / math.pi * 2

    def t_to_sigma(self, t):
        return (t * math.pi / 2).tan()

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class Heun_Scheduler:
    """Polyexponential-noise Heun-step scheduler used by Stable Audio Open."""

    def __init__(self, sigma_min=0.3, sigma_max=500, steps=200, rho=3, s_churn=0, s_noise=1, device="cpu"):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigmas = get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        self.s_churn = s_churn
        self.s_noise = s_noise
        self.timesteps = torch.arange(len(self.sigmas), device=device)
        self.init_noise_sigma = sigma_max

    def scale_model_input(self, latents, timestep):
        return latents

    def denoise(self, x, timestep, predict_noise, no_to_d=False):
        gamma = min(self.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1)
        sigma_hat = self.sigmas[timestep] * (gamma + 1)
        s_in = x.new_ones([x.shape[0]])
        noise_pred = predict_noise(x, sigma_hat * s_in)
        if no_to_d:
            return noise_pred
        d = to_d(x, sigma_hat, noise_pred)
        dt = self.sigmas[timestep + 1] - sigma_hat
        return d * dt

    def step(self, modified_noise, t, x, neu_args, pos_args, neg_args, denoiser, scale):
        gamma = min(self.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1)
        sigma_hat = self.sigmas[t] * (gamma + 1)
        s_in = x.new_ones([x.shape[0]])
        dt = self.sigmas[t + 1] - sigma_hat
        x_2 = x + modified_noise
        if t + 1 < len(self.sigmas) - 1:
            denoised_2_neu = denoiser(x_2, self.sigmas[t + 1] * s_in, **neu_args)
            denoised_2_neg = denoiser(x_2, self.sigmas[t + 1] * s_in, **neg_args)
            denoised_2_pos = denoiser(x_2, self.sigmas[t + 1] * s_in, **pos_args)
            d_2_neu = to_d(x_2, self.sigmas[t + 1], denoised_2_neu) / 2
            d_2_neg = to_d(x_2, self.sigmas[t + 1], denoised_2_neg) / 2
            d_2_pos = to_d(x_2, self.sigmas[t + 1], denoised_2_pos) / 2
            modified_noise_2 = modified_noise / 2 + (d_2_neu * dt + (d_2_pos - d_2_neg) * dt * scale)
            x = x + modified_noise_2
        else:
            x = x_2
        return x

    def full_step(self, x, timestep, predict_noise):
        gamma = min(self.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1)
        sigma = self.sigmas[timestep]
        sigma_hat = sigma * (gamma + 1)
        s_in = x.new_ones([x.shape[0]])
        noise_pred = predict_noise(x, sigma_hat * s_in)
        timestep, x, x_2, d = self._first_step(noise_pred, timestep, x, sigma, gamma)
        if timestep + 1 < len(self.sigmas) - 1:
            noise_pred_2 = predict_noise(x_2, self.sigmas[timestep + 1] * s_in)
            x = self._second_step(noise_pred_2, timestep, x, x_2, d, sigma_hat)
        else:
            x = x_2
        return x

    def _first_step(self, noise_pred, timestep, x, sigma, gamma):
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        d = to_d(x, sigma_hat, noise_pred)
        dt = self.sigmas[timestep + 1] - sigma_hat
        x_2 = x + d * dt
        return timestep, x, x_2, d

    def _second_step(self, noise_pred_2, timestep, x, x_2, d, sigma_hat):
        dt = self.sigmas[timestep + 1] - sigma_hat
        d_2 = to_d(x_2, self.sigmas[timestep + 1], noise_pred_2)
        d_prime = (d + d_2) / 2
        x = x + d_prime * dt
        return x

    def set_timesteps(self, num_inference_steps, device="cpu"):
        self.timesteps = torch.arange(len(self.sigmas), device=device)
        return self.timesteps


class BaseModel:
    """Stable Audio Open wrapper exposing the same surface area as BaseModel_StableDiffusion."""

    def __init__(self, config, run_id=1):
        super().__init__()

        self.run_id = run_id
        self.batch_size = 1

        audio_cfg = getattr(config, "audio", None)
        self.model_path = (
            audio_cfg.pretrained_model if audio_cfg is not None else "stabilityai/stable-audio-open-1.0"
        )
        self.seconds_total = audio_cfg.seconds_total if audio_cfg is not None else 10
        self.max_steps = audio_cfg.max_steps if audio_cfg is not None else 36

        self.device = get_device()
        # MPS doesn't support fp16 well in this stack; default to fp32 there.
        self.weight_dtype = torch.float32

        self.model, self.model_config = get_pretrained_model(self.model_path)
        self.model: ConditionedDiffusionModelWrapper = self.model.to(self.device, dtype=self.weight_dtype)
        self.model.eval()
        self.denoise_fn = VDenoiser(self.model.model)

        self.noise_scheduler = Heun_Scheduler(steps=self.max_steps, device=self.device)
        self.sample_rate = self.model_config["sample_rate"]
        self.sample_size = self.model_config["sample_size"]
        if self.model.pretransform is not None:
            self.sample_size = self.sample_size // self.model.pretransform.downsampling_ratio

        self.vid_output_path = config.save.output_path

        # CLAP / LPAPS evaluators are optional and lazily loaded; missing checkpoints
        # should not block inference.
        self.clap_model = None
        self.lpaps = None

    def encode_prompts(self, prompts):
        criteria = torch.nn.MSELoss()
        cache = PromptEmbedsCache()
        prompt_pairs: list[PromptEmbedsPair] = []

        def encode_prompt(prompt: str):
            if cache[prompt] is None:
                conditioning = [{
                    "prompt": prompt,
                    "seconds_start": 0,
                    "seconds_total": self.seconds_total,
                }]
                conditioning_tensors = self.model.conditioner(conditioning, self.device)
                conditioning_inputs = self.model.get_conditioning_inputs(conditioning_tensors)
                conditioning_inputs = {
                    k: (v.type(self.weight_dtype) if v is not None else v)
                    for k, v in conditioning_inputs.items()
                }
                conditioning_inputs["cfg_scale"] = 6
                conditioning_inputs["batch_cfg"] = True
                conditioning_inputs["rescale_cfg"] = True
                cache[prompt] = conditioning_inputs
            return cache[prompt]

        with torch.no_grad():
            for settings in prompts:
                neutral = encode_prompt(settings.neutral)
                positive = encode_prompt(settings.positive)
                negative = encode_prompt(settings.negative)
                prompt_pairs.append(
                    PromptEmbedsPair(criteria, neutral, positive, negative, settings)
                )

        return prompt_pairs

    def get_initial_latents(self, batch_size=1, seed=42):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(
            (batch_size, self.model.io_channels, self.sample_size),
            generator=generator,
        )

    def delete_and_flush(self):
        # Conditioner already produced the cached embeddings; drop it to save memory.
        if hasattr(self.model, "conditioner"):
            try:
                del self.model.conditioner
            except Exception:
                pass
        flush()

    def diffusion(
        self,
        latents,
        conditioning,
        start_timesteps=0,
        total_timesteps=-1,
        guidance_scale=9.0,
        process_str="Processing Diffusion",
        scheduler=None,
        attention_kwargs=None,
        normalize_noise=False,
        generator=None,
    ):
        scheduler = scheduler if scheduler is not None else self.noise_scheduler
        if not isinstance(scheduler, Heun_Scheduler):
            raise ValueError("Audio scheduler must be Heun_Scheduler")

        cond = dict(conditioning)
        cond["cfg_scale"] = guidance_scale

        def predict_noise(x, sigma):
            return self.denoise_fn(x, sigma, **cond)

        latents = latents.to(self.device, dtype=self.weight_dtype)
        steps = list(scheduler.timesteps[start_timesteps:total_timesteps])
        for timestep in tqdm(steps, desc=process_str):
            latents = scheduler.full_step(latents, timestep, predict_noise)
        return latents, generator

    def predict_noise(
        self,
        t,
        latents,
        conditioning,
        guidance_scale,
        do_classifier_free_guidance=False,
        scheduler=None,
        no_to_d=False,
        **kwargs,
    ):
        cond = dict(conditioning)
        cond["cfg_scale"] = guidance_scale

        def predict_noise_fn(x, sigma):
            return self.denoise_fn(x, sigma, **cond)

        return self.noise_scheduler.denoise(latents, t, predict_noise_fn, no_to_d=no_to_d)

    def step(
        self,
        modified_noise,
        t,
        latents,
        neutral_conditioning,
        positive_conditioning=None,
        negative_conditioning=None,
        scale=None,
        guidance_scale=None,
    ):
        """Heun-step that bakes in the FreeSliders noise mix; returns next latents."""
        neu = dict(neutral_conditioning)
        if guidance_scale is not None:
            neu["cfg_scale"] = guidance_scale

        if positive_conditioning is None or negative_conditioning is None:
            def predict_noise_fn(x, sigma):
                return self.denoise_fn(x, sigma, **neu)
            return self.noise_scheduler.full_step(latents, t, predict_noise_fn)

        pos = dict(positive_conditioning)
        neg = dict(negative_conditioning)
        if guidance_scale is not None:
            pos["cfg_scale"] = 1
            neg["cfg_scale"] = 1
        return self.noise_scheduler.step(
            modified_noise, t, latents, neu, pos, neg, self.denoise_fn, scale
        )

    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents into a waveform tensor, channels-first, on CPU."""
        latents = latents.to(self.device, dtype=self.weight_dtype)
        wave = self.model.pretransform.decode(latents)
        samples = self.seconds_total * self.sample_rate
        if wave.dim() == 3:  # [batch, channels, samples]
            wave = wave[:, :, :samples]
        elif wave.dim() == 2:
            wave = wave[:, :samples]
        wave = rearrange(wave, "b d n -> d (b n)")
        return wave.to(torch.float32).cpu()

    def save_audio(self, waveform, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchaudio.save(path, waveform, self.sample_rate)
        return path

    def _ensure_evaluators(self, clap_checkpoint_path: Optional[str] = None,
                          clap_ckpt_name: str = "music_speech_audioset_epoch_15_esc_89.98.pt",
                          model_arch: str = "HTSAT-base"):
        """Lazily load CLAP + LPAPS. Skips silently if checkpoints aren't available."""
        if self.clap_model is not None and self.lpaps is not None:
            return True
        if clap_checkpoint_path is None:
            return False
        ckpt = os.path.join(clap_checkpoint_path, clap_ckpt_name)
        if not os.path.exists(ckpt):
            return False
        from .lpaps.lpaps import LPAPS
        from .lpaps.meta_clap_consistency import CLAPTextConsistencyMetric

        self.clap_model = CLAPTextConsistencyMetric(
            ckpt, model_arch=model_arch, enable_fusion=False
        ).to(self.device)
        self.clap_model.eval()
        self.lpaps = LPAPS(
            net="clap",
            device=self.device,
            net_kwargs={"model_arch": model_arch, "enable_fusion": False},
            checkpoint_path=clap_checkpoint_path,
        )
        return True

    @torch.no_grad()
    def calculate_clap(self, audio_decoded, prompts):
        from evaluation.audio import calc_clap_win
        method, overlap, win_length = "mean", 0.1, 10
        contextual, positive, negative, neutral = {}, {}, {}, {}
        for scale, (audio, sr) in audio_decoded.items():
            ref = prompts.negative if scale < 0 else (prompts.positive if scale > 0 else prompts.neutral)
            contextual[scale] = calc_clap_win(self.clap_model, audio, sr, ref, win_length, method, overlap, self.device) * 100
            positive[scale] = calc_clap_win(self.clap_model, audio, sr, prompts.positive, win_length, method, overlap, self.device) * 100
            negative[scale] = calc_clap_win(self.clap_model, audio, sr, prompts.negative, win_length, method, overlap, self.device) * 100
            neutral[scale] = calc_clap_win(self.clap_model, audio, sr, prompts.neutral, win_length, method, overlap, self.device) * 100
        return contextual, positive, negative, neutral

    @torch.no_grad()
    def calculate_lpaps(self, audio_decoded):
        from evaluation.audio import calc_lpaps_win
        method, overlap, win_length = "mean", 0.1, 10
        scores = {}
        original, sr = audio_decoded[0]
        for scale, (edited, sr_e) in audio_decoded.items():
            scores[scale] = calc_lpaps_win(self.lpaps, original, edited, sr, sr_e, win_length, method, overlap, self.device)
        return scores

    def log_and_evaluate(
        self,
        audio_outputs,
        prompts,
        logger,
        tag,
        t=None,
        experiment="Misc",
        compute_metrics_results=True,
        prompt_list=None,
        save_path: Optional[str] = None,
        seed=None,
        clap_checkpoint_path: Optional[str] = None,
    ):
        """Decode latents, save .wav files, and (if evaluators are available) score CLAP/LPAPS."""
        output_dir = os.path.join(
            self.vid_output_path,
            experiment,
            tag,
            prompts.neutral.replace(" ", "_") if prompts else "unknown_prompt",
        )
        if seed is not None:
            output_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)

        audio_decoded = {}
        waveform_paths = {}
        for scale, latents in audio_outputs.items():
            wave = self.decode_latents(latents)
            file_path = os.path.join(output_dir, f"audio_output_scale_{scale}.wav")
            self.save_audio(wave, file_path)
            audio_decoded[scale] = torchaudio.load(file_path)
            waveform_paths[scale] = file_path
            if logger is not None:
                logger.log_audio(f"{tag}/audio_scale_{scale}", file_path)

        contextual_clap = positive_clap = negative_clap = neutral_clap = lpaps_score = None
        if compute_metrics_results and self._ensure_evaluators(clap_checkpoint_path):
            contextual_clap, positive_clap, negative_clap, neutral_clap = self.calculate_clap(audio_decoded, prompts)
            lpaps_score = self.calculate_lpaps(audio_decoded)
            if logger is not None:
                logger.log(f"{tag}/contextual_clap", contextual_clap)
                logger.log(f"{tag}/lpaps", lpaps_score)

        return audio_decoded, contextual_clap, positive_clap, negative_clap, lpaps_score, waveform_paths
