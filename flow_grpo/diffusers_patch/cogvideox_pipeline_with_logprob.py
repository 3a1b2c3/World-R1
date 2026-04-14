"""CogVideoX rollout helpers adapted for World-R1 training.

Source note:
- The pipeline body is adapted from Diffusers' CogVideoX pipeline.
- The log-prob path follows the DDPO-style Gaussian transition used in Flow-GRPO.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_cogvideox import (
    CogVideoXPipelineOutput,
    retrieve_timesteps,
)
from diffusers.schedulers.scheduling_ddim_cogvideox import CogVideoXDDIMScheduler
from diffusers.utils.torch_utils import randn_tensor


def _gaussian_log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    while std.ndim < sample.ndim:
        std = std.unsqueeze(-1)

    safe_std = std.clamp_min(1e-8)
    log_prob = (
        -((sample.detach() - mean) ** 2) / (2 * (safe_std**2))
        - torch.log(safe_std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device, dtype=sample.dtype)))
    )
    zero_var_mask = std <= 1e-8
    if zero_var_mask.any():
        log_prob = torch.where(zero_var_mask, torch.zeros_like(log_prob), log_prob)
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def get_cogvideox_additional_frames(
    num_frames: int,
    vae_scale_factor_temporal: int,
    patch_size_t: Optional[int],
) -> int:
    latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    if patch_size_t is None or patch_size_t <= 1 or latent_frames % patch_size_t == 0:
        return 0
    return patch_size_t - latent_frames % patch_size_t


def get_cogvideox_padded_num_frames(
    num_frames: int,
    vae_scale_factor_temporal: int,
    patch_size_t: Optional[int],
) -> Tuple[int, int]:
    additional_frames = get_cogvideox_additional_frames(
        num_frames=num_frames,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        patch_size_t=patch_size_t,
    )
    return num_frames + additional_frames * vae_scale_factor_temporal, additional_frames


def prepare_cogvideox_latents(
    latents: torch.Tensor,
    in_channels: int,
    patch_size_t: Optional[int],
) -> Tuple[torch.Tensor, int]:
    """Convert latents to CogVideoX layout and pad latent time if needed.

    Accepts either `[B, C, T, H, W]` or `[B, T, C, H, W]` and returns `[B, T, C, H, W]`.
    For CogVideoX 1.5, the latent time dimension must be divisible by `patch_size_t`.
    We prepend duplicated first-frame latents so the model can later discard them consistently.
    """

    if latents.ndim != 5:
        raise ValueError(f"`latents` must be 5D, got shape {latents.shape}.")

    if latents.shape[2] == in_channels:
        prepared = latents
    elif latents.shape[1] == in_channels:
        prepared = latents.permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(
            f"Cannot infer CogVideoX latent layout from shape {latents.shape}; expected channel dim {in_channels}."
        )

    additional_frames = 0
    if patch_size_t is not None and patch_size_t > 1:
        remainder = prepared.shape[1] % patch_size_t
        if remainder != 0:
            additional_frames = patch_size_t - remainder
            pad = prepared[:, :1].repeat(1, additional_frames, 1, 1, 1)
            prepared = torch.cat([pad, prepared], dim=1)

    return prepared, additional_frames


def decode_cogvideox_latents(self, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.permute(0, 2, 1, 3, 4).contiguous()
    scale = latents.new_tensor(float(self.vae_scaling_factor_image))
    latents = (latents / scale).to(dtype=self.vae.dtype)
    return self.vae.decode(latents).sample


def ddim_step_with_logprob(
    self: CogVideoXDDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[int, torch.IntTensor, torch.LongTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
):
    """DDIM-like Gaussian step with log-prob for CogVideoX.

    Diffusers' public CogVideoX DDIM scheduler is deterministic. For RL we use the standard
    DDPO-style Gaussian relaxation around the DDIM mean, controlled by `noise_level`.
    """

    if self.num_inference_steps is None:
        raise ValueError("`set_timesteps()` must be called before `ddim_step_with_logprob()`.")

    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    if not torch.is_tensor(timestep):
        timestep = torch.as_tensor([timestep], device=sample.device, dtype=torch.long)
    else:
        timestep = timestep.to(device=sample.device, dtype=torch.long).flatten()

    if timestep.numel() == 1 and sample.shape[0] > 1:
        timestep = timestep.repeat(sample.shape[0])

    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=torch.float32)
    alpha_prod_t = alphas_cumprod[timestep].view(-1, 1, 1, 1, 1)
    alpha_prod_t_prev = torch.where(
        prev_timestep.view(-1, 1, 1, 1, 1) >= 0,
        alphas_cumprod[prev_timestep.clamp_min(0)].view(-1, 1, 1, 1, 1),
        self.final_alpha_cumprod.to(device=sample.device, dtype=torch.float32).view(1, 1, 1, 1, 1),
    )
    beta_prod_t = 1 - alpha_prod_t

    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt().clamp_min(1e-8)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
    else:
        raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type!r}")

    variances = []
    for t_item, prev_t_item in zip(timestep.tolist(), prev_timestep.tolist()):
        variances.append(self._get_variance(int(t_item), int(prev_t_item)))
    variance = torch.stack(variances).to(device=sample.device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
    std_dev_t = float(noise_level) * variance.sqrt()

    pred_sample_direction = torch.clamp(1 - alpha_prod_t_prev - std_dev_t**2, min=0.0).sqrt() * pred_epsilon
    prev_sample_mean = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

    if prev_sample is None:
        if determistic or float(noise_level) <= 0:
            prev_sample = prev_sample_mean
        else:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise
    elif determistic:
        prev_sample = prev_sample_mean

    log_prob = _gaussian_log_prob(prev_sample.float(), prev_sample_mean.float(), std_dev_t.float())
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


@torch.no_grad()
def cogvideox_pipeline_with_logprob(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6.0,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pt",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
    noise_level: float = 0.7,
    determistic: bool = False,
    kl_reward: float = 0.0,
):
    del kl_reward  # kept for call-site compatibility with the Wan path

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames
    num_videos_per_prompt = 1

    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds,
        negative_prompt_embeds,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    transformer_dtype = self.transformer.dtype
    prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=transformer_dtype)

    prompt_embeds_input = (
        torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) if do_classifier_free_guidance else prompt_embeds
    )

    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    additional_frames = 0
    patch_size_t = self.transformer.config.patch_size_t
    latent_channels = self.transformer.config.in_channels

    if latents is None:
        num_frames, additional_frames = get_cogvideox_padded_num_frames(
            num_frames=num_frames,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            patch_size_t=patch_size_t,
        )
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    else:
        latents, additional_frames = prepare_cogvideox_latents(
            latents,
            in_channels=latent_channels,
            patch_size_t=patch_size_t,
        )
        latents = latents.to(device=device, dtype=prompt_embeds.dtype)

    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    all_latents = [latents]
    all_log_probs = []
    all_kl = []

    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds_input,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            if use_dynamic_cfg:
                self._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents, log_prob, prev_latents_mean, std_dev_t = ddim_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0),
                latents.float(),
                noise_level=noise_level,
                generator=generator,
                determistic=determistic,
            )

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for key in callback_on_step_end_tensor_inputs:
                    callback_kwargs[key] = locals()[key]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds_input = callback_outputs.pop("prompt_embeds", prompt_embeds_input)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                log_prob = _gaussian_log_prob(latents.float(), prev_latents_mean.float(), std_dev_t.float())

            latents = latents.to(dtype=prompt_embeds.dtype)
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_kl.append(torch.zeros(latents.shape[0], device=latents.device, dtype=latents.dtype))

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    self._current_timestep = None

    if output_type != "latent":
        if additional_frames > 0:
            latents = latents[:, additional_frames:]
        if latents.device.type == "cuda":
            torch.cuda.empty_cache()
        video = decode_cogvideox_latents(self, latents)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    self.maybe_free_model_hooks()

    if not return_dict:
        return video, all_latents, all_log_probs, all_kl

    return CogVideoXPipelineOutput(frames=video), all_latents, all_log_probs, all_kl
