"""Wan rollout helpers adapted for World-R1 training.

Source note:
- The log-prob / SDE rollout path is adapted from Flow-GRPO.
- The released file keeps only the Wan-specific path required by World-R1.
"""

import contextlib
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
import math
import numpy as np
import os
try:
    from flow_grpo.diffusers_patch.noise_visualizer import (
        noise_to_rgb,
        save_video_mp4,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def _gaussian_log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    log_prob = (
        -((sample.detach() - mean) ** 2) / (2 * (std**2))
        - torch.log(std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device, dtype=sample.dtype)))
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def _cache_context_or_null(model, name: str):
    if hasattr(model, "cache_context"):
        return model.cache_context(name)
    return contextlib.nullcontext()


def _save_latents_vis(
    latents: torch.Tensor,
    output_dir: str,
    step: int,
    name: str,
    max_frames: int = 21,
    fps: int = 12,
    upsample_size: Optional[Tuple[int, int]] = None,
    upsample_interp: str = "linear",
    channels: Tuple[int, int, int] = (0, 1, 2),
    scale: float = 6.0,
    normalize: str = "none",
    save_delta: bool = False,
):
    """
    Helper function to save latents visualization.

    Args:
        latents: Latent tensor [B, C, T, H, W] or [C, T, H, W]
        output_dir: Output directory
        step: Step number
        name: Name for the output file
        max_frames: Maximum frames to save
        fps: Video frame rate
        upsample_size: Optional (height, width) to upsample
    """
    if not VISUALIZATION_AVAILABLE:
        return

    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        if normalize == "none":
            return frame
        if normalize == "per_frame_std":
            mean = float(frame.mean())
            std = float(frame.std())
            return (frame - mean) / (std + 1e-6)
        if normalize == "global_std":
            return frame
        raise ValueError(f"Unknown `normalize` mode: {normalize!r}")

    try:
        # Handle batch dimension
        if latents.ndim == 5:
            latent = latents[0]  # Take first batch
        else:
            latent = latents

        # latent: [C, T, H, W]
        C, T, H, W = latent.shape

        if normalize == "global_std":
            global_mean = float(latent.mean().item())
            global_std = float(latent.std().item())

        # Limit number of frames
        if T > max_frames:
            indices = torch.linspace(0, T - 1, max_frames).long()
            latent = latent[:, indices]
            T = max_frames

        def _render_video(tensor_cthw: torch.Tensor, out_name: str) -> None:
            frames = []
            for t in range(tensor_cthw.shape[1]):
                # Get frame: [C, H, W]
                # NumPy doesn't support bfloat16; cast to float32 before converting.
                frame = tensor_cthw[:, t, :, :].float().cpu().numpy()
                if normalize == "global_std":
                    frame = (frame - global_mean) / (global_std + 1e-6)
                else:
                    frame = _normalize_frame(frame)

                # Convert to [H, W, C]
                frame = frame.transpose(1, 2, 0)
                # Convert to RGB
                frame_rgb = noise_to_rgb(frame, channels=list(channels), scale=scale)

                # Upsample if requested
                if upsample_size is not None:
                    try:
                        import cv2

                        interp = cv2.INTER_LINEAR
                        if upsample_interp == "nearest":
                            interp = cv2.INTER_NEAREST
                        elif upsample_interp == "linear":
                            interp = cv2.INTER_LINEAR
                        else:
                            raise ValueError(f"Unknown `upsample_interp`: {upsample_interp!r}")
                        frame_rgb = cv2.resize(
                            frame_rgb,
                            (upsample_size[1], upsample_size[0]),
                            interpolation=interp,
                        )
                    except ImportError:
                        pass

                # Convert to uint8
                frame_uint8 = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
                frames.append(frame_uint8)

            # Save video + numpy snapshot
            video_path = os.path.join(output_dir, f"{out_name}.mp4")
            save_video_mp4(frames, video_path, fps=fps)

            npy_path = os.path.join(output_dir, f"{out_name}.npy")
            np.save(npy_path, tensor_cthw.float().cpu().numpy().astype(np.float16))

        _render_video(latent, name)

        if save_delta and latent.shape[1] > 1:
            delta = latent[:, 1:] - latent[:, :-1]
            _render_video(delta, f"{name}_delta")

    except Exception as e:
        print(f"Warning: Failed to save visualization at step {step}: {e}")


def sde_step_with_logprob(
    self: UniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
    return_pixel_log_prob: bool = False,
    return_dt_and_std_dev_t: bool = False
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()
        
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]

    self.sigmas = self.sigmas.to(sample.device)
    sigma = self.sigmas[step_index].view(-1, 1, 1, 1, 1)
    sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1, 1, 1)
    sigma_max = self.sigmas[0].item()
    sigma_min = self.sigmas[-1].item()
    dt = sigma_prev - sigma

    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    scheduler_is_deterministic = not getattr(getattr(self, "config", object()), "stochastic_sampling", False)
    if scheduler_is_deterministic:
        prev_sample_mean = sample + dt * model_output
    else:
        prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (
            1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)
        ) * dt

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        if determistic or scheduler_is_deterministic:
            prev_sample = prev_sample_mean
        else:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2 + 1e-12))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt) + 1e-12)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)

def wan_pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    guidance_scale_2: Optional[float] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    determistic: bool = False,
    kl_reward: float = 0.0,
    return_pixel_log_prob: bool = False,
    use_camera_trajectory: bool = False,
    save_latents_vis: bool = True,  # Save prepared latents visualization (before inference)
    vis_output_dir: Optional[str] = None,
    vis_fps: int = 12,
    vis_max_frames: int = 16,
    vis_upsample_size: Optional[Tuple[int, int]] = None,
    vis_upsample_interp: str = "nearest",
    vis_noise_scale: float = 5.0,  # Match reference visualization: /5+.5
    vis_normalize: str = "none",  # "none" preserves temporal motion pattern; "per_frame_std" destroys it
    vis_save_delta: bool = True,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        height (`int`, defaults to `480`):
            The height in pixels of the generated image.
        width (`int`, defaults to `832`):
            The width in pixels of the generated image.
        num_frames (`int`, defaults to `81`):
            The number of frames in the generated video.
        num_inference_steps (`int`, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, defaults to `5.0`):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
            The dtype to use for the torch.amp.autocast.

    Examples:

    Returns:
        [`~WanPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
            the first element is a list with the generated images and the second element is a list of `bool`s
            indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
    """

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    if num_frames % self.vae_scale_factor_temporal != 1:
        print(
            f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    self._guidance_scale = guidance_scale
    self._guidance_scale_2 = guidance_scale_2
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = self._execution_device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = self.transformer.dtype
    transformer_2 = getattr(self, "transformer_2", None)
    if transformer_2 is not None:
        transformer_dtype = self.transformer.dtype if self.transformer is not None else transformer_2.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    print(f"num_channels_latents: {num_channels_latents}")
    if latents is not None:
        print(
            f"provided_latents: {latents.shape}, mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}"
        )

    # Respect caller-provided latents. Only generate latents when none are passed.
    if latents is None:
        if use_camera_trajectory:
            # Use camera-aware latents preparation (requires raw prompt text, not just embeddings)
            if prompt is None:
                raise ValueError(
                    "`use_camera_trajectory=True` requires `prompt` to be provided when `latents` is None."
                )
            from flow_grpo.diffusers_patch.camera_trajectory_utils import prepare_latents_with_camera

            latents = prepare_latents_with_camera(
                prompt=prompt,
                batch_size=batch_size * num_videos_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
                vae_scale_factor_temporal=self.vae_scale_factor_temporal,
                frames_per_trajectory=num_frames,
            )
        else:
            # Use default random latents
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
            )
    else:
        latents = latents.to(device=device, dtype=torch.float32)

    # Align latent temporal length to the VAE temporal compression convention.
    # This prevents silent off-by-one issues that typically show up as corruption in the last few frames.
    expected_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    if latents.ndim != 5:
        raise ValueError(f"`latents` must have shape [B, C, T, H, W], got {latents.shape}.")
    if latents.shape[2] != expected_latent_frames:
        src_t = latents.shape[2]
        dst_t = expected_latent_frames
        print(
            f"Warning: `latents` has T={src_t} but expected T={dst_t} for num_frames={num_frames} "
            f"(vae_scale_factor_temporal={self.vae_scale_factor_temporal}). Resizing in time."
        )
        if src_t > 1 and dst_t > 1:
            step = (src_t - 1) / (dst_t - 1)
        else:
            step = 0.0
        indices = [round(i * step) for i in range(dst_t)]
        idx = torch.as_tensor(indices, device=latents.device, dtype=torch.long)
        latents = latents.index_select(2, idx)
        if latents.shape[2] < dst_t:
            pad = latents[:, :, -1:, :, :].repeat(1, 1, dst_t - latents.shape[2], 1, 1)
            latents = torch.cat([latents, pad], dim=2)

    print(f"final_latents: {latents.shape}, mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}")

    all_latents = [latents]
    all_log_probs = []
    all_kl = []

    # Visualization setup - save prepared latents
    if save_latents_vis:
        if vis_output_dir is None:
            vis_output_dir = "pipeline_visualization"
        os.makedirs(vis_output_dir, exist_ok=True)

        if not VISUALIZATION_AVAILABLE:
            print("Warning: Visualization not available. Install opencv-python or imageio.")
            save_latents_vis = False
        else:
            # Set default upsample size if not provided
            if vis_upsample_size is None:
                vis_upsample_size = (height, width)

            # Save prepared latents
            print(f"Saving prepared latents visualization to: {vis_output_dir}/")
            _save_latents_vis(
                latents=latents,
                output_dir=vis_output_dir,
                step=0,
                name="prepared_latents",
                max_frames=vis_max_frames,
                fps=vis_fps,
                upsample_size=vis_upsample_size,
                upsample_interp=vis_upsample_interp,
                scale=vis_noise_scale,
                normalize=vis_normalize,
                save_delta=vis_save_delta,
            )

    mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)

    if getattr(self.config, "boundary_ratio", None) is not None and guidance_scale_2 is None:
        guidance_scale_2 = guidance_scale

    if getattr(self.config, "boundary_ratio", None) is not None:
        boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latents_ori = latents.clone()
            self._current_timestep = t

            if boundary_timestep is None or t >= boundary_timestep:
                current_model = self.transformer
                current_guidance_scale = guidance_scale
            else:
                current_model = transformer_2
                current_guidance_scale = guidance_scale_2

            if current_model is None:
                raise ValueError("No transformer is available for the current denoising stage.")

            current_transformer_dtype = current_model.dtype
            latent_model_input = latents.to(current_transformer_dtype)
            if getattr(self.config, "expand_timesteps", False):
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                timestep = t.expand(latents.shape[0])

            with _cache_context_or_null(current_model, "cond"):
                noise_pred = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            if self.do_classifier_free_guidance:
                with _cache_context_or_null(current_model, "uncond"):
                    noise_uncond = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

            latents_next = self.scheduler.step(
                noise_pred.float(),
                t,
                latents.float(),
                return_dict=False,
            )[0]
            _, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0),
                latents.float(),
                prev_sample=latents_next.float(),
                determistic=determistic,
                return_pixel_log_prob=return_pixel_log_prob
            )
            latents = latents_next
            prev_latents = latents.clone()

            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                prev_latents = latents.clone()
                log_prob = _gaussian_log_prob(prev_latents.float(), prev_latents_mean.float(), std_dev_t.float())

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # use kl_reward & is sampling process
            if kl_reward>0 and not determistic:
                latent_model_input = torch.cat([latents_ori] * 2) if self.do_classifier_free_guidance else latents_ori
                latent_model_input = latent_model_input.to(current_transformer_dtype)
                with current_model.disable_adapter():
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

                _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = sde_step_with_logprob(
                    self.scheduler, 
                    noise_pred.float(), 
                    t.unsqueeze(0), 
                    latents_ori.float(),
                    prev_sample=prev_latents.float(),
                    determistic=determistic,
                )
                assert std_dev_t == ref_std_dev_t
                kl = (prev_latents_mean - ref_prev_latents_mean)**2 / (2 * std_dev_t**2)
                kl = kl.mean(dim=tuple(range(1, kl.ndim)))
                all_kl.append(kl)
            else:
                # no kl reward, we do not need to compute, just put a pre-position value, kl will be 0
                all_kl.append(torch.zeros(len(latents), device=latents.device))

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            # if XLA_AVAILABLE:
            #     xm.mark_step()

    self._current_timestep = None

    if not output_type == "latent":
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents
    
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video, all_latents, all_log_probs, all_kl)

    return WanPipelineOutput(frames=video), all_latents, all_log_probs, all_kl
