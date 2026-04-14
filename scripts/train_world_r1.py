"""World-R1 training loop.

Source note:
- The policy optimization backbone is adapted from Flow-GRPO.
- The pipeline is simplified to the World-R1 training path for Wan 2.1 /
  CogVideoX with the reward and camera-conditioning logic kept in-tree.
"""

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import shutil
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import CogVideoXDDIMScheduler, CogVideoXPipeline, WanPipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob, sde_step_with_logprob
from flow_grpo.diffusers_patch.cogvideox_pipeline_with_logprob import (
    cogvideox_pipeline_with_logprob,
    ddim_step_with_logprob as cogvideox_ddim_step_with_logprob,
    prepare_cogvideox_latents,
)
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt
from flow_grpo.diffusers_patch.camera_trajectory_utils import (
    remove_camera_keywords_from_prompts,
    get_camera_trajectories_for_batch,
    lowpass_latent_delta,
    build_stepwise_delta_callback,
)
import torch
import wandb
from functools import partial
import tqdm
import tempfile
import itertools
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from peft.utils import get_peft_model_state_dict
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import imageio
from flow_grpo.diffusers_patch.camera_trajectory_utils import prepare_latents_with_camera
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9598))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

REWARD_TOTAL_KEY = flow_grpo.rewards.REWARD_TOTAL
RAW_REWARD_TOTAL_KEY = "raw_reward_total"
TRAJECTORY_COMPARISON_PATHS_KEY = flow_grpo.rewards.TRAJECTORY_COMPARISON_PATHS


def get_model_family(config):
    return getattr(config, "model_family", "wan")


def is_cogvideox_pipeline(pipeline):
    return isinstance(pipeline, CogVideoXPipeline)


def get_text_max_length(config):
    return getattr(config, "text_max_length", 512)


def mean_all_non_batch(tensor):
    return tensor.mean(dim=tuple(range(1, tensor.ndim)))


def compute_kl_loss(prev_sample_mean, prev_sample_mean_ref, std_dev_t, dt_ref, model_family):
    if model_family == "wan":
        noise_std = (std_dev_t * dt_ref).clamp_min(1e-8)
        return ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (
            2 * noise_std ** 2
        )

    diff = (prev_sample_mean - prev_sample_mean_ref) ** 2
    return mean_all_non_batch(diff / (2 * std_dev_t.clamp_min(1e-8) ** 2)).mean()


def prepare_model_latents_for_rollout(pipeline, latents, num_frames):
    if not is_cogvideox_pipeline(pipeline):
        return latents

    prepared, _ = prepare_cogvideox_latents(
        latents,
        in_channels=pipeline.transformer.config.in_channels,
        patch_size_t=pipeline.transformer.config.patch_size_t,
    )
    return prepared


def add_camera_trajectory_metadata(metadatas, camera_trajectory):
    if not isinstance(metadatas, list):
        return metadatas
    if isinstance(camera_trajectory, list) and len(camera_trajectory) == len(metadatas):
        camera_trajectories = camera_trajectory
    else:
        camera_trajectories = [camera_trajectory] * len(metadatas)
    updated = []
    for metadata, item_camera_trajectory in zip(metadatas, camera_trajectories):
        if isinstance(metadata, dict):
            new_metadata = dict(metadata)
            new_metadata["camera_trajectory"] = item_camera_trajectory
        else:
            new_metadata = {"camera_trajectory": item_camera_trajectory}
        updated.append(new_metadata)
    return updated


def prepare_rollout_latents_and_callback(
    pipeline,
    prompt,
    batch_size,
    num_channels_latents,
    height,
    width,
    num_frames,
    dtype,
    device,
    vae_scale_factor_temporal,
    frames_per_trajectory,
    force_camera_movement,
    noise_wrap_compute_dtype,
    noise_downtemp_interp,
    noise_downspatial_mode,
    noise_degradation,
    noise_wrap_flow_scale,
    wrap_strength,
    wrap_injection_mode,
    delta_lowpass_kernel,
    stepwise_guidance_steps,
    camera_trajectories,
    detected_movements_batch,
    debug_precompress_vis_dir=None,
):
    common_kwargs = dict(
        prompt=prompt,
        batch_size=batch_size,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=dtype,
        device=device,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        frames_per_trajectory=frames_per_trajectory,
        force_camera_movement=force_camera_movement,
        remove_camera_keywords_from_prompt=False,
        noise_wrap_compute_dtype=noise_wrap_compute_dtype,
        noise_downtemp_interp=noise_downtemp_interp,
        noise_downspatial_mode=noise_downspatial_mode,
        noise_degradation=noise_degradation,
        noise_wrap_flow_scale=noise_wrap_flow_scale,
        camera_trajectories=camera_trajectories,
        detected_movements_batch=detected_movements_batch,
        debug_precompress_vis_dir=debug_precompress_vis_dir,
    )

    if wrap_injection_mode != "stepwise_delta":
        latents = prepare_latents_with_camera(
            **common_kwargs,
            wrap_strength=wrap_strength,
            wrap_injection_mode=wrap_injection_mode,
            delta_lowpass_kernel=delta_lowpass_kernel,
        )
        latents = prepare_model_latents_for_rollout(pipeline, latents, num_frames)
        return latents, None

    wrapped_latents, base_latents = prepare_latents_with_camera(
        **common_kwargs,
        return_base_latents=True,
    )
    wrapped_latents = wrapped_latents.float()
    base_latents = base_latents.float()
    wrapped_latents = prepare_model_latents_for_rollout(pipeline, wrapped_latents, num_frames)
    base_latents = prepare_model_latents_for_rollout(pipeline, base_latents, num_frames)

    if wrap_strength is None or wrap_strength <= 0:
        return base_latents.to(dtype=dtype), None

    delta = wrapped_latents - base_latents
    delta_low = lowpass_latent_delta(delta, delta_lowpass_kernel)
    callback = build_stepwise_delta_callback(
        delta_low=delta_low,
        wrap_strength=float(wrap_strength),
        guidance_steps=stepwise_guidance_steps,
    )
    return base_latents.to(dtype=dtype), callback


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train', filter_type=None):
        """
        Args:
            dataset: path to dataset directory
            split: 'train' or 'test'
            filter_type: None (load all), 'main' (only main data), or 'dynamic' (only dynamic data)
        """
        self.file_path = os.path.join(dataset, f'{split}.txt')
        self.prompts = []
        self.is_dynamic = []

        # Load main dataset
        with open(self.file_path, 'r') as f:
            main_prompts = [line.strip() for line in f.readlines()]

        # Load dynamic dataset if exists
        dynamic_prompts = []
        dynamic_file_path = os.path.join(dataset, f'dynamic.txt')
        if os.path.exists(dynamic_file_path):
            with open(dynamic_file_path, 'r') as f:
                dynamic_prompts = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(dynamic_prompts)} dynamic prompts from {dynamic_file_path}")

        # Filter based on filter_type
        if filter_type == 'main':
            self.prompts = main_prompts
            self.is_dynamic = [False] * len(main_prompts)
        elif filter_type == 'dynamic':
            self.prompts = dynamic_prompts
            self.is_dynamic = [True] * len(dynamic_prompts)
        else:  # None - load all
            self.prompts.extend(main_prompts)
            self.is_dynamic.extend([False] * len(main_prompts))
            self.prompts.extend(dynamic_prompts)
            self.is_dynamic.extend([True] * len(dynamic_prompts))

        self.num_main = len(main_prompts)
        self.num_dynamic = len(dynamic_prompts)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {"is_dynamic": self.is_dynamic[idx]}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # 每卡的batch大小
        self.k = k                    # 每个样本重复的次数
        self.num_replicas = num_replicas  # 总卡数
        self.rank = rank              # 当前卡编号
        self.seed = seed              # 随机种子，用于同步
        
        # 计算每个迭代需要的不同样本数
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # 不同样本数
        self.epoch=0

    def __iter__(self):
        while True:
            # 生成确定性的随机序列，确保所有卡同步
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # print('epoch', self.epoch)
            # 随机选择m个不同的样本
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            # print(self.rank, 'indices', indices)
            # 每个样本重复k次，生成总样本数n*b
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # 打乱顺序确保均匀分配
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            # print(self.rank, 'shuffled_samples', shuffled_samples)
            # 将样本分割到各个卡
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            # print(self.rank, 'per_card_samples', per_card_samples[self.rank])
            # 返回当前卡的样本索引
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # 用于同步不同 epoch 的随机状态

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device, model_family="wan"):
    with torch.no_grad():
        if model_family == "cogvideox":
            tokenizer = tokenizers[0]
            text_encoder = text_encoders[0]
            prompt_list = [prompt] if isinstance(prompt, str) else prompt
            text_inputs = tokenizer(
                prompt_list,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
            prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
        else:
            prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, max_sequence_length
            )
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds

def set_adapter_and_freeze_params(transformer, adapter_name):
    transformer.module.set_adapter(adapter_name)
    for name, param in transformer.named_parameters():
        if "learner" in name:
            param.requires_grad_(True)
        elif "ref" in name:
            param.requires_grad_(False)

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    计算每个唯一提示词对应奖励值的标准差为零的比例
    
    参数:
        prompts: 提示词列表
        gathered_rewards: 包含奖励值的字典，须包含'raw_reward_total'键
        
    返回:
        zero_std_ratio: 标准差为零的比例
        prompt_std_devs: 每个唯一提示词对应的标准差数组
    """
    # 将提示词列表转换为NumPy数组
    prompt_array = np.array(prompts)
    
    # 获取唯一提示词及其分组信息
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # 分组获取每个提示词对应的奖励值
    grouped_rewards = gathered_rewards[RAW_REWARD_TOTAL_KEY][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # 计算每个分组的标准差
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # 计算零标准差的比例
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio
    

def get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def build_cogvideox_image_rotary_emb(pipeline, latents, config):
    if not getattr(pipeline.transformer.config, "use_rotary_positional_embeddings", False):
        return None
    return pipeline._prepare_rotary_positional_embeddings(
        config.height,
        config.width,
        latents.size(1),
        latents.device,
    )
        
def compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config, **kwargs):
    model_family = get_model_family(config)
    attention_kwargs = kwargs.get('attention_kwargs', getattr(config, 'attention_kwargs', None))
    if model_family == "cogvideox":
        latents = sample["latents"][:, j]
        timesteps = sample["timesteps"][:, j]
        image_rotary_emb = build_cogvideox_image_rotary_emb(pipeline, latents, config)

        if config.train.cfg:
            latent_model_input = torch.cat([latents, latents], dim=0)
            timestep_input = torch.cat([timesteps, timesteps], dim=0)
            encoder_hidden_states = torch.cat([negative_embeds, embeds], dim=0)
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = transformer(
                hidden_states=latents,
                timestep=timesteps,
                encoder_hidden_states=embeds,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        prev_sample, log_prob, prev_sample_mean, std_dev_t = cogvideox_ddim_step_with_logprob(
            pipeline.scheduler,
            noise_pred.float(),
            sample["timesteps"][:, j],
            sample["latents"][:, j].float(),
            prev_sample=sample["next_latents"][:, j].float(),
            noise_level=config.sample.noise_level,
        )
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, std_dev_t

    if config.train.cfg:
        noise_pred_text = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,  # Should contain both neg and pos embeds
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred_uncond = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=negative_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        return_dt_and_std_dev_t=True
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt


def rollout_with_logprob(
    pipeline,
    prompt_embeds,
    negative_prompt_embeds,
    latents,
    latent_callback,
    config,
    num_inference_steps,
    save_noise_vis=False,
    noise_vis_base=None,
    determistic=False,
):
    if is_cogvideox_pipeline(pipeline):
        return cogvideox_pipeline_with_logprob(
            pipeline,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=config.sample.guidance_scale,
            output_type="pt",
            return_dict=False,
            num_frames=config.frames,
            height=config.height,
            width=config.width,
            noise_level=0.0 if determistic else config.sample.noise_level,
            determistic=determistic,
            callback_on_step_end=latent_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            max_sequence_length=get_text_max_length(config),
        )

    return wan_pipeline_with_logprob(
        pipeline,
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=config.sample.guidance_scale,
        output_type="pt",
        return_dict=False,
        num_frames=config.frames,
        height=config.height,
        width=config.width,
        determistic=determistic,
        save_latents_vis=save_noise_vis,
        vis_output_dir=os.path.join(noise_vis_base, "prepared_latents") if noise_vis_base else None,
        callback_on_step_end=latent_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    text_max_length = get_text_max_length(config)
    neg_prompt_embed = compute_text_embeddings(
        [""],
        text_encoders,
        tokenizers,
        max_sequence_length=text_max_length,
        device=accelerator.device,
        model_family=get_model_family(config),
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)

    all_rewards = defaultdict(list)
    eval_trajectory_entries = []
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=text_max_length,
            device=accelerator.device,
            model_family=get_model_family(config),
        )
        # 最后一个batch可能不够batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]

        # Handle camera keyword removal for noise wrapping ablation (eval)
        eval_prompts = prompts
        if hasattr(config.sample, 'remove_camera_keywords') and config.sample.remove_camera_keywords:
            logger.info("Eval: Removing camera keywords from prompts for noise wrapping ablation")
            eval_prompts = remove_camera_keywords_from_prompts(prompts)

        force_camera_movement = getattr(config.sample, 'force_camera_movement', None)
        noise_wrap_compute_dtype = getattr(config.sample, "noise_wrap_compute_dtype", "fp32")
        noise_downtemp_interp = getattr(config.sample, "noise_downtemp_interp", "nearest")
        noise_downspatial_mode = getattr(config.sample, "noise_downspatial_mode", "area")
        noise_degradation = getattr(config.sample, "noise_degradation", 0.15)
        noise_wrap_flow_scale = getattr(config.sample, "noise_wrap_flow_scale", 32)
        wrap_strength = getattr(config.sample, "wrap_strength", None)
        wrap_injection_mode = getattr(config.sample, "wrap_injection_mode", "lowpass_delta")
        delta_lowpass_kernel = getattr(config.sample, "delta_lowpass_kernel", 9)
        stepwise_guidance_steps = getattr(config.sample, "stepwise_guidance_steps", 8)
        frames_per_trajectory = 81
        camera_trajectories, detected_movements_batch, _, _ = get_camera_trajectories_for_batch(
            eval_prompts,
            batch_size=len(prompts),
            frames_per_trajectory=frames_per_trajectory,
            force_camera_movement=force_camera_movement,
        )
        prompt_metadata = add_camera_trajectory_metadata(prompt_metadata, camera_trajectories)

        # Noise visualization for eval: save to {config.save_dir}/noise_vis/eval/ when enabled (main process only)
        save_noise_vis = getattr(config.sample, 'save_noise_vis', False) and accelerator.is_main_process
        noise_vis_base = os.path.join(config.save_dir, "noise_vis", f"eval_step_{global_step}") if save_noise_vis else None

        latents, latent_callback = prepare_rollout_latents_and_callback(
            pipeline=pipeline,
            prompt=eval_prompts,
            batch_size=prompt_embeds.shape[0] * 1,
            num_channels_latents=pipeline.transformer.config.in_channels,
            height=config.height,
            width=config.width,
            num_frames=config.frames,
            dtype=torch.bfloat16,
            device=accelerator.device,
            vae_scale_factor_temporal=pipeline.vae_scale_factor_temporal,
            frames_per_trajectory=frames_per_trajectory,
            force_camera_movement=force_camera_movement,
            noise_wrap_compute_dtype=noise_wrap_compute_dtype,
            noise_downtemp_interp=noise_downtemp_interp,
            noise_downspatial_mode=noise_downspatial_mode,
            noise_degradation=noise_degradation,
            noise_wrap_flow_scale=noise_wrap_flow_scale,
            wrap_strength=wrap_strength,
            wrap_injection_mode=wrap_injection_mode,
            delta_lowpass_kernel=delta_lowpass_kernel,
            stepwise_guidance_steps=stepwise_guidance_steps,
            camera_trajectories=camera_trajectories,
            detected_movements_batch=detected_movements_batch,
            debug_precompress_vis_dir=os.path.join(noise_vis_base, "precompress") if noise_vis_base else None,
        )
        with autocast():
            with torch.no_grad():
                videos, latents, log_probs, _ = rollout_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    latents=latents,
                    latent_callback=latent_callback,
                    config=config,
                    num_inference_steps=config.sample.eval_num_steps,
                    determistic=True,
                    save_noise_vis=save_noise_vis,
                    noise_vis_base=noise_vis_base,
                )
        rewards = executor.submit(reward_fn, videos, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()
        if accelerator.is_main_process and isinstance(reward_metadata, dict):
            traj_paths = reward_metadata.get(TRAJECTORY_COMPARISON_PATHS_KEY, [])
            if traj_paths:
                for idx, path in enumerate(traj_paths):
                    if not path or not os.path.exists(path):
                        continue
                    prompt = prompts[idx] if idx < len(prompts) else ""
                    reward_val = (
                        rewards[REWARD_TOTAL_KEY][idx]
                        if REWARD_TOTAL_KEY in rewards and idx < len(rewards[REWARD_TOTAL_KEY])
                        else None
                    )
                    eval_trajectory_entries.append(
                        {
                            "path": path,
                            "prompt": prompt,
                            "reward": float(reward_val) if reward_val is not None else None,
                        }
                    )

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_videos_gather = (
        accelerator.gather(torch.as_tensor(videos, device=accelerator.device)).float().cpu().numpy()
    )
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=text_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)

    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_videos_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                video = last_batch_videos_gather[index].transpose(0, 2, 3, 1)
                frames = [img for img in video]
                frames = [(frame * 255).astype(np.uint8) for frame in frames]
                imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=8, codec="libx264", format='FFMPEG')

            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            eval_payload = {
                "eval_images": [
                    wandb.Video(
                        os.path.join(tmpdir, f"{idx}.mp4"),
                        caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        format="mp4",
                        fps=8 
                    )
                    for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                ],
                **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
            }
            if eval_trajectory_entries:
                eval_payload["eval_trajectory"] = [
                    wandb.Image(
                        item["path"],
                        caption=f"{item['prompt']:.200} | total: {item['reward']:.2f}" if item["reward"] is not None else item["prompt"][:200],
                    )
                    for item in eval_trajectory_entries[: min(8, len(eval_trajectory_entries))]
                ]
            accelerator.log(eval_payload, step=global_step)
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
    gradient_accumulation_steps = config.train.gradient_accumulation_steps * num_train_timesteps

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    wandb_project_name = os.getenv("WANDB_PROJECT", "world-r1")
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_project_name,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    model_family = get_model_family(config)

    # load scheduler, tokenizer and models.
    if model_family == "cogvideox":
        pipeline = CogVideoXPipeline.from_pretrained(config.pretrained.model)
        pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config)
    else:
        pipeline = WanPipeline.from_pretrained(config.pretrained.model)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder]
    tokenizers = [pipeline.tokenizer]

    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move transformer, vae and text_encoder to device and cast to inference_dtype
    if model_family == "cogvideox":
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    else:
        pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    # pipeline.scheduler.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # pipeline.transformer.to(accelerator.device, dtype=inference_dtype)
        pipeline.transformer.to(accelerator.device)
        
        # pipeline.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if config.use_lora:
        if model_family == "cogvideox":
            target_modules = [
                "to_k",
                "to_out.0",
                "to_q",
                "to_v",
            ]
        else:
            target_modules = [
                "add_k_proj",
                "add_q_proj",
                "add_v_proj",
                "to_add_out",
                "to_k",
                "to_out.0",
                "to_q",
                "to_v",
            ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # 使用PeftModel.from_pretrained load后所有参数的requires_grad都是False，需要set_adapter来使得adapter参数梯度为True
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer.enable_gradient_checkpointing()
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # 平均影响到之前的20*8=160个step
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # 创建无限循环的DataLoader
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,  # 你的k值
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # 创建DataLoader，注意这里不需要shuffle，由Sampler控制
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create dynamic training dataloaders if enabled
        train_dataloader_dynamic = None
        if config.dynamic_training.enabled and train_dataset.num_dynamic > 0:
            train_dataset_main = TextPromptDataset(config.dataset, 'train', filter_type='main')
            train_dataset_dynamic = TextPromptDataset(config.dataset, 'train', filter_type='dynamic')

            train_sampler_main = DistributedKRepeatSampler(
                dataset=train_dataset_main,
                batch_size=config.sample.train_batch_size,
                k=config.sample.num_image_per_prompt,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                seed=42
            )
            train_sampler_dynamic = DistributedKRepeatSampler(
                dataset=train_dataset_dynamic,
                batch_size=config.sample.train_batch_size,
                k=config.sample.num_image_per_prompt,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                seed=43
            )

            train_dataloader_main = DataLoader(
                train_dataset_main,
                batch_sampler=train_sampler_main,
                num_workers=1,
                collate_fn=TextPromptDataset.collate_fn,
            )
            train_dataloader_dynamic = DataLoader(
                train_dataset_dynamic,
                batch_sampler=train_sampler_dynamic,
                num_workers=1,
                collate_fn=TextPromptDataset.collate_fn,
            )
            # Override train_dataloader with main dataloader
            train_dataloader = train_dataloader_main
            train_sampler = train_sampler_main

        # 创建正常的DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')
        # 创建无限循环的DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,  # 你的k值
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # 创建DataLoader，注意这里不需要shuffle，由Sampler控制
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        # 创建正常的DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")
    neg_prompt_embed = compute_text_embeddings(
        [""],
        text_encoders,
        tokenizers,
        max_sequence_length=get_text_max_length(config),
        device=accelerator.device,
        model_family=model_family,
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size * config.sample.sample_time_per_prompt, 1, 1)

    if config.sample.num_image_per_prompt * config.sample.sample_time_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    if config.dynamic_training.enabled and train_dataloader_dynamic is not None:
        transformer, optimizer, train_dataloader, train_dataloader_dynamic, test_dataloader = accelerator.prepare(
            transformer, optimizer, train_dataloader, train_dataloader_dynamic, test_dataloader
        )
    else:
        transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            transformer, optimizer, train_dataloader, test_dataloader
        )

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    if config.dynamic_training.enabled and train_dataloader_dynamic is not None:
        train_iter_dynamic = iter(train_dataloader_dynamic)
    else:
        train_iter_dynamic = None

    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # Determine which dataloader to use based on global_step
            if config.dynamic_training.enabled and train_iter_dynamic is not None:
                cycle_length = config.dynamic_training.main_steps + config.dynamic_training.dynamic_steps
                step_in_cycle = global_step % cycle_length
                use_dynamic = step_in_cycle >= config.dynamic_training.main_steps

                if use_dynamic:
                    current_sampler = train_sampler_dynamic
                    current_iter = train_iter_dynamic
                    current_dataloader_name = "dynamic"
                else:
                    current_sampler = train_sampler_main
                    current_iter = train_iter
                    current_dataloader_name = "main"

                current_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
                prompts, prompt_metadata = next(current_iter)

                if accelerator.is_local_main_process and i == 0:
                    logger.info(f"Epoch {epoch}, global_step {global_step}, using {current_dataloader_name} dataloader")
            else:
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
                prompts, prompt_metadata = next(train_iter)

            prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=get_text_max_length(config),
                device=accelerator.device,
                model_family=model_family,
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=get_text_max_length(config),
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            if i==0 and epoch % config.eval_freq == 0 and epoch>0:
                eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
            if i==0 and epoch % config.save_freq == 0 and epoch>0 and accelerator.is_main_process:
                save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
            # 这里是故意的，因为前两个epoch收集的group size会有bug,经过两个epoch后，group_size稳定成指定的
            if epoch < 2:
                continue
            # sample
            for j in tqdm(
                range(config.sample.sample_time_per_prompt),
                desc=f"Epoch {epoch}: sampling | multi sample per prompt",
                disable=not accelerator.is_local_main_process,
                position=1,
            ):
                # Handle camera keyword removal for noise wrapping ablation (training)
                train_prompts = prompts
                if hasattr(config.sample, 'remove_camera_keywords') and config.sample.remove_camera_keywords:
                    train_prompts = remove_camera_keywords_from_prompts(prompts)

                force_camera_movement = getattr(config.sample, 'force_camera_movement', None)
                noise_wrap_compute_dtype = getattr(config.sample, "noise_wrap_compute_dtype", "fp32")
                noise_downtemp_interp = getattr(config.sample, "noise_downtemp_interp", "nearest")
                noise_downspatial_mode = getattr(config.sample, "noise_downspatial_mode", "area")
                noise_degradation = getattr(config.sample, "noise_degradation", 0.15)
                noise_wrap_flow_scale = getattr(config.sample, "noise_wrap_flow_scale", 32)
                wrap_strength = getattr(config.sample, "wrap_strength", None)
                wrap_injection_mode = getattr(config.sample, "wrap_injection_mode", "lowpass_delta")
                delta_lowpass_kernel = getattr(config.sample, "delta_lowpass_kernel", 9)
                stepwise_guidance_steps = getattr(config.sample, "stepwise_guidance_steps", 8)
                frames_per_trajectory = 81
                camera_trajectories, detected_movements_batch, _, _ = get_camera_trajectories_for_batch(
                    train_prompts,
                    batch_size=len(prompts),
                    frames_per_trajectory=frames_per_trajectory,
                    force_camera_movement=force_camera_movement,
                )
                prompt_metadata = add_camera_trajectory_metadata(prompt_metadata, camera_trajectories)

                # Noise visualization: save to {config.save_dir}/noise_vis/ when enabled (main process only)
                save_noise_vis = getattr(config.sample, 'save_noise_vis', False) and accelerator.is_main_process
                noise_vis_base = os.path.join(config.save_dir, "noise_vis", f"epoch_{epoch}", f"batch_{i}", f"iter_{j}") if save_noise_vis else None

                latents, latent_callback = prepare_rollout_latents_and_callback(
                    pipeline=pipeline,
                    prompt=train_prompts,
                    batch_size=prompt_embeds.shape[0] * 1,
                    num_channels_latents=pipeline.transformer.config.in_channels,
                    height=config.height,
                    width=config.width,
                    num_frames=config.frames,
                    dtype=inference_dtype,
                    device=accelerator.device,
                    vae_scale_factor_temporal=pipeline.vae_scale_factor_temporal,
                    frames_per_trajectory=frames_per_trajectory,
                    force_camera_movement=force_camera_movement,
                    noise_wrap_compute_dtype=noise_wrap_compute_dtype,
                    noise_downtemp_interp=noise_downtemp_interp,
                    noise_downspatial_mode=noise_downspatial_mode,
                    noise_degradation=noise_degradation,
                    noise_wrap_flow_scale=noise_wrap_flow_scale,
                    wrap_strength=wrap_strength,
                    wrap_injection_mode=wrap_injection_mode,
                    delta_lowpass_kernel=delta_lowpass_kernel,
                    stepwise_guidance_steps=stepwise_guidance_steps,
                    camera_trajectories=camera_trajectories,
                    detected_movements_batch=detected_movements_batch,
                    debug_precompress_vis_dir=os.path.join(noise_vis_base, "precompress") if noise_vis_base else None,
                )
                with autocast():
                    with torch.no_grad():
                        videos, latents, log_probs, kls = rollout_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            latents=latents,
                            latent_callback=latent_callback,
                            config=config,
                            num_inference_steps=config.sample.num_steps,
                            determistic=(model_family != "cogvideox"),
                            save_noise_vis=save_noise_vis,
                            noise_vis_base=noise_vis_base,
                        )
                # Compute save directory for this batch/iter
                save_root = os.path.join(config.save_dir, "rl_videos", f"epoch_{epoch}", f"batch_{i}", f"iter_{j}")

                # Save all videos in this batch for RL training (main process only)
                if accelerator.is_main_process:
                    try:
                        os.makedirs(save_root, exist_ok=True)
                        # videos: (B, T, C, H, W)
                        bsz = videos.shape[0]
                        for b in range(bsz):
                            video = videos[b]  # (T, C, H, W)
                            frames = [img for img in video.float().cpu().numpy().transpose(0, 2, 3, 1)]
                            frames = [(frame * 255).astype(np.uint8) for frame in frames]
                            out_path = os.path.join(save_root, f"vid_{b}.mp4")
                            imageio.mimsave(out_path, frames, fps=8, codec="libx264", format='FFMPEG')

                        # Save prompts explicitly alongside videos for easier inspection.
                        with open(os.path.join(save_root, "prompts.json"), "w", encoding="utf-8") as f:
                            json.dump(list(prompts), f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to save RL batch videos: {e}")

                latents = torch.stack(
                    latents, dim=1
                )  # (batch_size, num_steps + 1, 16, 96, 96)
                log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
                kls = torch.stack(kls, dim=1) 
                kl = kls.detach()

                timesteps = pipeline.scheduler.timesteps.repeat(
                    config.sample.train_batch_size, 1
                )  # (batch_size, num_steps)

                # compute rewards asynchronously
                rewards = executor.submit(reward_fn, videos, prompts, prompt_metadata, only_strict=True)
                # images b, 3, 512, 512
                # yield to to make sure reward computation starts
                time.sleep(0)
                
                samples.append(
                    {
                        "prompt_ids": prompt_ids,   # b, 77
                        "prompt_embeds": prompt_embeds,    # b, 205, 4096
                        "negative_prompt_embeds": sample_neg_prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[
                            :, :-1
                        ],  # each entry is the latent before timestep t.   b, 11, 16, 64, 64
                        "next_latents": latents[
                            :, 1:
                        ],  # each entry is the latent after timestep t
                        "log_probs": log_probs,   # b, t + 1
                        "kl": kl,
                        "rewards": rewards,
                        "save_root": save_root,
                        "prompts": prompts,
                    }
                )

        if epoch < 2:
            continue
        # wait for all rewards to be computed
        trajectory_log_entries = []
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            # Save per-video reward JSON alongside saved videos (main process only)
            if accelerator.is_main_process:
                try:
                    os.makedirs(sample["save_root"], exist_ok=True)
                    copied_trajectory_paths = []
                    if isinstance(reward_metadata, dict):
                        for idx, src_path in enumerate(reward_metadata.get(TRAJECTORY_COMPARISON_PATHS_KEY, [])):
                            if not src_path or not os.path.exists(src_path):
                                copied_trajectory_paths.append("")
                                continue
                            dst_path = os.path.join(sample["save_root"], f"trajectory_{idx}.png")
                            shutil.copy2(src_path, dst_path)
                            copied_trajectory_paths.append(dst_path)
                    # Structure: list of dicts per video index
                    # Each dict: {"prompt": str, "rewards": {k: v_i}}
                    per_video = []
                    batch_size = len(sample["rewards"][next(iter(sample["rewards"]))])
                    for b in range(batch_size):
                        current_entry = {
                            "prompt": sample["prompts"][b] if b < len(sample["prompts"]) else "",
                            "rewards": {k: float(sample["rewards"][k][b].item()) for k in sample["rewards"]}
                        }
                        if b < len(copied_trajectory_paths) and copied_trajectory_paths[b]:
                            current_entry["trajectory_comparison_path"] = copied_trajectory_paths[b]
                            trajectory_log_entries.append(
                                {
                                    "path": copied_trajectory_paths[b],
                                    "prompt": current_entry["prompt"],
                                    "reward": current_entry["rewards"].get(REWARD_TOTAL_KEY),
                                }
                            )
                        per_video.append(current_entry)
                    with open(os.path.join(sample["save_root"], "rewards.json"), "w", encoding="utf-8") as f:
                        json.dump(per_video, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save rewards.json: {e}")
            # Remove non-tensor metadata to avoid torch.cat errors in collation
            if "save_root" in sample:
                del sample["save_root"]
            if "prompts" in sample:
                del sample["prompts"]

        # Synchronize CUDA and clear cache after all rewards are computed
        # This prevents CUDA errors from pointVLM from affecting subsequent operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # if epoch % 10 == 0 and accelerator.is_main_process:
        if accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(videos))
                sample_indices = random.sample(range(len(videos)), num_samples)

                for idx, i in enumerate(sample_indices):
                    video = videos[i]
                    frames = [img for img in video.float().cpu().numpy().transpose(0, 2, 3, 1)]
                    frames = [(frame * 255).astype(np.uint8) for frame in frames]
                    imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=8, codec="libx264", format='FFMPEG')         

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards[REWARD_TOTAL_KEY][i] for i in sample_indices]

                train_payload = {
                    "video": [
                        wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{prompt:.100} | total: {avg_reward:.2f}",
                            format="mp4",
                            fps=8 
                        )
                        for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                }
                if trajectory_log_entries:
                    train_payload["trajectory"] = [
                        wandb.Image(
                            item["path"],
                            caption=f"{item['prompt']:.200} | total: {item['reward']:.2f}" if item.get("reward") is not None else item["prompt"][:200],
                        )
                        for item in trajectory_log_entries[: min(8, len(trajectory_log_entries))]
                    ]
                accelerator.log(train_payload, step=global_step)
        samples["rewards"][RAW_REWARD_TOTAL_KEY] = samples["rewards"][REWARD_TOTAL_KEY]
        samples["rewards"][REWARD_TOTAL_KEY] = (
            samples["rewards"][REWARD_TOTAL_KEY].unsqueeze(-1) - config.sample.kl_reward * samples["kl"]
        )
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        accelerator.log(
            {
                "epoch": epoch,
                **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                "kl": samples["kl"].mean().cpu().numpy(),
                "kl_abs": samples["kl"].abs().mean().cpu().numpy()
            },
            step=global_step,
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            # print(f"[Rank {accelerator.process_index}] prompt_ids shape before gather: {samples['prompt_ids'].shape}")
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )

            advantages = stat_tracker.update(prompts, gathered_rewards[REWARD_TOTAL_KEY])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio = calculate_zero_std_ratio(prompts, gathered_rewards)

            accelerator.log(
                {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                },
                step=global_step,
            )
            stat_tracker.clear()
        else:
            advantages = (
                gathered_rewards[REWARD_TOTAL_KEY] - gathered_rewards[REWARD_TOTAL_KEY].mean()
            ) / (gathered_rewards[REWARD_TOTAL_KEY].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())
            print("kl: ", samples["kl"].mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)

        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt
        true_count = mask.sum().item()  # Convert to Python int
        if true_count == 0:
            print("advantages: ", samples["advantages"].abs().mean())
            print("mask.sum() == 0. revise in this rank")
            samples["advantages"] = samples["advantages"] + 1e-6
            print("after revise advantages: ", samples["advantages"].abs().mean())
            mask = (samples["advantages"].abs().sum(dim=1) != 0)
            true_count = mask.sum().item()  # Recompute as Python int

        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices), device=mask.device)[:num_to_change]
                # Use .clone() to avoid in-place modification issues
                indices_to_flip = false_indices[random_indices]
                mask = mask.clone()
                mask[indices_to_flip] = True

        accelerator.log(
            {
                "actual_batch_size": mask.sum().item()// (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt),
            },
            step=global_step,
        )

        # Filter out samples where the entire time dimension of advantages is zero
        # Only apply mask to tensors (skip dicts or other types)
        # samples = {k: v[mask.to(v.device)] if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        samples = {k: v if isinstance(v, torch.Tensor) else v for k, v in samples.items()}
        # samples = {k: v[mask] if isinstance(v, torch.Tensor) else v for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            # perm = torch.randperm(total_batch_size, device=accelerator.device)
            # # perm = torch.arange(total_batch_size, device=accelerator.device)
            # samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    # torch.randperm(num_timesteps, device=accelerator.device)
                    torch.arange(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            micoe_batch = total_batch_size // (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt)

            samples_batched = {
                k: v.reshape(-1, micoe_batch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = sample["prompt_embeds"]
                    negative_embeds = train_neg_prompt_embeds[:len(sample["prompt_embeds"])]
                else:
                    embeds = sample["prompt_embeds"]
                    negative_embeds = None

                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    try:
                                        with transformer.module.disable_adapter():
                                            prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref, dt_ref = compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config)
                                    # if no module, try modules, no modules, try named_modules:
                                    except:
                                        try:
                                            with transformer.modules.disable_adapter():
                                                prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref, dt_ref = compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config)
                                        except:
                                            with transformer.disable_adapter():
                                                prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref, dt_ref = compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config)
                                    
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        if config.train.beta > 0:
                            kl_loss = compute_kl_loss(
                                prev_sample_mean,
                                prev_sample_mean_ref,
                                std_dev_t,
                                dt_ref,
                                model_family,
                            )
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients

    accelerator.wait_for_everyone()
    accelerator.end_training()
        
if __name__ == "__main__":
    app.run(main)
