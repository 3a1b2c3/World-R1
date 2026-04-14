"""World-R1 experiment configs."""

import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def _build_world_r1_config(model_path: str, run_name: str, model_family: str = "wan"):
    config = base.get_config()

    config.dataset = os.path.join(os.getcwd(), "dataset/enhanced")
    config.pretrained.model = model_path
    config.model_family = model_family

    config.run_name = run_name
    config.height = 480
    config.width = 832
    config.frames = 81

    config.sample.num_steps = 50
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 5.0
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 2
    config.sample.num_batches_per_epoch = 24
    config.sample.sample_time_per_prompt = 1
    config.sample.test_batch_size = 2
    config.sample.global_std = False
    config.sample.kl_reward = 0
    config.sample.noise_level = 0.7

    config.sample.remove_camera_keywords = False
    config.sample.force_camera_movement = None
    config.sample.noise_wrap_compute_dtype = "fp32"
    config.sample.noise_downtemp_interp = "nearest"
    config.sample.noise_downspatial_mode = "resize_noise"
    config.sample.noise_degradation = 0.35
    config.sample.noise_wrap_flow_scale = 16
    config.sample.wrap_strength = 0.45
    config.sample.wrap_injection_mode = "stepwise_delta"
    config.sample.delta_lowpass_kernel = 9
    config.sample.stepwise_guidance_steps = 8

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 2 * config.sample.train_batch_size * config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.49
    config.train.beta = 0.004
    config.train.learning_rate = 1e-4
    config.train.clip_range = 1e-3
    config.train.sft = 0.0
    config.train.sft_batch_size = 3
    config.train.ema = True

    config.mixed_precision = "bf16"
    config.diffusion_loss = True
    config.num_epochs = 100000
    config.save_freq = 60
    config.eval_freq = 30
    config.save_dir = f"logs/world_r1/{config.run_name}"
    config.resume_from = None

    config.reward_fn = ml_collections.ConfigDict(
        {
            "reward_3d": 1.0,
            "reward_general": 1.0,
        }
    )
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True

    config.dynamic_training.enabled = True
    config.dynamic_training.main_steps = 100
    config.dynamic_training.dynamic_steps = 50

    return config


def world_r1_small():
    config = _build_world_r1_config(
        model_path="hf_cache/Wan2.1-T2V-1.3B-Diffusers",
        run_name="world_r1_small",
    )
    config.sample.wrap_strength = 0.35
    return config


def world_r1_large():
    config = _build_world_r1_config(
        model_path="hf_cache/Wan2.1-T2V-14B-Diffusers",
        run_name="world_r1_large",
    )
    config.sample.wrap_strength = 0.4
    return config


def world_r1_cogvideox_5b():
    config = _build_world_r1_config(
        model_path="THUDM/CogVideoX1.5-5B",
        run_name="world_r1_cogvideox_5b",
        model_family="cogvideox",
    )
    config.text_max_length = 226
    config.sample.guidance_scale = 6.0
    config.sample.noise_level = 0.7
    config.sample.wrap_strength = 0.4
    return config


def get_config(name="world_r1_large"):
    configs = {
        "world_r1_small": world_r1_small,
        "world_r1_large": world_r1_large,
        "world_r1_cogvideox_5b": world_r1_cogvideox_5b,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {sorted(configs)}")
    return configs[name]()
