#!/usr/bin/env python3
"""Multi-GPU backend for the 3D reward."""

import torch
import numpy as np
from PIL import Image
from io import BytesIO
from multiprocessing import Process, Queue
import os
import json
import datetime

from reward_server.reward_3d_backend import DEFAULT_RECONSTRUCTION_MODEL, Reward3DBackend


def reward_3d_worker_process(gpu_id, model_name, scorer_type, task_queue, result_queue):
    """
    Worker process that runs the 3D reward stack on a specific GPU.

    Args:
        gpu_id: CUDA device ID
        model_name: Reconstruction model name
        scorer_type: Type of scorer ('qwen' or 'openai')
        task_queue: Queue to receive tasks (batch_idx, video_frames, prompt, save_dir)
        result_queue: Queue to send results including per-video artifact paths
    """
    try:
        # CRITICAL: Set CUDA device BEFORE any CUDA operations
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Re-import torch after setting CUDA_VISIBLE_DEVICES
        import torch as torch_local
        device = torch_local.device("cuda:0")  # Maps to the physical GPU

        print(f"[Process {os.getpid()}] Initializing 3D reward backend on physical GPU {gpu_id}")
        reward_3d_backend = Reward3DBackend(device=device, model_name=model_name)

        # Initialize Qwen3-VL scorer
        # IMPORTANT: In multiprocessing with CUDA_VISIBLE_DEVICES, we need to use explicit device
        # instead of device_map="auto" to avoid device mismatch issues
        print(f"[Process {os.getpid()}] Initializing Qwen3-VL scorer on GPU {gpu_id}")

        # Create a custom QwenVLScorer that doesn't use device_map="auto"
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import re

        class LocalQwenVLScorer:
            """Local QwenVL scorer without device_map for multiprocessing"""
            def __init__(self, device, dtype):
                self.device = device
                self.dtype = dtype

                # Load model on specific device (NOT device_map="auto")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen3-VL-4B-Instruct",
                    torch_dtype=dtype,
                ).to(device)
                self.model.requires_grad_(False)
                self.model.eval()

                if hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = False

                self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

                # Temperature for logit-based scoring
                self.temperature = 1.0

                # Video evaluation prompt (from qwenvl3.py) - for GS video
                # Modified to only output a single digit for logit-based scoring
                self.video_task_template = '''You are given a text prompt: "{prompt}"
According the generated video sequence and the prompt, evaluate the video quality.

1. Analyze the video content and progression across all frames.
2. Identify key visual elements and instructions from the prompt.
3. Evaluate how well this video follows the prompt:
   - Does the camera movement align with the prompt? **If the camera is static, the score should be 0.**
   - Are all required elements present across the video?
   - Are object counts, colors, and positions accurate?
   - Is there logical temporal consistency between frames?

Provide a score from 0 to 9:
- 9: Perfect alignment with prompt and high quality
- 7-8: Very good alignment with minor issues
- 5-6: Good alignment but noticeable problems
- 3-4: Poor alignment with major issues
- 1-2: No alignment or very low quality
- 0: The camera is static, no movement.

Output only a single digit (0-9):'''

                # Image evaluation prompt (from pointvlm_v3.py) - for meta view (GS render)
                # Modified to only output a single digit for logit-based scoring
                self.image_task_template = '''You are a professional 3D vision expert. I used a text prompt to generate a video and reconstructed a corresponding 3D Pointmap from the video.
Original Prompt:
{text_prompt}

Your task is to judge the quality of the original video by analyzing the provided image of its resulting 3D pointmap. A good video (smooth, orbiting camera) creates a good pointmap. A bad video (static, jittery, or zooming) creates a bad pointmap.
Please provide a score from 0 to 9 based on these criteria:

- 9: Excellent - A dense, clean, and complete 3D model. Perfect 360° orbital motion, high stability.
- 7-8: Good - A clear object with strong 3D structure. May have minor holes or noise. Good, smooth camera arc with strong parallax.
- 4-6: Mediocre - Object is recognizable, but the map is sparse, noisy, or "flat" (lacks 3D depth). Poor parallax (e.g., just a zoom or pan instead of an orbit), or the video was jittery, blurry, or had object/lighting inconsistencies.
- 2-3: Poor - A chaotic jumble of points or a simple 2D projection. Static camera (no motion) or completely unstable.
- 0-1: Very Poor - Empty or just random noise. Unusable.

Output only a single digit (0-9):'''

            def __call__(self, prompts, videos):
                """Score videos with prompts"""
                import base64
                from io import BytesIO
                import torchvision.transforms.functional as F_vision

                def pil_image_to_base64(image):
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    return f"data:image;base64,{encoded_image_text}"

                all_rewards = []

                for prompt, video in zip(prompts, videos):
                    try:
                        # Convert tensor to PIL frames
                        if isinstance(video, torch_local.Tensor):
                            # IMPORTANT: Move to CPU first
                            video = video.cpu()
                            if video.dtype.is_floating_point:
                                video = (video.clamp(0, 1) * 255.0).to(torch_local.uint8)
                            pil_frames = [F_vision.to_pil_image(video[i]) for i in range(video.shape[0])]
                        else:
                            pil_frames = video

                        # Sample frames
                        # num_frames = len(pil_frames)
                        # target_frames = max(1, round(num_frames / 4))
                        # if num_frames > target_frames:
                        #     indices = list(range(0, num_frames, num_frames // target_frames))[:target_frames]
                        #     pil_frames = [pil_frames[i] for i in indices]

                        # Choose template and message type based on input
                        if len(pil_frames) == 1:
                            # Single image (meta_view): use image prompt from pointvlm_v3.py
                            task = self.image_task_template.format(text_prompt=prompt)
                            image_base64 = pil_image_to_base64(pil_frames[0])
                            message = {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image_base64},
                                    {"type": "text", "text": task},
                                ],
                            }
                        else:
                            # Multiple frames (gs_video): use video prompt from qwenvl3.py
                            task = self.video_task_template.format(prompt=prompt)
                            video_base64 = [pil_image_to_base64(frame) for frame in pil_frames]
                            message = {
                                "role": "user",
                                "content": [
                                    {"type": "video", "video": video_base64},
                                    {"type": "text", "text": task},
                                ],
                            }

                        # Process
                        text = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info([[message]])
                        batch_data = self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )

                        # Move inputs to device
                        input_ids = batch_data['input_ids'].to(self.device)
                        attention_mask = batch_data['attention_mask'].to(self.device)
                        pixel_values = batch_data.get('pixel_values')
                        if pixel_values is not None:
                            pixel_values = pixel_values.to(self.device)
                        pixel_values_videos = batch_data.get('pixel_values_videos')
                        if pixel_values_videos is not None:
                            pixel_values_videos = pixel_values_videos.to(self.device)
                        image_grid_thw = batch_data.get('image_grid_thw')
                        if image_grid_thw is not None:
                            image_grid_thw = image_grid_thw.to(self.device)
                        video_grid_thw = batch_data.get('video_grid_thw')
                        if video_grid_thw is not None:
                            video_grid_thw = video_grid_thw.to(self.device)

                        # Forward pass to get logits (not generation)
                        cache_position = torch_local.arange(0, input_ids.shape[1], device=self.device)

                        with torch_local.no_grad():
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=None,
                                past_key_values=None,
                                inputs_embeds=None,
                                labels=None,
                                use_cache=False,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=True,
                                pixel_values=pixel_values,
                                pixel_values_videos=pixel_values_videos,
                                image_grid_thw=image_grid_thw,
                                video_grid_thw=video_grid_thw,
                                rope_deltas=None,
                                cache_position=cache_position,
                                second_per_grid_ts=None
                            )

                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                        # Get logits at the last valid token position
                        last_valid_indices = (attention_mask != 0).cumsum(dim=1).argmax(dim=1)
                        last_token_logits = logits[torch_local.arange(logits.size(0)), last_valid_indices, :]

                        # Extract logits for tokens 15-24 (corresponding to digits 0-9)
                        # Token IDs: 15='0', 16='1', ..., 24='9'
                        digit_logits = last_token_logits[:, 15:25]  # (batch_size, 10)

                        # Compute score as weighted average using softmax
                        scores_range = torch_local.arange(0, 10, device=self.device).float()  # [0, 1, 2, ..., 9]
                        probs = (digit_logits / self.temperature).softmax(dim=-1)
                        raw_score = (probs * scores_range).sum(dim=-1).item()

                        # Normalize to 0-1 range (0-9 scale -> 0-1)
                        score = raw_score / 9.0

                        print(f"[Process {os.getpid()}] Logit-based score: {raw_score:.2f}/9 = {score:.4f}")
                        print(f"[Process {os.getpid()}] Digit probabilities: {probs[0]}")

                        # Debug: Also generate text to see what the model would output
                        try:
                            generated_ids = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                pixel_values_videos=pixel_values_videos,
                                image_grid_thw=image_grid_thw,
                                video_grid_thw=video_grid_thw,
                                max_new_tokens=128,
                                do_sample=False,
                            )
                            # Decode only the newly generated tokens
                            generated_text = self.processor.batch_decode(
                                generated_ids[:, input_ids.shape[1]:],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False
                            )[0]
                            print(f"[Process {os.getpid()}] Generated text (debug): {generated_text.strip()}")
                        except Exception as gen_e:
                            print(f"[Process {os.getpid()}] Debug generation failed: {gen_e}")

                        all_rewards.append(score)

                        # Clear memory
                        del batch_data, input_ids, attention_mask, outputs, logits
                        if pixel_values is not None:
                            del pixel_values
                        if pixel_values_videos is not None:
                            del pixel_values_videos
                        torch_local.cuda.empty_cache()

                    except Exception as e:
                        print(f"[Process {os.getpid()}] Error scoring video: {e}")
                        import traceback
                        traceback.print_exc()
                        all_rewards.append(0.0)
                        torch_local.cuda.empty_cache()

                return all_rewards

        # Create OpenAI scorer (doesn't need GPU)
        class OpenAIScorer:
            def __init__(self):
                from flow_grpo.azure_openai_vision import AzureOpenAIVision

                API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
                ENDPOINT = os.environ.get(
                    "AZURE_OPENAI_ENDPOINT",
                    "",
                )

                self.vision_client = AzureOpenAIVision(API_KEY, ENDPOINT)

                # Video evaluation prompt (adapted from pointvlm_v2.py) - for GS video
                self.video_task_template = '''You are a professional 3D vision expert. I used a text prompt to generate a video and reconstructed a corresponding 3D Pointmap from the video.
Original Prompt:
{text_prompt}

Your task is to judge the quality of the original video by analyzing the provided image of its resulting 3D pointmap. A good video (smooth, orbiting camera) creates a good pointmap. A bad video (static, jittery, or zooming) creates a bad pointmap.
Please provide a score from 0 to 10 based on these criteria:

- 9-10 (Excellent): A dense, clean, and complete 3D model. Perfect 360° orbital motion, high stability.
- 7-8 (Good): A clear object with strong 3D structure. May have minor holes or noise. Good, smooth camera arc with strong parallax.
- 4-6 (Mediocre): Object is recognizable, but the map is sparse, noisy, or "flat" (lacks 3D depth). Poor parallax (e.g., just a zoom or pan instead of an orbit), or the video was jittery, blurry, or had object/lighting inconsistencies.
- 1-3 (Poor): A chaotic jumble of points or a simple 2D projection. Static camera (no motion) or completely unstable.
- 0 (Total Failure): Empty or just random noise. Unusable.

Format your final answer as: <Score>X</Score>
'''

                # Image evaluation prompt (same as video for OpenAI)
                self.image_task_template = self.video_task_template

            def __call__(self, prompts, videos):
                """Score videos/images with prompts using OpenAI API"""
                import tempfile
                import cv2 as cv2_local

                all_rewards = []

                for prompt, video in zip(prompts, videos):
                    try:
                        # Convert tensor to image and save temporarily
                        if isinstance(video, torch_local.Tensor):
                            video = video.cpu()
                            if video.dtype.is_floating_point:
                                video = (video.clamp(0, 1) * 255.0).to(torch_local.uint8)

                            # For multi-frame video, use middle frame
                            if video.shape[0] > 1:
                                mid_idx = video.shape[0] // 2
                                frame = video[mid_idx]
                            else:
                                frame = video[0]

                            # Convert to numpy and save as temporary image
                            frame_np = frame.permute(1, 2, 0).cpu().numpy()
                            frame_bgr = cv2_local.cvtColor(frame_np, cv2_local.COLOR_RGB2BGR)

                            # Save to temporary file
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                                tmp_path = tmp_file.name
                                cv2_local.imwrite(tmp_path, frame_bgr)
                        else:
                            tmp_path = video  # Assume it's already a path

                        # Choose prompt template
                        question = self.video_task_template.format(text_prompt=prompt)

                        # Query OpenAI API
                        print(f"[Process {os.getpid()}] Querying OpenAI API...")
                        response = self.vision_client.get_answer(tmp_path, question)
                        print(f"[Process {os.getpid()}] OpenAI response: {response[:500]}")

                        # Extract score (0-10 scale)
                        match = re.search(r'<Score>(\d+(?:\.\d+)?)</Score>', response)
                        if match:
                            raw_score = float(match.group(1))
                            # OpenAI uses 0-10 scale, normalize to 0-1
                            score = raw_score / 10.0
                            print(f"[Process {os.getpid()}] Extracted score: {raw_score}/10 = {score}")
                        else:
                            print(f"[Process {os.getpid()}] Warning: No score found in OpenAI response, returning 0.0")
                            score = 0.0

                        all_rewards.append(score)

                        # Clean up temporary file
                        if isinstance(video, torch_local.Tensor):
                            os.remove(tmp_path)

                    except Exception as e:
                        print(f"[Process {os.getpid()}] Error scoring with OpenAI: {e}")
                        import traceback
                        traceback.print_exc()
                        all_rewards.append(0.0)

                return all_rewards

        # Initialize scorer based on type
        if scorer_type == 'qwen':
            gs_scorer = LocalQwenVLScorer(device=device, dtype=torch_local.bfloat16)
            meta_scorer = gs_scorer  # Same scorer for both
            print(f"[Process {os.getpid()}] 3D reward backend + Qwen3-VL ready on GPU {gpu_id}")
        elif scorer_type == 'openai':
            # Hybrid mode: Qwen for video, OpenAI for image
            gs_scorer = LocalQwenVLScorer(device=device, dtype=torch_local.bfloat16)
            meta_scorer = OpenAIScorer()
            print(f"[Process {os.getpid()}] 3D reward backend + Qwen3-VL (video) + OpenAI Vision (image) ready on GPU {gpu_id}")
        else:
            raise ValueError(f"Unknown scorer_type: {scorer_type}. Must be 'qwen' or 'openai'")

        lpips_model = None

        def compute_lpips_gs_score(gs_video_tensor, input_video_tensor):
            nonlocal lpips_model
            if lpips_model is None:
                try:
                    import lpips as lpips_lib
                except Exception as e:
                    print(f"[Process {os.getpid()}] Failed to import lpips: {e}")
                    return 0.0
                lpips_model = lpips_lib.LPIPS(net="alex").to(device)
                lpips_model.eval()

            if gs_video_tensor.dim() != 4 or input_video_tensor.dim() != 4:
                print(f"[Process {os.getpid()}] Invalid LPIPS tensor shapes: gs={gs_video_tensor.shape}, input={input_video_tensor.shape}")
                return 0.0

            num_frames = min(gs_video_tensor.shape[0], input_video_tensor.shape[0])
            if num_frames <= 0:
                print(f"[Process {os.getpid()}] Empty video for LPIPS scoring")
                return 0.0

            gs_clip = gs_video_tensor[:num_frames].float()
            input_clip = input_video_tensor[:num_frames].float()

            if input_clip.shape[-2:] != gs_clip.shape[-2:]:
                input_clip = torch_local.nn.functional.interpolate(
                    input_clip,
                    size=gs_clip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            gs_norm = gs_clip * 2.0 - 1.0
            input_norm = input_clip * 2.0 - 1.0

            with torch_local.no_grad():
                lpips_values = lpips_model(gs_norm, input_norm)

            lpips_mean = lpips_values.mean().item()
            gs_score = float(np.clip(1.0 - lpips_mean, 0.0, 1.0))
            print(f"[Process {os.getpid()}] LPIPS mean: {lpips_mean:.4f}, GS score: {gs_score:.4f}")
            return gs_score

        # Signal that initialization is complete
        result_queue.put(("READY", gpu_id))

        # Process tasks from queue
        while True:
            task = task_queue.get()

            # Check for shutdown signal
            if task is None:
                print(f"[Process {os.getpid()}] Received shutdown signal")
                break

            camera_trajectory = None
            if len(task) == 4:
                batch_idx, video_frames, prompt, save_dir = task
                use_lpips = False
            elif len(task) == 5:
                batch_idx, video_frames, prompt, save_dir, use_lpips = task
            elif len(task) == 6:
                batch_idx, video_frames, prompt, save_dir, use_lpips, camera_trajectory = task
            else:
                raise ValueError(f"Unexpected task format: {task}")

            try:
                print(f"[Process {os.getpid()}] Processing batch {batch_idx} on GPU {gpu_id}")

                # Convert PIL images to tensor
                frames_tensors = []
                for img in video_frames:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Convert to tensor (C, H, W) in range [0, 1]
                    frame_tensor = torch_local.from_numpy(np.array(img)).float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
                    frames_tensors.append(frame_tensor)

                # Stack to (T, C, H, W)
                video_tensor = torch_local.stack(frames_tensors, dim=0).to(device)

                gs_video, meta_view, camera_motion_score, trajectory_comparison_image = reward_3d_backend(
                    video_tensor,
                    camera_trajectory=camera_trajectory,
                )

                # Save gs_video and meta_view
                print(f"[Process {os.getpid()}] Saving GS video and meta view for batch {batch_idx}")
                import cv2 as cv2_local

                # Save GS video
                gs_video_path = os.path.join(save_dir, f"batch_{batch_idx}_gs_video.mp4")
                gs_frames_np = (gs_video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                height, width = gs_frames_np.shape[1:3]
                fourcc = cv2_local.VideoWriter_fourcc(*'mp4v')
                gs_writer = cv2_local.VideoWriter(gs_video_path, fourcc, 24, (width, height))
                for frame in gs_frames_np:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2_local.cvtColor(frame, cv2_local.COLOR_RGB2BGR)
                    gs_writer.write(frame_bgr)
                gs_writer.release()
                print(f"[Process {os.getpid()}] Saved GS video to {gs_video_path}")

                # Save meta view (single image, not video)
                meta_view_path = os.path.join(save_dir, f"batch_{batch_idx}_meta_view.png")
                meta_view_np = (meta_view.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                meta_view_bgr = cv2_local.cvtColor(meta_view_np, cv2_local.COLOR_RGB2BGR)
                cv2_local.imwrite(meta_view_path, meta_view_bgr)
                print(f"[Process {os.getpid()}] Saved meta view to {meta_view_path}")

                trajectory_comparison_path = ""
                if trajectory_comparison_image is not None:
                    trajectory_comparison_path = os.path.join(save_dir, f"batch_{batch_idx}_trajectory_comparison.png")
                    trajectory_comparison_bgr = cv2_local.cvtColor(trajectory_comparison_image, cv2_local.COLOR_RGB2BGR)
                    cv2_local.imwrite(trajectory_comparison_path, trajectory_comparison_bgr)
                    print(f"[Process {os.getpid()}] Saved trajectory comparison to {trajectory_comparison_path}")

                # Debug: Check tensor properties
                print(f"[Process {os.getpid()}] gs_video shape: {gs_video.shape}, dtype: {gs_video.dtype}, range: [{gs_video.min():.3f}, {gs_video.max():.3f}]")
                print(f"[Process {os.getpid()}] meta_view shape: {meta_view.shape}, dtype: {meta_view.dtype}, range: [{meta_view.min():.3f}, {meta_view.max():.3f}]")

                # Score gs_video with gs_scorer or LPIPS
                print(f"[Process {os.getpid()}] Scoring GS video for batch {batch_idx} (lpips={use_lpips})")
                if use_lpips:
                    gs_score = compute_lpips_gs_score(gs_video, video_tensor)
                else:
                    gs_scores = gs_scorer([prompt], [gs_video])
                    gs_score = gs_scores[0] if gs_scores else 0.0

                # Score meta_view with meta_scorer (single image - Qwen or OpenAI)
                # Need to add batch dimension for image: (3, H, W) -> (1, 3, H, W)
                print(f"[Process {os.getpid()}] Scoring meta view for batch {batch_idx}")
                meta_view_batch = meta_view.unsqueeze(0)  # (1, 3, H, W)
                meta_scores = meta_scorer([prompt], [meta_view_batch])
                meta_score = meta_scores[0] if meta_scores else 0.0

                # Paper setting: R_3D = S_meta + S_recon + S_traj, each bounded in [0, 1].
                gs_score = float(np.clip(gs_score, 0.0, 1.0))
                meta_score = float(np.clip(meta_score, 0.0, 1.0))
                camera_motion_score = float(np.clip(camera_motion_score, 0.0, 1.0))
                final_score = gs_score + meta_score + camera_motion_score

                print(f"[Process {os.getpid()}] Batch {batch_idx} completed:")
                print(f"  GS score: {gs_score:.3f}, Meta score: {meta_score:.3f}, Camera motion: {camera_motion_score:.3f}, Final: {final_score:.3f}")

                result_queue.put((
                    batch_idx,
                    gs_score,
                    meta_score,
                    camera_motion_score,
                    final_score,
                    gs_video_path,
                    meta_view_path,
                    trajectory_comparison_path,
                ))

            except Exception as e:
                print(f"[Process {os.getpid()}] Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

                # Try to recover CUDA state
                try:
                    torch_local.cuda.empty_cache()
                    torch_local.cuda.synchronize()
                except:
                    pass

                result_queue.put((batch_idx, 0.0, 0.0, 0.0, 0.0, "", "", ""))

    except Exception as e:
        print(f"[Process {os.getpid()}] Fatal error in worker process: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(("ERROR", gpu_id))


class MultiGPUReward3DManager:
    """Manager for multi-GPU 3D reward evaluation."""

    def __init__(self, model_name=DEFAULT_RECONSTRUCTION_MODEL, scorer_type="qwen", use_lpips=False):
        self.model_name = model_name
        self.scorer_type = scorer_type
        self.use_lpips = use_lpips
        self.num_gpus = 0
        self.processes = []
        self.task_queues = []
        self.result_queue = None
        self.current_gpu_index = 0  # For round-robin assignment
        self.call_counter = 0  # Track number of calls for logging
        self.last_results = None

    def initialize(self):
        if not torch.cuda.is_available():
            print("CUDA not available. Cannot run the 3D reward backend.")
            return

        self.num_gpus = torch.cuda.device_count()
        print(f"Initializing the 3D reward backend on {self.num_gpus} GPUs")

        # Create shared result queue
        self.result_queue = Queue()

        # Create a task queue and worker process for each GPU
        for gpu_id in range(self.num_gpus):
            task_queue = Queue()
            self.task_queues.append(task_queue)

            # Start worker process
            process = Process(
                target=reward_3d_worker_process,
                args=(gpu_id, self.model_name, self.scorer_type, task_queue, self.result_queue)
            )
            process.start()
            self.processes.append(process)
            print(f"Started worker process for GPU {gpu_id} (PID: {process.pid})")

        # Wait for all workers to finish initialization
        ready_count = 0
        while ready_count < self.num_gpus:
            msg_type, gpu_id = self.result_queue.get()
            if msg_type == "READY":
                print(f"Worker for GPU {gpu_id} is ready")
                ready_count += 1
            elif msg_type == "ERROR":
                print(f"Worker for GPU {gpu_id} failed to initialize")

        print(f"Multi-GPU 3D reward backend initialized with {self.num_gpus} worker processes")

    def compute_batch_scores(self, batch_videos, batch_prompts, camera_trajectories=None, use_lpips=None):
        """
        Compute scores for a batch with load balancing across GPUs

        Args:
            batch_videos: List of videos, each video is a list of frame bytes (JPEG)
            batch_prompts: List of text prompts (length = batch_size)
            camera_trajectories: Optional list of camera trajectories (length = batch_size)
            use_lpips: Use LPIPS to score GS video instead of Qwen3-VL

        Returns:
            List of final scores (length = batch_size)
        """
        if use_lpips is None:
            use_lpips = self.use_lpips
        if not self.processes:
            print("Error: 3D reward worker processes not initialized")
            return [0.0] * len(batch_videos)

        batch_size = len(batch_videos)
        results = {
            'final_scores': [0.0] * batch_size,
            'gs_scores': [0.0] * batch_size,
            'meta_scores': [0.0] * batch_size,
            'camera_motion_scores': [0.0] * batch_size,
        }

        # Create output directory for logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = f"logs/reward_3d/call_{self.call_counter}_{timestamp}"
        os.makedirs(batch_dir, exist_ok=True)
        self.call_counter += 1

        if camera_trajectories is None:
            camera_trajectories = [None] * batch_size
        elif not isinstance(camera_trajectories, list):
            camera_trajectories = [camera_trajectories] * batch_size
        elif len(camera_trajectories) != batch_size:
            if len(camera_trajectories) < batch_size:
                camera_trajectories = camera_trajectories + [None] * (batch_size - len(camera_trajectories))
            else:
                camera_trajectories = camera_trajectories[:batch_size]

        # Prepare tasks
        tasks = []
        for batch_idx, (video_frames, prompt, camera_trajectory) in enumerate(
            zip(batch_videos, batch_prompts, camera_trajectories)
        ):
            try:
                # Convert frame bytes to PIL Images
                frames = [Image.open(BytesIO(frame_bytes)).convert('RGB') for frame_bytes in video_frames]
                print(f"Preparing batch {batch_idx}: {len(frames)} frames")
                tasks.append((batch_idx, frames, prompt, use_lpips, camera_trajectory))
            except Exception as e:
                print(f"Error preparing batch {batch_idx}: {e}")
                # Will use default score 0.0

        # Distribute tasks to workers using round-robin
        for batch_idx, frames, prompt, task_use_lpips, camera_trajectory in tasks:
            gpu_idx = self.current_gpu_index % self.num_gpus
            print(f"Assigning batch {batch_idx} to GPU {gpu_idx}")
            # Pass batch_dir so worker can save videos
            self.task_queues[gpu_idx].put(
                (batch_idx, frames, prompt, batch_dir, task_use_lpips, camera_trajectory)
            )
            self.current_gpu_index += 1

        # Collect results
        completed = 0
        per_video_results = []

        while completed < len(tasks):
            batch_idx, gs_score, meta_score, camera_motion_score, final_score, gs_video_path, meta_view_path, trajectory_comparison_path = self.result_queue.get()
            results['gs_scores'][batch_idx] = gs_score
            results['meta_scores'][batch_idx] = meta_score
            results['camera_motion_scores'][batch_idx] = camera_motion_score
            results['final_scores'][batch_idx] = final_score
            completed += 1

            # Store result for logging
            per_video_results.append({
                "video_id": batch_idx,
                "prompt": batch_prompts[batch_idx],
                "gs_score": float(gs_score),
                "meta_score": float(meta_score),
                "camera_motion_score": float(camera_motion_score),
                "final_score": float(final_score),
                "lpips": bool(use_lpips),
                "gs_video_path": gs_video_path,
                "meta_view_path": meta_view_path,
                "trajectory_comparison_path": trajectory_comparison_path,
            })

            print(f"Received result for batch {batch_idx}: GS={gs_score:.3f}, Meta={meta_score:.3f}, Motion={camera_motion_score:.3f}, Final={final_score:.3f}")
            if gs_video_path:
                print(f"  GS video: {gs_video_path}")
            if meta_view_path:
                print(f"  Meta view: {meta_view_path}")
            if trajectory_comparison_path:
                print(f"  Trajectory compare: {trajectory_comparison_path}")

        # Save results to JSON file
        results_path = os.path.join(batch_dir, "reward_3d_scores.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(per_video_results, f, ensure_ascii=False, indent=2)
        print(f"Saved 3D reward scores to {results_path}")

        self.last_results = {
            "batch_dir": batch_dir,
            "final_scores": list(results["final_scores"]),
            "gs_scores": list(results["gs_scores"]),
            "meta_scores": list(results["meta_scores"]),
            "camera_motion_scores": list(results["camera_motion_scores"]),
            "per_video_results": per_video_results,
        }

        return results['final_scores']

    def shutdown(self):
        """Shutdown all worker processes"""
        print("Shutting down 3D reward worker processes...")

        # Send shutdown signal to all workers
        for task_queue in self.task_queues:
            task_queue.put(None)

        # Wait for all processes to finish
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                print(f"Force terminating process {process.pid}")
                process.terminate()
                process.join()

        print("All 3D reward worker processes shut down")
