"""Multi-GPU backend for the general reward."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import threading

import hpsv2
import torch
from PIL import Image


class GeneralRewardInstance:
    """Single general-reward worker pinned to one GPU."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self._busy = False
        self._lock = threading.Lock()

    def load_model(self):
        print(f"General reward worker {self.gpu_id} initialized")

    def set_busy(self, busy_state: bool):
        with self._lock:
            self._busy = busy_state

    @torch.no_grad()
    def compute_score(self, images, prompts):
        try:
            self.set_busy(True)
            if torch.cuda.is_available():
                torch.cuda.set_device(self.gpu_id)

            prompt_to_images = {}
            for index, prompt in enumerate(prompts):
                prompt_to_images.setdefault(prompt, []).append((index, images[index]))

            scores = [0.0] * len(images)
            for prompt, image_group in prompt_to_images.items():
                if len(image_group) > 1:
                    pil_images = [image for _, image in image_group]
                    try:
                        hps_scores = hpsv2.score(pil_images, prompt, hps_version="v2.1")
                        for group_index, (origin_index, _) in enumerate(image_group):
                            scores[origin_index] = float(hps_scores[group_index])
                        torch.cuda.empty_cache()
                    except Exception:
                        for origin_index, _ in image_group:
                            scores[origin_index] = 0.5
                else:
                    origin_index, pil_image = image_group[0]
                    try:
                        hps_scores = hpsv2.score([pil_image], prompt, hps_version="v2.1")
                        scores[origin_index] = float(hps_scores[0])
                    except Exception:
                        scores[origin_index] = 0.5

            return scores
        finally:
            self.set_busy(False)


class MultiGPUGeneralRewardManager:
    """Multi-GPU manager for general reward evaluation."""

    def __init__(self):
        self.instances = []
        self.num_gpus = 0
        self.executor = None

    def initialize(self):
        if not torch.cuda.is_available():
            instance = GeneralRewardInstance(0)
            instance.load_model()
            self.instances = [instance]
            self.num_gpus = 1
            return

        self.num_gpus = torch.cuda.device_count()
        for gpu_id in range(self.num_gpus):
            instance = GeneralRewardInstance(gpu_id)
            instance.load_model()
            self.instances.append(instance)

        self.executor = ThreadPoolExecutor(max_workers=self.num_gpus)

    def compute_batch_scores(self, batch_images, batch_prompts):
        if not self.instances:
            return [0.5] * len(batch_images)
        if self.num_gpus == 1:
            return self._compute_sequential(batch_images, batch_prompts)
        return self._compute_parallel(batch_images, batch_prompts)

    def _compute_sequential(self, batch_images, batch_prompts):
        results = []
        instance = self.instances[0]
        for batch_index, image_bytes in enumerate(batch_images):
            try:
                image = Image.open(BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                results.append(instance.compute_score([image], [batch_prompts[batch_index]])[0])
            except Exception:
                results.append(0.5)
        return results

    def _compute_parallel(self, batch_images, batch_prompts):
        tasks = []
        for batch_index, image_bytes in enumerate(batch_images):
            try:
                image = Image.open(BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                tasks.append((batch_index, image, batch_prompts[batch_index]))
            except Exception:
                tasks.append((batch_index, None, None))

        future_to_batch = {}
        results = [0.5] * len(batch_images)
        for worker_index, (batch_index, image, prompt) in enumerate(tasks):
            if image is None:
                continue
            instance = self.instances[worker_index % self.num_gpus]
            future = self.executor.submit(instance.compute_score, [image], [prompt])
            future_to_batch[future] = batch_index

        for future in as_completed(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                batch_scores = future.result(timeout=60)
                results[batch_index] = batch_scores[0] if batch_scores else 0.5
            except Exception:
                results[batch_index] = 0.5

        return results

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True)
