import pickle
import os
import argparse
import traceback
import signal
import sys
from reward_server.reward_3d import MultiGPUReward3DManager

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9588))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

from flask import Flask, request, Blueprint

root = Blueprint("root", __name__)

reward_3d_manager = None

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global reward_3d_manager
    print("\nReceived shutdown signal. Cleaning up...")
    if reward_3d_manager:
        reward_3d_manager.shutdown()
    sys.exit(0)

def create_app(scorer_type='qwen', use_lpips=True):
    global reward_3d_manager
    print(f"Initializing multi-GPU 3D reward server (scorer: {scorer_type}, lpips: {use_lpips})...")
    reward_3d_manager = MultiGPUReward3DManager(scorer_type=scorer_type, use_lpips=use_lpips)
    reward_3d_manager.initialize()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"])
def inference():
    print(f"received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        # expects a dict with "videos" and "prompts"
        # videos: List[List[bytes]] - outer list is batch_size, inner list is frames per video
        # prompts: List[str] - text prompts for each video
        data = pickle.loads(data)

        batch_videos = data["videos"]  # List[List[bytes]]
        batch_prompts = data["prompts"]  # List[str]
        batch_camera_trajectories = data.get("camera_trajectories")
        batch_size = len(batch_videos)
        print(f"Got batch of size {batch_size} for 3D reward evaluation")

        global reward_3d_manager
        if reward_3d_manager is None:
            print("Error: 3D reward server is not initialized")
            outputs = [0.0] * batch_size
            details = None
        else:
            outputs = reward_3d_manager.compute_batch_scores(
                batch_videos,
                batch_prompts,
                camera_trajectories=batch_camera_trajectories,
            )
            details = getattr(reward_3d_manager, "last_results", {}).get("per_video_results")

        print(f"3D reward batch processing results: {outputs}")

        response = {"outputs": outputs, "details": details}

        # returns: a dict with "outputs"
        # outputs: List of scores (float values) with length = batch_size
        response = pickle.dumps(response)

        returncode = 200
    except Exception:
        response = traceback.format_exc()
        print(response)
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


HOST = "127.0.0.1"
PORT = 8089  # Default port used by flow_grpo/reward-server integration

if __name__ == "__main__":
    # CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This ensures each subprocess gets a fresh Python interpreter and CUDA context
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Start method already set
        pass

    parser = argparse.ArgumentParser(description="3D reward server")
    parser.add_argument(
        "--host",
        default=os.getenv("REWARD_3D_HOST", HOST),
        help="Host to bind (env: REWARD_3D_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("REWARD_3D_PORT", PORT)),
        help="Port to listen on (env: REWARD_3D_PORT)",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        choices=['qwen', 'openai'],
        default=os.getenv("REWARD_3D_SCORER", "qwen"),
        help="Scorer type for meta-view and reconstruction evaluation (env: REWARD_3D_SCORER)",
    )
    parser.add_argument(
        "--lpips",
        dest="lpips",
        action="store_true",
        help="Use LPIPS for reconstruction scoring instead of Qwen3-VL",
    )
    parser.add_argument(
        "--no-lpips",
        dest="lpips",
        action="store_false",
        help="Disable LPIPS reconstruction scoring and use Qwen3-VL instead",
    )
    parser.set_defaults(
        lpips=os.getenv("REWARD_3D_USE_LPIPS", "1").strip().lower() not in {"0", "false", "no"}
    )
    args = parser.parse_args()

    create_app(scorer_type=args.scorer, use_lpips=args.lpips).run(args.host, args.port)
