#!/usr/bin/env python3
import argparse
import os
import pickle
import traceback

from flask import Blueprint, Flask, request

from reward_server.general_reward import MultiGPUGeneralRewardManager

root = Blueprint("root", __name__)
general_reward_manager = None


def create_app():
    global general_reward_manager
    general_reward_manager = MultiGPUGeneralRewardManager()
    general_reward_manager.initialize()
    app = Flask(__name__)
    app.register_blueprint(root)
    return app


@root.route("/", methods=["POST"])
def inference():
    data = request.get_data()
    try:
        payload = pickle.loads(data)
        batch_images = payload["images"]
        batch_prompts = payload["prompts"]
        batch_size = len(batch_images)

        global general_reward_manager
        if general_reward_manager is None:
            outputs = [0.5] * batch_size
        else:
            outputs = general_reward_manager.compute_batch_scores(batch_images, batch_prompts)

        return pickle.dumps({"outputs": outputs}), 200
    except Exception:
        return traceback.format_exc().encode("utf-8"), 500


HOST = "127.0.0.1"
PORT = 8090


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General reward server")
    parser.add_argument("--host", default=os.getenv("GENERAL_REWARD_HOST", HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("GENERAL_REWARD_PORT", PORT)))
    args = parser.parse_args()
    create_app().run(args.host, args.port)
