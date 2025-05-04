"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import jax
import tensorflow_datasets as tfds
import tqdm
import mediapy
import numpy as np
import draccus
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from octo.model.octo_model import OctoModel

# === Server Interface ===
class OctoServer:
    def __init__(self, octo_path: Union[str, Path]) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.octo_path = octo_path

        # Load VLA Model using HF AutoClasses
        self.model = OctoModel.load_pretrained(octo_path)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            task = self.model.create_tasks(texts=[instruction]) 
            observation = {
                'image_primary': image,
                'timestep_pad_mask': np.full((1, image.shape[1]), True, dtype=bool)
            }
    
            actions = self.model.sample_actions(
                observation, 
                task, 
                unnormalization_statistics=self.model.dataset_statistics["berkeley_autolab_ur5"]["action"], 
                rng=jax.random.PRNGKey(0)
            )
            
            actions = np.array(actions)[0]
            if double_encode:
                return JSONResponse(json_numpy.dumps(actions))
            else:
                return JSONResponse(actions)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    octo_path: Union[str, Path] = "hf://rail-berkeley/octo-base-1.5"               # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OctoServer(cfg.octo_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()