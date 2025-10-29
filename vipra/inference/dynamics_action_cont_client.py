import logging
import time
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional orjson (faster); will fallback if missing
try:
    import orjson

    def _json_dumps(obj) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)

    _USE_ORJSON = True
except Exception:
    import json

    def _json_dumps(obj) -> str:
        return json.dumps(obj)

    _USE_ORJSON = False


class APIError(Enum):
    CONNECTION_ERROR = 1
    TIMEOUT = 2
    SERVER_ERROR = 3
    CLIENT_ERROR = 4


class ViPRAClient:
    """Client for ViPRA Inference Server with dual request modes (JSON or Bytes)."""

    def __init__(
        self,
        server_url: str = "http://localhost:8005",
        timeout: Tuple[float, float] = (1.0, 2.0),  # (connect, read)
        max_retries: int = 3,
        pool_maxsize: int = 32,
        image_size: int = 256,  # match server
        jpeg_quality: int = 100,  # used in bytes mode
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.image_size = image_size
        self.jpeg_quality = jpeg_quality

        self.logger = logging.getLogger("ViPRAClient")
        self.session = requests.Session()
        self.session.trust_env = False  # avoid proxy env lookups

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            backoff_factor=0.2,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Default headers for JSON mode; in bytes mode requests sets multipart headers automatically
        self.json_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }

        self._current_task: Optional[str] = "unspecified_task"

    # ----------------- Public API -----------------

    def reset_policy(self, task_description: str) -> bool:
        """Reset policy/task on server."""
        self.current_task = task_description
        endpoint = f"{self.server_url}/reset"
        payload = {"task_description": task_description}

        try:
            if _USE_ORJSON:
                resp = self.session.post(
                    endpoint, data=_json_dumps(payload), headers=self.json_headers, timeout=self.timeout
                )
            else:
                resp = self.session.post(endpoint, json=payload, headers=self.json_headers, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            self._handle_error(e, "reset")
            return False  # usually not reached; _handle_error raises

    def get_action(
        self,
        image_list: List[np.ndarray],
        task_description: Optional[str] = None,
        mode: str = "json",  # "json" or "bytes"
    ) -> np.ndarray:
        """
        Get raw actions from the server.
        - mode="json": posts /step with nested lists (baseline)
        - mode="bytes": posts /step_bytes with JPEG uploads (faster)
        """
        if mode not in ("json", "bytes"):
            raise ValueError("mode must be 'json' or 'bytes'")
        if len(image_list) < 2:
            raise ValueError("Expect at least two images")

        if mode == "json":
            return self._get_action_json(image_list, task_description)
        else:
            return self._get_action_bytes(image_list, task_description)

    # ----------------- Mode: JSON -----------------

    def _get_action_json(self, image_list: List[np.ndarray], task_description: Optional[str]) -> np.ndarray:
        images_payload = [self._prep_image_for_json(img) for img in image_list]
        payload = {"image": images_payload, "task_description": task_description or self.current_task}
        endpoint = f"{self.server_url}/step"

        t0 = time.perf_counter()
        try:
            if _USE_ORJSON:
                resp = self.session.post(
                    endpoint, data=_json_dumps(payload), headers=self.json_headers, timeout=self.timeout
                )
            else:
                resp = self.session.post(endpoint, json=payload, headers=self.json_headers, timeout=self.timeout)

            ttfb = resp.elapsed.total_seconds()
            t1 = time.perf_counter()
            resp.raise_for_status()

            parsed = resp.json()
            t2 = time.perf_counter()
            raw_actions = np.asarray(parsed["raw_actions"], dtype=np.float32)
            t3 = time.perf_counter()

            self.logger.info(
                "Action latency [JSON]: end2end=%.3fs | ttfb=%.3fs | recv=%.3fs | json=%.3fs | np=%.3fs",
                (t3 - t0),
                ttfb,
                (t1 - t0) - ttfb,
                (t2 - t1),
                (t3 - t2),
            )
            return raw_actions

        except requests.exceptions.RequestException as e:
            self._handle_error(e, "step(JSON)")
            return np.empty((0, 0), dtype=np.float32)

    def _prep_image_for_json(self, img: np.ndarray) -> list:
        self._validate_rgb8(img)
        if (img.shape[0], img.shape[1]) != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
        return img.tolist()  # costly—baseline mode

    # ----------------- Mode: BYTES (multipart) -----------------

    def _get_action_bytes(self, image_list: List[np.ndarray], task_description: Optional[str]) -> np.ndarray:
        # Server expects exactly two names: image0, image1 (adjust if you changed the server)
        img0_bytes = self._prep_jpeg(image_list[0])
        img1_bytes = self._prep_jpeg(image_list[1])

        files = [
            ("image0", ("img0.jpg", img0_bytes, "image/jpeg")),
            ("image1", ("img1.jpg", img1_bytes, "image/jpeg")),
        ]
        data = {} if task_description is None else {"task_description": task_description}
        endpoint = f"{self.server_url}/step_bytes"

        t0 = time.perf_counter()
        try:
            resp = self.session.post(endpoint, files=files, data=data, timeout=self.timeout)
            ttfb = resp.elapsed.total_seconds()
            t1 = time.perf_counter()
            resp.raise_for_status()

            parsed = resp.json()
            t2 = time.perf_counter()
            raw_actions = np.asarray(parsed["raw_actions"], dtype=np.float32)
            t3 = time.perf_counter()

            self.logger.info(
                "Action latency [BYTES]: end2end=%.3fs | ttfb=%.3fs | recv=%.3fs | json=%.3fs | np=%.3fs",
                (t3 - t0),
                ttfb,
                (t1 - t0) - ttfb,
                (t2 - t1),
                (t3 - t2),
            )
            return raw_actions

        except requests.exceptions.RequestException as e:
            self._handle_error(e, "step(BYTES)")
            return np.empty((0, 0), dtype=np.float32)

    def _prep_jpeg(self, img: np.ndarray) -> bytes:
        self._validate_rgb8(img)
        if (img.shape[0], img.shape[1]) != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
        # OpenCV expects BGR for encoding
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [
                cv2.IMWRITE_JPEG_QUALITY,
                int(self.jpeg_quality),  # e.g., 95
                cv2.IMWRITE_JPEG_OPTIMIZE,
                1,
            ],  # slightly smaller file
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    # ----------------- Utilities -----------------

    @staticmethod
    def _validate_rgb8(img: np.ndarray):
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
            raise ValueError("Each image must be a uint8 numpy array")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Each image must be HxWx3 (RGB)")

    def _handle_error(self, error: requests.exceptions.RequestException, operation: str):
        if isinstance(error, requests.exceptions.Timeout):
            etype = APIError.TIMEOUT
        elif isinstance(error, requests.exceptions.ConnectionError):
            etype = APIError.CONNECTION_ERROR
        else:
            status = getattr(error.response, "status_code", None)
            if status is None:
                etype = APIError.CONNECTION_ERROR
            elif 500 <= status < 600:
                etype = APIError.SERVER_ERROR
            elif 400 <= status < 500:
                etype = APIError.CLIENT_ERROR
            else:
                etype = APIError.CONNECTION_ERROR
        self.logger.warning("%s failed with %s (%s)", operation, etype.name, str(error))
        raise RuntimeError(f"{operation} failed: {etype.name}") from error

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.uint8)

    @property
    def current_task(self) -> Optional[str]:
        return self._current_task

    @current_task.setter
    def current_task(self, value: Optional[str]):
        self._current_task = value or "unspecified_task"


# Example usage
if __name__ == "__main__":
    import json
    import os
    import re
    from collections import deque

    import matplotlib.pyplot as plt
    from PIL import Image
    from tqdm import tqdm

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create agent instance
    agent = ViPRAClient(server_url="http://localhost:8005", timeout=45, max_retries=5)  # Replace with actual server IP

    # Initialize policy
    pred_horizon = 14
    json_path = "/fsx2/shared/sroutray/LIBEROv2/libero_10_modified/libero_10_dynamics14_norm_v2.jsonl"
    image_root = "/fsx2/shared/sroutray/LIBEROv2"
    action_norm_stats_path = "/fsx2/shared/sroutray/LIBEROv2/libero_10_modified/normalization_stats_v2.json"
    traj_id = "ep00118/"
    traj_data = []
    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if traj_id in data["id"]:
                traj_data.append(data)
    traj_data.sort(key=lambda x: x["id"])
    instruction = traj_data[0]["instruction"]
    task_description = re.findall(r"`(.*?)`", instruction)[0]

    with open(action_norm_stats_path, "r") as f:
        action_norm_stats = json.load(f)

    agent.reset_policy(task_description)
    curr_action_sequence = None
    curr_action_index = 0
    state_buffer = deque(maxlen=3)

    pred_action_log = []
    real_action_log = []
    # Open loop control
    for step, data in tqdm(enumerate(traj_data)):
        img_path = traj_data[step]["image"][-1]
        img = Image.open(os.path.join(image_root, img_path))
        # import ipdb; ipdb.set_trace()
        img_arr = np.array(img)
        state_buffer.append(img_arr)
        if curr_action_sequence is None or curr_action_index >= pred_horizon:
            # Get action from agent
            pred_action_sequence = agent.get_action([state_buffer[0], state_buffer[-1]], task_description, mode="bytes")
            pred_action_sequence = np.array(pred_action_sequence)
            curr_action_sequence = pred_action_sequence
            curr_action_index = 0
        mean = np.array(action_norm_stats["mean"])
        std = np.array(action_norm_stats["std"])
        pred_action = curr_action_sequence[curr_action_index]
        real_action = np.array([float(x) for x in data["raw_action"]])
        pred_action = pred_action * std + mean
        pred_action_log.append(pred_action)
        real_action_log.append(real_action)
        curr_action_index += 1
    pred_action_log = np.array(pred_action_log)
    real_action_log = np.array(real_action_log)
    num_actions = len(pred_action_log[0])

    # Create a single figure with subplots arranged horizontally
    fig, axes = plt.subplots(nrows=1, ncols=num_actions, figsize=(5 * num_actions, 5))

    for adim in range(num_actions):
        ax = axes[adim] if num_actions > 1 else axes  # Handle single subplot case
        ax.plot(pred_action_log[:, adim], label=f"pred_{adim}", color="r")
        ax.plot(real_action_log[:, adim], label=f"real_{adim}", color="b")
        ax.legend()
        ax.set_title(f"Action Dimension {adim}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action Value")

    # Adjust layout and save the combined figure
    plt.tight_layout()
    plt.savefig("actions_log.png")

    # Save action logs
    # os.makedirs("logs", exist_ok=True)
    # np.save("logs/ours_pred_action_log.npy", pred_action_log)
    # np.save("logs/ours_real_action_log.npy", real_action_log)
