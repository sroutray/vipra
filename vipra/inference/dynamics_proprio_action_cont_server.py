import os
from sys import flags

# Set JAX memory behavior EARLY (before importing JAX)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")  # optional

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple

import jax
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse, Response
from ml_collections import ConfigDict
from PIL import Image
from policy.dynamics_proprio_action_cont_llama import VideoLLaMAConfig
from pydantic import BaseModel
from samplers.sampler_dynamics_proprio_action_cont import ProprioActionContSampler
from tux import JaxDistributedConfig, define_flags_with_default, set_random_seed

# ---------------- Models ----------------


@dataclass
class ViPRAConfig:
    gpu_id: int = 0
    seed: int = 1234
    init_rng: int = 0
    dtype: str = "bf16"
    ckpt_path: str = (
        "trainstate_params::/mnt/home/sroutray/vipra_project/vipra/outputs/vipra_finetune_libero_long_ph7_ca14_run2/streaming_train_state"
    )
    policy_setup: str = "dynamics_action_cont"

    mesh_dim: str = "1,1,1,1"
    load_llama_config: str = "7b"
    update_llama_config: str = (
        "dict(action_dims=7,proprio_dims=15,latent_action_vocab_size=9,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=1024,scan_key_chunk_size=1024,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)"
    )

    image_size: int = 256
    image_aug: bool = False
    vertical_flip: bool = False
    proprio_dims: int = 15
    proprio_history_len: int = 7
    action_chunk_size: int = 14
    action_dims: int = 7
    tokens_per_delta: int = 16
    eola_token: int = 8
    state_buffer_size: int = 3
    num_inference_steps: int = 10

    vocab_file: str = "/mnt/home/sroutray/vipra_project/vipra/lwm/tokenizer.model"
    vqgan_checkpoint: str = "/mnt/home/sroutray/vipra_project/vipra/lwm/vqgan"
    load_checkpoint: str = ckpt_path

    tokenizer: ConfigDict = VideoLLaMAConfig.get_tokenizer_config(updates=dict(vocab_file=vocab_file))
    llama: ConfigDict = VideoLLaMAConfig.get_default_config()

    def update(self, **kwargs):
        if len(kwargs) == 1 and isinstance(list(kwargs.values())[0], ViPRAConfig):
            other_config = list(kwargs.values())[0]
            for k, v in vars(other_config).items():
                setattr(self, k, v)
        else:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise AttributeError(f"Config has no attribute named '{k}'")


class ViPRAInference:
    def __init__(self, task_description: str, **kwargs) -> None:
        flags = ViPRAConfig()
        flags.update(**kwargs)
        self.flags = flags

        self.model = ProprioActionContSampler(FLAGS=flags)

        self.image_size = flags.image_size
        self.policy_setup = flags.policy_setup
        self.rng = jax.random.PRNGKey(flags.init_rng)
        self.task_description = task_description

    def reset(self, task_description: str) -> None:
        self.task_description = task_description

    def step(
        self, image_list: List[np.ndarray], proprio_list: List[np.ndarray], task_description: Optional[str] = None
    ) -> np.ndarray:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        image_prompt = []
        for image in image_list:
            # Avoid extra copies: ascontiguous + dtype assurance
            img = np.ascontiguousarray(image, dtype=np.uint8)
            pil_img = Image.fromarray(img)
            if pil_img.size != (self.image_size, self.image_size):
                pil_img = pil_img.resize((self.image_size, self.image_size), resample=Image.Resampling.LANCZOS)
            image_prompt.append(pil_img)

        proprio_prompt = np.ascontiguousarray(proprio_list, dtype=np.float32)

        prompts = [{"image": image_prompt, "question": task_description, "proprio": proprio_prompt}]
        action_output = self.model(prompts)
        return action_output.astype(np.float32, copy=False)


# -------------- Schemas --------------


class StepRequest(BaseModel):
    image: List[List[List[List[int]]]]  # list of images (H,W,3), each a nested list
    proprio: List[List[float]]  # list of proprioceptive states (D_proprio), each a list
    task_description: Optional[str] = None


class ResetRequest(BaseModel):
    task_description: str


# -------------- App init --------------

app = FastAPI(
    title="FM Action Decoder Inference API",
    default_response_class=ORJSONResponse,  # faster JSON
)

inference_model = None

JaxDistributedConfig.initialize(JaxDistributedConfig.get_default_config())
set_random_seed(1234)


@app.on_event("startup")
def startup_event():
    global inference_model
    inference_model = ViPRAInference(task_description="")

    # Warm-up compile (keep it as small as possible)
    try:
        s = inference_model.image_size
        d0 = np.zeros((s, s, 3), dtype=np.uint8)
        d1 = np.ones((s, s, 3), dtype=np.uint8)
        ph = inference_model.flags.proprio_history_len
        pd = inference_model.flags.proprio_dims
        p0 = [np.zeros((pd,), dtype=np.float32) for _ in range(ph)]
        _ = inference_model.step([d0, d1], p0, "compilation_trigger")
        inference_model.reset("default task")
        logging.info("JAX model compilation completed successfully")
    except Exception as e:
        logging.exception("Model compilation failed during startup")
        raise RuntimeError("Model compilation failed during startup") from e


# -------------- Routes --------------


# Make this a sync route so FastAPI runs it in a threadpool (no event-loop blocking)
@app.post("/step")
def step_endpoint(request: StepRequest):
    global inference_model
    if inference_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    try:
        # Convert nested lists -> contiguous uint8 arrays
        image_list = [np.ascontiguousarray(x, dtype=np.uint8) for x in request.image]
        proprio_list = [np.ascontiguousarray(x, dtype=np.float32) for x in request.proprio]

        actions = inference_model.step(
            image_list=image_list, proprio_list=proprio_list, task_description=request.task_description
        )

        # JSON (fast path with ORJSON)
        return {"raw_actions": actions.tolist()}
    except Exception as e:
        logging.exception("Error processing request")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/reset")
def reset_endpoint(request: ResetRequest):
    global inference_model
    if inference_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    try:
        inference_model.reset(task_description=request.task_description)
        return {"status": "success", "message": "Model reset successfully"}
    except Exception as e:
        logging.exception("Error resetting model")
        raise HTTPException(status_code=500, detail=f"Error resetting model: {str(e)}")


def _imgfile_to_uint8(upload: UploadFile, target_size: int) -> np.ndarray:
    # read bytes -> PIL -> resize -> np.uint8 RGB (H, W, 3)
    data = upload.file.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), resample=Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _load_proprio_raw(upload: UploadFile, shape: Tuple[int, int], dtype=np.float32) -> np.ndarray:
    T, D = shape
    nbytes = T * D * np.dtype(dtype).itemsize
    upload.file.seek(0)
    data = upload.file.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"Expected {nbytes} bytes, got {len(data)}")
    return np.frombuffer(data, dtype=dtype).reshape(T, D)


@app.post("/step_bytes")
def step_bytes_endpoint(
    image0: UploadFile = File(...),
    image1: UploadFile = File(...),
    proprio: UploadFile = File(...),
    task_description: Optional[str] = Form(None),
):
    global inference_model
    if inference_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    try:
        s = inference_model.image_size
        img0 = _imgfile_to_uint8(image0, s)
        img1 = _imgfile_to_uint8(image1, s)
        ph = inference_model.flags.proprio_history_len
        pd = inference_model.flags.proprio_dims
        proprio_arr = _load_proprio_raw(proprio, (ph, pd), dtype=np.float32)
        proprio_list = [proprio_arr[i] for i in range(ph)]

        actions = inference_model.step(
            image_list=[img0, img1],
            proprio_list=proprio_list,
            task_description=task_description,
        )
        return {"raw_actions": actions.tolist()}
    except Exception as e:
        logging.exception("Error processing request")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    # Use faster loop & HTTP parser
    uvicorn.run(app, host="0.0.0.0", port=8006, loop="uvloop", http="httptools")
