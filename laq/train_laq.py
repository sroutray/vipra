import logging
import os

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from configs.config import DATA_CFG, DATASET_CFG, MODEL_CFG, TRAIN_CFG, VAL_DATASET_CFG
from model.data import VideoDatasetCoTrain
from model.latent_action_quantization import LAQ
from model.trainer import LAQTrainer
from model.utils import setup_logging

accelerator_kwargs = {"log_with": "wandb", "find_unused_parameters": True}
find_unused_params = accelerator_kwargs.pop("find_unused_parameters", True)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_params)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerator_kwargs)

setup_logging(accelerator)
logger = logging.getLogger(__name__)

model = LAQ(**MODEL_CFG)

train_dataset = VideoDatasetCoTrain(
    DATASET_CFG,
    image_size=DATA_CFG["image_size"],
    max_frames=DATA_CFG["max_frames"],
    seed=12345,  # Different seed for training dataset
)
val_dataset = VideoDatasetCoTrain(
    VAL_DATASET_CFG,
    image_size=DATA_CFG["image_size"],
    max_frames=DATA_CFG["max_frames"],
    seed=123,
)

save_folder = "results/"
run_name = "exp_laq_cotrain_flow_warmup_rope_fstdecv2_filmv2_peg_abs_run1"
run_id = ""
save_postfix = "_" + run_id if run_id else ""
results_folder = os.path.join(save_folder, run_name + save_postfix)

wandb_mode = "online"
wandb_kwargs = {
    "wandb": {
        "mode": wandb_mode,
        "name": results_folder.split("/")[-1],
        "config": None,
        "id": run_name + "_" + run_id,
        "resume": "allow",
    }
}

# Resume training if a checkpoint exists
ckpt_path = os.path.join(results_folder, "laq_model_latest.pt")
if os.path.exists(ckpt_path):
    print(f"Training will resume from checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

trainer = LAQTrainer(
    model,
    accelerator,
    train_dataset,
    results_folder=results_folder,
    val_dataset=val_dataset,
    resume_checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None,
    wandb_kwargs=wandb_kwargs,
    **TRAIN_CFG,
)

trainer.train()
