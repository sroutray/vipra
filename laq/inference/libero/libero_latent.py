from __future__ import annotations

import concurrent.futures
import glob
import json
import os

os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["NCCL_TIMEOUT"] = "2147483647"
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from beartype import beartype
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from configs.config import DATA_CFG, DATASET_CFG, MODEL_CFG, TRAIN_CFG
from model.latent_action_quantization import LAQ

# Data configurations
H_P = 1
H_F = 14
LIBERO_CONFIG = DATASET_CFG["libero"]
LIBERO_TASKS = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]

# Model configurations
MODEL_CFG["use_lpips_loss"] = False
MODEL_CFG["use_flow_loss"] = False
MODEL_CFG["encode_deltas"] = False  # Disable delta encoding
LAQ_CKPT = Path(
    "/fsx/sroutray/Work1/vipra/laq_v5/results/exp_laq_cotrain_flow_warmup_rope_fstdecv2_filmv2_peg_abs_run1/laq_model_milestone_240000.pt"
)

# Save configurations
SAVE_ROOT = Path("/fsx2/shared/sroutray/vipra_data/libero/")

_step_num = re.compile(r"step(\d+)$")


@beartype
def discover_libero_sequences(data_root: Path, task: str) -> List[str]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root {data_root} does not exist.")

    task_dir = data_root / f"{task}_modified" / "images"
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory {task_dir} does not exist.")

    video_paths = [str(pth) for pth in task_dir.iterdir() if pth.is_dir()]

    if not video_paths:
        raise ValueError(f"No videos found in task: {task}. Check the dataset structure.")

    return video_paths


def step_key(e: Dict[str, Any]) -> tuple[int, int]:
    m = _step_num.search(e.get("id", ""))
    return (0, int(m.group(1))) if m else (1, 0)  # unknown steps go last


# sort each episode by numeric step using parallel processing
def sort_episode_kv(item):
    ep_id, entries = item
    return ep_id, sorted(entries, key=step_key)


def group_by_episode(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    episodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in data:
        id_ = entry.get("id", "")
        parts = id_.rstrip("/").split("/")
        if len(parts) < 2:
            logger.warning(f"Unexpected id format: {id_}, skipping entry.")
            continue
        ep_id = parts[-2]  # exact parent folder (e.g., "ep00000")
        episodes[ep_id].append(entry)

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for ep_id, sorted_data in executor.map(sort_episode_kv, episodes.items()):
            episodes[ep_id] = sorted_data

    return dict(episodes)


def write_global_shard(output_dir: Path, rank: int, entries: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"all_with_latents.rank{rank}.jsonl"
    with open(shard_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    logger.info(f"[rank {rank}] wrote {len(entries)} lines -> {shard_path}")
    return shard_path


def merge_global_shards(output_dir: Path, delete_shards: bool = True) -> Path:
    shard_paths = sorted(glob.glob(str(output_dir / "all_with_latents.rank*.jsonl")))
    assert shard_paths, f"No shard files found in {output_dir}"
    final_path = output_dir / "all_with_latents.jsonl"
    tmp_path = output_dir / "all_with_latents.tmp.jsonl"

    with open(tmp_path, "w") as out:
        for sp in shard_paths:
            with open(sp, "r") as f:
                for line in f:
                    out.write(line)

    os.replace(tmp_path, final_path)  # atomic

    if delete_shards:
        for sp in shard_paths:
            try:
                os.remove(sp)
            except OSError:
                pass

    logger.info(f"[merge] wrote {sum(1 for _ in open(final_path))} lines -> {final_path}")
    return final_path


class SingleEpisodeDataset(Dataset):
    """
    Returns, for every center time t in an episode, a repeat-padded window:
      video: (Win, C, H, W) where Win = H_p + H_f + 1
    Also returns metadata helpful for writing latents back.

    Assumptions:
      - ep_entries are already sorted by step.
      - Each entry has keys: "image" (filepath), "id" (e.g., TASK/epXXXX/stepYYYY).
    """

    def __init__(
        self,
        ep_entries: List[Dict[str, Any]],
        H_p: int,
        H_f: int,
    ):
        self.ep = ep_entries
        self.ids = [e["id"] for e in ep_entries]
        self.paths = [str(e["image"]) for e in ep_entries]
        self.H_p = int(H_p)
        self.H_f = int(H_f)
        self.W = self.H_p + self.H_f + 1
        assert self.W >= 1, "Window size must be >= 1"

        self.N = len(self.paths)
        assert self.N > 0, "Episode is empty"
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) and img.mode != "RGB" else img),
                T.ToImage(),
                T.Resize(DATA_CFG["image_size"]),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self) -> int:
        return self.N

    def _clamp(self, i: int) -> int:
        if i < 0:
            return 0
        if i >= self.N:
            return self.N - 1
        return i

    def _window_indices(self, t: int) -> List[int]:
        # [t-H_p, ..., t, ..., t+H_f] with repeat-padding
        start = t - self.H_p
        return [self._clamp(start + k) for k in range(self.W)]

    def __getitem__(self, t: int) -> Dict[str, Any]:
        idxs = self._window_indices(t)
        window_paths = [self.paths[i] for i in idxs]

        frames: List[torch.Tensor] = []
        for p in window_paths:
            with Image.open(p) as im:
                frames.append(self.transform(im))
        video = torch.stack(frames, dim=0)  # (W, C, H, W)

        # Split window paths into history, present, and future
        history_paths = window_paths[: self.H_p]  # [t-H_p, ..., t-1]
        present_path = window_paths[self.H_p]  # [t]
        future_paths = window_paths[self.H_p + 1 :]  # [t+1, ..., t+H_f]

        return {
            "video": video,  # (W, C, H, W)
            "t": t,  # center timestep
            "id": self.ids[t],  # center id
            "center_path": present_path,
            "history_paths": history_paths,  # list of past frame paths
            "future_paths": future_paths,  # list of future frame paths
            "raw_actions": self.ep[t]["raw_action"],
            "instruction": self.ep[t]["instruction"],
        }


def collate_function(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(batch) > 0, "Empty batch"

    videos = torch.stack([b["video"] for b in batch], dim=0)  # (B, W, C, H, W)
    out: Dict[str, Any] = {"video": videos}

    # Batch every other key as a list
    for k in batch[0].keys():
        if k == "video":
            continue
        out[k] = [b[k] for b in batch]

    return out


def main():
    ddp_kwargs = DistributedDataParallelKwargs()
    acc = Accelerator(kwargs_handlers=[ddp_kwargs])
    rank, world = acc.process_index, acc.num_processes
    device = acc.device
    logger.info(f"Rank {rank}/{world} on {device}")

    laq = LAQ(**MODEL_CFG)
    laq.load_weights(LAQ_CKPT, map_location=device)
    laq = acc.prepare(laq)
    laq = acc.unwrap_model(laq)
    laq.eval()

    global_updated_entries = []

    for task in LIBERO_TASKS:
        logger.info(f"Processing task: {task}")
        task_dir = Path(LIBERO_CONFIG["root_dir"]) / f"{task}_modified"
        task_jsonl = task_dir / f"{task}_raw.jsonl"

        # Load raw annotations
        with open(task_jsonl, "r") as f:
            task_data = [json.loads(line) for line in f]

        # Fix paths in annotations
        for entry in task_data:
            entry["id"] = f"{task}/" + entry["id"]
            entry["image"] = str(Path(task_dir) / "images" / entry["image"])

        episodes = group_by_episode(task_data)
        ep_keys = sorted(episodes.keys())

        # Shard episodes across processes
        my_eps = [k for i, k in enumerate(ep_keys) if i % world == rank]

        # Output buffer for this rank
        updated_entries: List[Dict[str, Any]] = []

        acc.wait_for_everyone()

        for ep_id in tqdm(my_eps, desc=f"Rank {rank} Processing {task} Episodes"):
            ep_data = episodes[ep_id]
            ep_ds = SingleEpisodeDataset(ep_data, H_P, H_F)
            ep_dl = DataLoader(
                ep_ds,
                batch_size=128,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                collate_fn=collate_function,
            )
            for batch in ep_dl:
                vids = batch["video"].to(device)  # (B, Win, C, H, W)
                B, W = vids.shape[0], vids.shape[1]
                assert W == (H_P + H_F + 1), f"Unexpected window W={W}"

                # Model expects (B,T,C,H,W); if needed rename
                results = laq.inference(
                    vids,
                    return_reconstructions=False,
                    return_quantized_actions=False,
                    return_quantized_actions_idxs=True,
                )

                latent_action_idxs = results["quantized_actions_idxs"]  # (B, W-1, Hq, Wq)

                # Time t is at index H_P (t -> t+1)
                center_transition_idx = H_P

                for b_idx in range(B):
                    t = batch["t"][b_idx]  # center time
                    lat_t = latent_action_idxs[b_idx, center_transition_idx:].reshape(-1)  # (H_F*Hq*Wq)
                    ep_data[t]["latent_action_idxs"] = lat_t.cpu().numpy().tolist()
                    ep_data[t]["image"] = batch["history_paths"][b_idx] + [batch["center_path"][b_idx]]
                    ep_data[t]["latent_state"] = [batch["future_paths"][b_idx][-1]]  # last future frame path
                    ep_data[t]["fields_la"] = "[instruction],[vision],latent_action"
                    ep_data[t]["fields_ls"] = "[instruction],[vision],latent_state"
                    ep_data[t]["fields_ls_la"] = "[instruction],[vision],latent_state,latent_action"
                    # Adjust image paths to be relative to data_root
                    data_root = Path(LIBERO_CONFIG["root_dir"])
                    ep_data[t]["image"] = [str(Path(p).relative_to(data_root)) for p in ep_data[t]["image"]]
                    ep_data[t]["latent_state"] = [
                        str(Path(p).relative_to(data_root)) for p in ep_data[t]["latent_state"]
                    ]
                    # Already have raw_action and instruction
            updated_entries.extend(ep_data)

        global_updated_entries.extend(updated_entries)
        updated_entries = []  # reset per-task buffer if you keep the variable

    output_dir = SAVE_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    # Each rank writes ONE shard
    _ = write_global_shard(output_dir, rank, global_updated_entries)

    # Sync all ranks
    acc.wait_for_everyone()

    # Main merges into a single JSONL and deletes shards
    if acc.is_main_process:
        merge_global_shards(output_dir, delete_shards=True)

    acc.wait_for_everyone()

    if acc.is_main_process:
        logger.info("All done.")


if __name__ == "__main__":
    main()
