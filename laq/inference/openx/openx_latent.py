from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
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
# Episode loading workers: Total workers divided by number of GPU ranks
# This ensures we don't oversubscribe CPU resources across multiple GPU processes
MAX_TOTAL_EPISODE_WORKERS = 72  # Total CPU workers across all ranks
EPISODE_LOADING_WORKERS = None  # Will be calculated per rank in main()

# Model configurations
MODEL_CFG["use_lpips_loss"] = False
MODEL_CFG["use_flow_loss"] = False
MODEL_CFG["encode_deltas"] = False  # Disable delta encoding
LAQ_CKPT = Path(
    "/fsx/sroutray/Work1/vipra/laq_v5/results/exp_laq_cotrain_flow_warmup_rope_fstdecv2_filmv2_peg_abs_run1/laq_model_milestone_240000.pt"
)


@beartype
def discover_openx_episodes(data_root: Path, dataset_name: str) -> List[str]:
    """Discover all episode directories in OpenX format."""
    processed_dir = data_root / "processed"
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory {processed_dir} does not exist.")

    episode_paths = [str(pth) for pth in processed_dir.iterdir() if pth.is_dir() and pth.name.startswith("episode_")]

    if not episode_paths:
        raise ValueError(f"No episodes found in {processed_dir}. Check the dataset structure.")

    return sorted(episode_paths)


def load_openx_episode_data(episode_path: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Load data for a single OpenX episode."""
    episode_dir = Path(episode_path)
    episode_id = episode_dir.name  # e.g., "episode_000000"

    # Load task description
    task_file = episode_dir / "task_description.txt"
    if task_file.exists():
        with open(task_file, "r") as f:
            instruction = f.read().strip()
    else:
        instruction = "No instruction provided"

    # Load actions
    actions_file = episode_dir / "actions.npy"
    if actions_file.exists():
        actions = np.load(actions_file)  # Shape: (T, action_dim)
    else:
        raise FileNotFoundError(f"Actions file not found: {actions_file}")

    # Get image files (support both PNG and JPG)
    images_dir = episode_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Find all image files and sort them
    png_files = list(images_dir.glob("*.png"))
    jpg_files = list(images_dir.glob("*.jpg"))
    jpeg_files = list(images_dir.glob("*.jpeg"))

    image_files = sorted(png_files + jpg_files + jpeg_files)

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    # Create entries for each timestep
    episode_data = []
    for i, img_file in enumerate(image_files):
        step_id = f"step{i:04d}"
        entry = {
            "id": f"{dataset_name}/{episode_id}/{step_id}",
            "image": str(img_file),
            "raw_action": actions[i].tolist() if i < len(actions) else [0.0] * actions.shape[1],
            "instruction": f"<s> You are a helpful assistant. USER: What action should the robot take to `{instruction}` ASSISTANT:",
        }
        episode_data.append(entry)

    return episode_data


def _load_single_episode_wrapper(args: Tuple[str, str]) -> List[Dict[str, Any]]:
    """Wrapper function for parallel episode loading - needs to be top-level for pickling."""
    episode_path, dataset_name = args
    try:
        return load_openx_episode_data(episode_path, dataset_name)
    except Exception as e:
        # Can't use logger here due to multiprocessing, will be handled in main function
        return []


def load_episodes_parallel(
    episode_paths: List[str], dataset_name: str, max_workers: int = None
) -> List[List[Dict[str, Any]]]:
    """Load multiple episodes in parallel using ProcessPoolExecutor."""
    logger.info(f"Loading {len(episode_paths)} episodes with {max_workers} parallel workers...")

    all_episodes_data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all episode loading tasks with (episode_path, dataset_name) tuples
        future_to_path = {
            executor.submit(_load_single_episode_wrapper, (ep_path, dataset_name)): ep_path for ep_path in episode_paths
        }

        # Collect results as they complete with progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(future_to_path):
            ep_path = future_to_path[future]
            try:
                ep_data = future.result()
                if ep_data:  # Only add non-empty episodes
                    all_episodes_data.append(ep_data)
                else:
                    logger.warning(f"Empty episode data for: {ep_path}")

                completed += 1
                if completed % 100 == 0:  # Log progress every 100 episodes
                    logger.info(f"Loaded {completed}/{len(episode_paths)} episodes...")

            except Exception as e:
                logger.error(f"Exception loading episode {ep_path}: {e}")

    logger.info(f"Successfully loaded {len(all_episodes_data)} out of {len(episode_paths)} episodes")
    return all_episodes_data


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
      - Each entry has keys: "image" (filepath), "id", "raw_action", "instruction".
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
            "raw_actions": self.ep[t]["raw_action"],  # Map to "raw_actions" to match LIBERO format
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
    parser = argparse.ArgumentParser(description="Process dataset for latent action quantization")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to process")
    parser.add_argument("--rank", type=int, default=None, help="Process rank (0-indexed)")
    parser.add_argument("--world_size", type=int, default=None, help="Total number of processes")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU ID to use")
    args = parser.parse_args()

    # Auto-detect rank and world size from environment if not provided
    if args.rank is None:
        rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    else:
        rank = args.rank

    if args.world_size is None:
        world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    else:
        world = args.world_size

    # Determine GPU device
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
    elif torch.cuda.is_available():
        # Auto-assign GPU based on rank
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    torch.cuda.set_device(device)

    OPENX_CONFIG = DATASET_CFG[args.dataset]
    SAVE_ROOT = Path("/fsx2/shared/sroutray/vipra_data") / args.dataset

    # Calculate episode loading workers per rank to avoid CPU oversubscription
    episode_workers_per_rank = max(1, MAX_TOTAL_EPISODE_WORKERS // world)

    # Setup logging with rank info
    logging.basicConfig(level=logging.INFO, format=f"[rank {rank}] %(asctime)s - %(levelname)s - %(message)s")

    logger.info(f"Starting rank {rank}/{world} on {device}, using {episode_workers_per_rank} episode loading workers")

    # Load model
    laq = LAQ(**MODEL_CFG)
    laq.load_weights(LAQ_CKPT, map_location=device)
    laq.to(device)
    laq.eval()

    global_updated_entries = []

    logger.info(f"Processing dataset: {args.dataset}")
    data_root = Path(OPENX_CONFIG["root_dir"])

    # Discover all episodes
    episode_paths = discover_openx_episodes(data_root, args.dataset)
    logger.info(f"Found {len(episode_paths)} episodes")

    # Shard episodes across processes
    my_episodes = [ep for i, ep in enumerate(episode_paths) if i % world == rank]
    logger.info(f"Rank {rank} processing {len(my_episodes)} episodes")

    # Load episodes in parallel for this rank
    logger.info(f"Rank {rank} loading {len(my_episodes)} episodes in parallel...")
    start_time = time.time()
    my_episodes_data = load_episodes_parallel(my_episodes, args.dataset, max_workers=episode_workers_per_rank)
    loading_time = time.time() - start_time
    logger.info(f"Rank {rank} loaded {len(my_episodes_data)} episodes in {loading_time:.2f} seconds")

    for ep_data in tqdm(my_episodes_data, desc=f"Rank {rank} Processing Episodes"):
        if len(ep_data) == 0:
            logger.warning(f"Empty episode data, skipping...")
            continue

        ep_ds = SingleEpisodeDataset(ep_data, H_P, H_F)
        ep_dl = DataLoader(
            ep_ds,
            batch_size=160,
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
            with torch.no_grad():
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
                # Extract latent actions for H_F future steps (t+1 to t+H_F)
                lat_t = latent_action_idxs[b_idx, center_transition_idx : center_transition_idx + H_F].reshape(
                    -1
                )  # (H_F*Hq*Wq)
                ep_data[t]["latent_action_idxs"] = lat_t.cpu().numpy().tolist()
                ep_data[t]["image"] = batch["history_paths"][b_idx] + [batch["center_path"][b_idx]]
                ep_data[t]["latent_state"] = [batch["future_paths"][b_idx][-1]]  # last future frame path
                ep_data[t]["fields_la"] = "[instruction],[vision],latent_action"
                ep_data[t]["fields_ls"] = "[instruction],[vision],latent_state"
                ep_data[t]["fields_ls_la"] = "[instruction],[vision],latent_state,latent_action"

                # Adjust image paths to be relative to data_root
                ep_data[t]["image"] = [str(Path(p).relative_to(data_root)) for p in ep_data[t]["image"]]
                ep_data[t]["latent_state"] = [str(Path(p).relative_to(data_root)) for p in ep_data[t]["latent_state"]]

        global_updated_entries.extend(ep_data)

    output_dir = SAVE_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    # Each rank writes ONE shard
    shard_path = write_global_shard(output_dir, rank, global_updated_entries)
    logger.info(f"Rank {rank} finished processing. Wrote shard: {shard_path}")

    # Wait for all shards to be written (including our own)
    expected_shards = [output_dir / f"all_with_latents.rank{r}.jsonl" for r in range(world)]

    logger.info(f"Rank {rank} waiting for all {world} shards to be written...")
    while True:
        existing_shards = [s for s in expected_shards if s.exists()]
        if len(existing_shards) == world:
            logger.info(f"Rank {rank} found all {world} shards, checking if merge is needed...")
            break

        missing = [s for s in expected_shards if not s.exists()]
        logger.info(f"Rank {rank} waiting for {len(missing)} shards: {[s.name for s in missing[:3]]}")
        time.sleep(5)

    # Try to acquire merge lock (first one to succeed becomes the merger)
    merge_lock_path = output_dir / "merge.lock"
    try:
        # Atomic operation: create lock file if it doesn't exist
        with open(merge_lock_path, "x") as f:  # 'x' mode fails if file exists
            f.write(f"rank_{rank}")

        # We got the lock! We are the merger
        logger.info(f"Rank {rank} acquired merge lock, performing merge...")

        # Double-check all shards still exist
        if all(s.exists() for s in expected_shards):
            final_path = merge_global_shards(output_dir, delete_shards=True)
            logger.info(f"Rank {rank} successfully merged ALL {world} shards to: {final_path}")
        else:
            logger.error(f"Rank {rank} found missing shards during final check!")

        # Remove lock file
        try:
            os.remove(merge_lock_path)
        except OSError:
            pass

    except FileExistsError:
        # Another rank is already merging or has merged
        logger.info(f"Rank {rank} found existing merge lock, skipping merge (another rank is handling it)")

    except Exception as e:
        logger.error(f"Rank {rank} error during merge: {e}")
        # Remove lock file in case of error
        try:
            os.remove(merge_lock_path)
        except OSError:
            pass

    logger.info(f"Rank {rank} completed and exiting.")


if __name__ == "__main__":
    main()
