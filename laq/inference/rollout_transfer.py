import json
import random
from pathlib import Path
from typing import Any, Callable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T_old
import torchvision.utils as vutils
from configs.config import MODEL_CFG
from model.latent_action_quantization import LAQ
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices, clips_at_regular_indices
from torchvision.transforms import v2 as T
from tqdm import tqdm


def normalize_for_save(video_tensor):
    return torch.clamp(video_tensor, 0.0, 1.0)


def video_to_tensor(
    video_path: str,
    transform: Callable[[Any], Any] = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
    max_frames: int = 8,
    stepsize: int = 1,
    ffmpeg_threads: int = 1,
) -> torch.Tensor:
    decoder = VideoDecoder(
        video_path,
        dimension_order="NCHW",
        num_ffmpeg_threads=ffmpeg_threads,
        device="cpu",
        seek_mode="exact",
    )

    frame_batch = clips_at_random_indices(
        decoder,
        num_clips=1,
        num_frames_per_clip=max_frames,
        num_indices_between_frames=stepsize,
        policy="repeat_last",
    )

    frames_uint8 = frame_batch.data[0]
    frames = transform(frames_uint8)

    return frames


class SSv2PairDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[dict, dict]],
        video_dir: Path,
        image_size: Tuple[int, int] = (224, 224),
        num_frames: int = 10,
        stepsize: int = 4,
    ):
        self.pairs = pairs
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.stepsize = stepsize
        self.image_size = image_size
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) and img.mode != "RGB" else img),
                T.ToImage(),
                T.Resize(self.image_size),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        meta1, meta2 = self.pairs[idx]
        path1 = self.video_dir / f"{meta1['id']}.webm"
        path2 = self.video_dir / f"{meta2['id']}.webm"

        vid1 = video_to_tensor(path1, stepsize=self.stepsize, transform=self.transform, max_frames=self.num_frames)
        vid2 = video_to_tensor(path2, stepsize=self.stepsize, transform=self.transform, max_frames=self.num_frames)
        return vid1, vid2


def run_autoregressive_inference(lam, dataloader, device, num_initial_frames=3):
    for batch in tqdm(dataloader, desc="Processing pairs"):
        vid1_batch, vid2_batch = batch
        vid1_batch = vid1_batch.to(device)
        vid2_batch = vid2_batch.to(device)

        # Extract vid1 latents
        results_dict1 = lam.inference(
            vid1_batch,
            return_reconstructions=True,
            return_quantized_actions=True,
        )
        vid1_latents = results_dict1["quantized_actions"]

        # Autoregressive rollout on vid2 using vid1 latents
        vid2_batch_pred = lam.rollout_ar(vid2_batch[:, :num_initial_frames], vid1_latents)
        vid2_batch_pred = torch.cat(
            [vid2_batch[:, :num_initial_frames], vid2_batch_pred], dim=1
        )  # prepend first frames

        yield vid1_batch, vid2_batch, vid2_batch_pred


# === Configuration ===
ssv2_root = Path("/fsx2/shared/ssv2")
val_json_path = ssv2_root / "labels" / "validation.json"
video_dir = ssv2_root / "20bn-something-something-v2"
NUM_INITIAL_FRAMES = 3

# === Load Video Pairs ===
with open(val_json_path, "r") as f:
    val_data = json.load(f)

# video1 = [x for x in val_data if x["template"] == "Moving [something] up"]
# video2 = [x for x in val_data if x["template"] == "Moving [something] down"]
video1 = [x for x in val_data if x["template"] == "Pushing [something] from left to right"]
video2 = [x for x in val_data if x["template"] == "Pushing [something] from right to left"]

random.seed(42)
random.shuffle(video1)
random.shuffle(video2)
paired_videos = list(zip(video1[: min(len(video1), len(video2))], video2[: min(len(video1), len(video2))]))
# paired_videos = list(zip(video1[:],
#                          video1[:]))  # for reconstruction
# === Load Model ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
laq_ckpt = Path(
    "/fsx/sroutray/Work1/vipra/laq_v5/results/exp_laq_cotrain_flow_warmup_rope_fstdecv2_filmv2_peg_abs_run1/laq_model_milestone_240000.pt"
)
ckpt_id = "_".join(laq_ckpt.with_suffix("").parts[-2:])

MODEL_CFG["use_lpips_loss"] = False
MODEL_CFG["use_flow_loss"] = False

MODEL_CFG["encode_deltas"] = False  # Disable delta encoding
MODEL_CFG["discarding_threshold"] = 0.015
MODEL_CFG["max_codebook_update_step"] = 150_000
laq = LAQ(**MODEL_CFG)

laq.load_weights(laq_ckpt, map_location=device)
laq.eval()
laq.to(device)

# === Run Inference ===
# save_root = Path("plots/transfer/up_down")
save_root = Path(f"plots/reconstruction/") / ckpt_id
save_root.mkdir(parents=True, exist_ok=True)
dataset = SSv2PairDataset(paired_videos, video_dir, num_frames=8, stepsize=2)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

for batch_idx, (vid1_batch, vid2_batch, vid2_batch_pred) in enumerate(
    run_autoregressive_inference(laq, dataloader, device, num_initial_frames=NUM_INITIAL_FRAMES)
):
    # Reorder to [C, T, H, W] and move to CPU
    vid1 = vid1_batch[0].permute(1, 0, 2, 3).cpu()  # [C, T, H, W]
    vid2 = vid2_batch[0].permute(1, 0, 2, 3).cpu()  # [C, T, H, W]
    pred = vid2_batch_pred[0].permute(1, 0, 2, 3).cpu()  # [C, T, H, W]

    vid1 = normalize_for_save(vid1)  # Normalize to [0, 1]
    vid2 = normalize_for_save(vid2)  # Normalize to [0, 1]
    pred = normalize_for_save(pred)  # Normalize to [0, 1]

    # Get the video id for naming
    video_id = paired_videos[batch_idx][1]["id"]

    # Build three horizontal strips (gt1, gt2, pred)
    grid_rows = []
    for row in (vid1, vid2, pred):
        # row: [C, T, H, W]
        # Treat T as batch dimension to make a strip of T frames
        frames = row.permute(1, 0, 2, 3)  # [T, C, H, W]
        grid_row = vutils.make_grid(
            frames,
            nrow=frames.shape[0],
            pad_value=1.0,
            normalize=True,
            value_range=(0, 1),
        )  # [C, H, W*T]
        grid_rows.append(grid_row)

    # Stack the three strips vertically: [C, 3*H, W*T]
    final_grid = torch.cat(grid_rows, dim=1)

    # Save out
    out_dir = save_root
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{video_id}_{batch_idx}.png"
    vutils.save_image(final_grid, save_path)

    # Also save as GIF - concatenate the 3 rows horizontally for each frame
    gif_frames = []
    T = vid1.shape[1]  # Number of frames

    for t in range(NUM_INITIAL_FRAMES - 1, T):
        # Get frame t from each video: [C, H, W]
        frame1 = vid1[:, t, :, :]  # [C, H, W]
        frame2 = vid2[:, t, :, :]  # [C, H, W]
        frame_pred = pred[:, t, :, :]  # [C, H, W]

        # Concatenate horizontally: [C, H, 3*W]
        concat_frame = torch.cat([frame1, frame2, frame_pred], dim=2)

        # Convert to PIL Image (need to permute to HWC and convert to uint8)
        concat_frame_pil = T_old.ToPILImage()(concat_frame)
        gif_frames.append(concat_frame_pil)

    # Save as GIF
    gif_path = out_dir / f"{video_id}_{batch_idx}.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=250,  # 250ms per frame (4 FPS)
        loop=0,  # Loop indefinitely
    )

    print(f"[{batch_idx}] Saved interpolation grid to: {save_path}")
    print(f"[{batch_idx}] Saved GIF to: {gif_path}")

    if batch_idx >= 16:
        import ipdb

        ipdb.set_trace()
