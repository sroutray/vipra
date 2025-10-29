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


def run_inference(lam, dataloader, device):
    for batch in tqdm(dataloader, desc="Processing pairs"):
        vid1_batch, vid2_batch = batch
        vid1_batch = vid1_batch.to(device)
        vid2_batch = vid2_batch.to(device)
        # import ipdb; ipdb.set_trace()

        results_dict1 = lam.inference(
            vid1_batch,
            return_reconstructions=True,
            return_quantized_actions=True,
            return_quantized_actions_idxs=True,
        )
        vid1_latents = results_dict1["quantized_actions_idxs"]

        results_dict2 = lam.inference(
            vid2_batch,
            return_reconstructions=True,
            return_quantized_actions=True,
            return_quantized_actions_idxs=True,
        )
        vid2_latents = results_dict2["quantized_actions_idxs"]

        yield vid1_batch, vid2_batch, vid1_latents, vid2_latents


# === Configuration ===
ssv2_root = Path("/fsx2/shared/ssv2")
val_json_path = ssv2_root / "labels" / "validation.json"
video_dir = ssv2_root / "20bn-something-something-v2"

# === Load Up and Down Videos ===
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
save_root = Path(f"plots/codebook_usage_right_left/") / ckpt_id
save_root.mkdir(parents=True, exist_ok=True)
dataset = SSv2PairDataset(paired_videos, video_dir, num_frames=16, stepsize=2)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

codebook_size = MODEL_CFG["codebook_size"]
code_seq_len = MODEL_CFG["code_seq_len"]
codebook_usage1 = torch.zeros((code_seq_len, codebook_size), dtype=torch.int32)
codebook_usage2 = torch.zeros((code_seq_len, codebook_size), dtype=torch.int32)

for batch_idx, (vid1_batch, vid2_batch, vid1_latents, vid2_latents) in enumerate(
    run_inference(laq, dataloader, device)
):
    B = vid1_batch.shape[0]
    for i in range(B):
        vid1 = vid1_batch[i].cpu()  # [T, C, H, W]
        vid2 = vid2_batch[i].cpu()  # [T, C, H, W]
        act1 = vid1_latents[i].cpu()  # [T-1, ...]
        act2 = vid2_latents[i].cpu()  # [T-1, ...]

        for time in range(act1.shape[0]):
            act_1 = act1[time].view(-1)  # [code_seq_len]
            act_2 = act2[time].view(-1)
            assert act_1.shape[0] == act_2.shape[0] == code_seq_len
            for j in range(code_seq_len):
                for c in range(codebook_size):
                    if act_1[j] == c:
                        codebook_usage1[j, c] += 1
                    if act_2[j] == c:
                        codebook_usage2[j, c] += 1

codebook_usage1 = codebook_usage1.float() / codebook_usage1.sum(dim=1, keepdim=True)
codebook_usage1 = codebook_usage1.cpu().numpy()
print(f"Codebook usage for category1: {codebook_usage1}")
# Save codebook usage
np.save(save_root / f"codebook_usage1.npy", codebook_usage1)

codebook_usage2 = codebook_usage2.float() / codebook_usage2.sum(dim=1, keepdim=True)
codebook_usage2 = codebook_usage2.cpu().numpy()
print(f"Codebook usage for category2: {codebook_usage2}")
# Save codebook usage
np.save(save_root / f"codebook_usage2.npy", codebook_usage2)
