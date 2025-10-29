import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from beartype import beartype
from model.utils import pair
from PIL import Image
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
from torchvision.transforms import v2 as T

logger = logging.getLogger(__name__)


@beartype
def discover_ssv2_paths(root_dir: Path, split: Literal["train", "val", "trainval", "test", "all"]) -> List[str]:
    paths: List[str] = []
    if split == "train":
        label_paths = [root_dir / "labels" / "train.json"]
    elif split == "val":
        label_paths = [root_dir / "labels" / "validation.json"]
    elif split == "trainval":
        label_paths = [root_dir / "labels" / "train.json", root_dir / "labels" / "validation.json"]
    elif split == "test":
        label_paths = [root_dir / "labels" / "test.json"]
    elif split == "all":
        label_paths = [
            root_dir / "labels" / "train.json",
            root_dir / "labels" / "validation.json",
            root_dir / "labels" / "test.json",
        ]
    else:
        raise ValueError(f"Unknown split: {split}")

    for label_path in label_paths:
        with open(label_path, "r") as f:
            data = json.load(f)
        for ent in data:
            vid_name = ent["id"] + ".webm"
            for d in [
                root_dir / "20bn-something-something-v2",
                root_dir / "20bn-something-something-v2-00",
                root_dir / "20bn-something-something-v2-01",
            ]:
                p = d / vid_name
                if p.exists():
                    paths.append(str(p))
                    break
    return paths


@beartype
def discover_oxe_sequences(
    data_root: Path, mode: str, num_trajs: Dict[str, int], rng: Optional[random.Random] = None
) -> List[str]:
    robot_data_root = data_root / "processed"
    if not robot_data_root.exists():
        raise FileNotFoundError(f"Data root {robot_data_root} does not exist.")
    if rng is None:
        rng = random.Random(42)
    num_trajs_to_sample = num_trajs[mode]
    video_paths = [str(pth) for pth in robot_data_root.iterdir() if pth.is_dir()]
    rng.shuffle(video_paths)
    video_paths = video_paths[:num_trajs_to_sample]
    return video_paths


@beartype
def discover_libero_sequences(
    data_root: Path, mode: str, num_trajs: Dict[str, Union[int, float]], rng: Optional[random.Random] = None
) -> List[str]:
    libero_tasks = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    if not data_root.exists():
        raise FileNotFoundError(f"Data root {data_root} does not exist.")
    if rng is None:
        rng = random.Random(42)
    video_paths = []
    for task in libero_tasks:
        task_dir = data_root / f"{task}_modified" / "images"
        if not task_dir.exists():
            logger.warning(f"Task directory {task_dir} does not exist, skipping.")
            continue
        task_videos = [str(pth) for pth in task_dir.iterdir() if pth.is_dir()]
        rng.shuffle(task_videos)
        num_trajs_to_sample = (
            num_trajs[mode] if isinstance(num_trajs[mode], int) else int(len(task_videos) * num_trajs[mode])
        )
        task_videos = task_videos[:num_trajs_to_sample]
        video_paths.extend(task_videos)
    if not video_paths:
        raise ValueError(f"No videos found in any of the Libero tasks: {libero_tasks}. Check the dataset structure.")
    return video_paths


@beartype
def sequence_to_tensor(
    path: str,
    transform: Callable[[Any], Any] = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
    max_frames: int = 8,
    stepsize: int = 1,
) -> torch.Tensor:
    img_list = os.listdir(path)
    img_list = sorted(img_list)
    start_idx = random.randint(0, len(img_list) - 1)
    idxs = [min(start_idx + i * stepsize, len(img_list) - 1) for i in range(max_frames)]
    selected_imgs = [img_list[i] for i in idxs]
    frames = [Image.open(os.path.join(path, img)) for img in selected_imgs]
    frames_torch = tuple(map(transform, frames))
    frames_torch = torch.stack(frames_torch, dim=0)  # (T, C, H, W)
    return frames_torch


@beartype
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


class VideoDatasetCoTrain(Dataset):
    @beartype
    def __init__(
        self,
        dataset_cfg: Dict[str, Any],
        image_size: int = 224,
        max_frames: int = 8,
        seed: Optional[int] = 42,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.image_size = pair(image_size)
        self.max_frames = max_frames
        self.seed = seed
        self.rng = random.Random(seed)
        self.debug_mode = debug_mode

        self.paths: List[(str, str)] = []
        for name, cfg in dataset_cfg.items():
            if "ssv2" in name:
                paths = discover_ssv2_paths(cfg["root_dir"], cfg["split"])
            elif any(k in name for k in ("fractal", "bridge", "kuka")):
                paths = discover_oxe_sequences(cfg["root_dir"], cfg["split"], cfg["num_trajs"], self.rng)
            elif "libero" in name:
                paths = discover_libero_sequences(cfg["root_dir"], cfg["split"], cfg["num_trajs"], self.rng)
            else:
                raise ValueError(f"Unsupported dataset: {name}")
            self.paths.extend([(name, p) for p in paths])
            logger.info(f"Discovered {len(paths)} videos for dataset {name} in split {cfg['split']}.")

        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) and img.mode != "RGB" else img),
                T.ToImage(),
                T.Resize(self.image_size),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

        self.rng.shuffle(self.paths)
        logger.info(f"Total videos across all datasets: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        try:
            dataset_name, data_path = self.paths[idx]
            if "ssv2" in dataset_name:
                video_tensor = video_to_tensor(
                    data_path,
                    transform=self.transform,
                    max_frames=self.max_frames,
                    stepsize=self.dataset_cfg[dataset_name]["stepsize"],
                )
            elif any(k in dataset_name for k in ("fractal", "bridge", "kuka")):
                data_path = os.path.join(data_path, "images")
                video_tensor = sequence_to_tensor(
                    data_path,
                    transform=self.transform,
                    max_frames=self.max_frames,
                    stepsize=self.dataset_cfg[dataset_name]["stepsize"],
                )
            elif "libero" in dataset_name:
                video_tensor = sequence_to_tensor(
                    data_path,
                    transform=self.transform,
                    max_frames=self.max_frames,
                    stepsize=self.dataset_cfg[dataset_name]["stepsize"],
                )
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            assert video_tensor.shape[0] == self.max_frames
            # All True mask for now
            # Repeat ending frames if video is shorter than max_frames
            mask = torch.ones((self.max_frames,), dtype=torch.bool)
            return video_tensor, mask
        except Exception as e:
            logger.warning(f"Error loading video at index {idx} from path {data_path}: {e}")
            if self.debug_mode:
                raise e
            if idx < self.__len__() - 1:
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))
