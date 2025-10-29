import json
import os
import random
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import absl.logging as logging
import albumentations as A
import jax
import numpy as np
from annotated_types import Not
from jax.experimental.multihost_utils import host_local_array_to_global_array
from jax.sharding import PartitionSpec as PS
from ml_collections import ConfigDict
from PIL import Image
from policy.vqgan import VQGAN
from tqdm import tqdm
from tux import open_file


def parse_image_id_to_components(image_id: str) -> Tuple[str, str]:
    """
    Parse cache image_id to extract dataset and relative path.

    Examples:
    - "fractal20220817_data_episode_028981_0003.png" -> ("fractal", "episode_028981/images/0003.png")
    - "131791_000059.jpg" -> ("ssv2", "20bn-something-something-v2-frames/131791/000059.jpg")
    - "libero_ep00000_step0000.png" -> ("libero", "libero_10_modified/images/ep00000/step0000.png")
    """
    if image_id.startswith("fractal20220817_data_"):
        # Parse fractal format: fractal20220817_data_episode_028981_0003.png
        parts = image_id.replace("fractal20220817_data_", "").replace(".png", "")
        episode_parts = parts.split("_")
        episode_num = episode_parts[1]  # 028981
        step_num = episode_parts[2]  # 0003
        relative_path = f"processed/episode_{episode_num}/images/{step_num}.png"
        return "fractal", relative_path

    elif image_id.startswith("libero_") and "_step" in image_id:
        # Parse libero formats:
        # - "libero_ep00000_step0000.png" -> ("libero", "libero_10_modified/images/ep00000/step0000.png") (old format)
        # - "libero_10_ep00000_step0000.png" -> ("libero", "libero_10_modified/images/ep00000/step0000.png") (new format)
        # - "libero_goal_ep00426_step0089.png" -> ("libero", "libero_goal_modified/images/ep00426/step0089.png")
        # - "libero_object_ep00002_step0000.png" -> ("libero", "libero_object_modified/images/ep00002/step0000.png")
        # - "libero_spatial_ep00000_step0000.png" -> ("libero", "libero_spatial_modified/images/ep00000/step0000.png")

        parts = image_id.replace(".png", "").split("_")

        if len(parts) == 3:  # libero_ep00000_step0000 (old format, backward compatibility)
            # This is the original libero_10_modified format
            ep_part = parts[1]  # ep00000
            step_part = parts[2]  # step0000
            relative_path = f"libero_10_modified/images/{ep_part}/{step_part}.png"
        elif len(parts) == 4:  # libero_subtype_ep00426_step0089 (new consistent format)
            # This is the subtype format (including "10" as a subtype)
            subtype = parts[1]  # goal, object, spatial, or 10
            ep_part = parts[2]  # ep00426
            step_part = parts[3]  # step0089
            relative_path = f"libero_{subtype}_modified/images/{ep_part}/{step_part}.png"
        else:
            # Fallback for unexpected format
            relative_path = image_id

        return "libero", relative_path

    elif image_id.startswith("bridge_episode_"):
        # Parse bridge format: bridge_episode_001234_0005.png
        parts = image_id.replace("bridge_episode_", "").replace(".png", "")
        episode_parts = parts.split("_")
        episode_num = episode_parts[0]  # 001234
        step_num = episode_parts[1]  # 0005
        relative_path = f"processed/episode_{episode_num}/images/{step_num}.png"
        return "bridge", relative_path

    elif image_id.startswith("kuka_episode_"):
        # Parse kuka format: kuka_episode_005678_0010.png
        parts = image_id.replace("kuka_episode_", "").replace(".png", "")
        episode_parts = parts.split("_")
        episode_num = episode_parts[0]  # 005678
        step_num = episode_parts[1]  # 0010
        relative_path = f"processed/episode_{episode_num}/images/{step_num}.png"
        return "kuka", relative_path

    elif "_" in image_id and image_id.endswith(".jpg"):
        # Parse SSV2 format: 131791_000059.jpg
        parts = image_id.replace(".jpg", "").split("_")
        video_id = parts[0]  # 131791
        frame_num = parts[1]  # 000059
        relative_path = f"20bn-something-something-v2-frames/{video_id}/{frame_num}.jpg"
        return "ssv2", relative_path

    else:
        # Unknown format
        return "unknown", image_id


def image_path_to_cache_key(dataset: str, image_path: str) -> str:
    """
    Convert cotrain image path to cache key format.

    Examples:
    - ("fractal", "processed/episode_000032/images/0000.png") -> "fractal20220817_data_episode_000032_0000.png"
    - ("ssv2", "20bn-something-something-v2-frames/157360/000059.jpg") -> "157360_000059.jpg"
    """
    if dataset == "fractal":
        # Convert: processed/episode_000032/images/0000.png -> fractal20220817_data_episode_000032_0000.png
        path_obj = Path(image_path)
        if "episode_" in str(path_obj):
            episode_dir = [p for p in path_obj.parts if p.startswith("episode_")][0]
            episode_num = episode_dir.replace("episode_", "")
            step_num = path_obj.stem  # Remove .png extension
            return f"fractal20220817_data_episode_{episode_num}_{step_num}.png"

    elif dataset.startswith("ssv2"):
        # Convert: 20bn-something-something-v2-frames/157360/000059.jpg -> 157360_000059.jpg
        path_obj = Path(image_path)
        if len(path_obj.parts) >= 2:
            video_id = path_obj.parts[-2]  # 157360
            frame_file = path_obj.parts[-1]  # 000059.jpg
            return f"{video_id}_{frame_file}"

    elif dataset == "bridge":
        # Bridge format similar to fractal but might need different handling
        path_obj = Path(image_path)
        if "episode_" in str(path_obj):
            episode_dir = [p for p in path_obj.parts if p.startswith("episode_")][0]
            episode_num = episode_dir.replace("episode_", "")
            step_num = path_obj.stem
            return f"bridge_episode_{episode_num}_{step_num}.png"

    elif dataset == "kuka":
        # Kuka format similar to fractal
        path_obj = Path(image_path)
        if "episode_" in str(path_obj):
            episode_dir = [p for p in path_obj.parts if p.startswith("episode_")][0]
            episode_num = episode_dir.replace("episode_", "")
            step_num = path_obj.stem
            return f"kuka_episode_{episode_num}_{step_num}.png"

    elif dataset.startswith("libero"):
        # Libero format: libero_goal_modified/images/ep00426/step0089.png -> libero_goal_ep00426_step0089.png
        # Different libero subdirs: libero_10_modified, libero_goal_modified, libero_object_modified, libero_spatial_modified
        path_obj = Path(image_path)
        parts = path_obj.parts

        # Find the libero subdir (e.g., "libero_goal_modified")
        libero_subdir = None
        for part in parts:
            if part.startswith("libero_") and part.endswith("_modified"):
                libero_subdir = part
                break

        # Find the episode part (e.g., "ep00426")
        ep_part = None
        for part in parts:
            if part.startswith("ep") and part[2:].isdigit():
                ep_part = part
                break

        # Get step from filename (e.g., "step0089" from "step0089.png")
        step_part = path_obj.stem

        if libero_subdir and ep_part and step_part.startswith("step"):
            # Extract subtype from libero_subdir: libero_goal_modified -> goal
            subtype = libero_subdir.replace("libero_", "").replace("_modified", "")
            # Use consistent format for all libero subtypes including "10"
            return f"libero_{subtype}_{ep_part}_{step_part}.png"
        else:
            # Fallback: try to extract numbers from path
            logging.debug(f"Debug: Libero path parts: {parts}, stem: {path_obj.stem}")
            return f"libero_{path_obj.parent.name}_{path_obj.stem}.png"

    # Fallback: return original path as key
    return image_path


def extract_dataset_from_id(entry_id: str) -> str:
    """Extract dataset name from cotrain entry id."""
    if entry_id.startswith("bridge/"):
        return "bridge"
    elif entry_id.startswith("fractal/"):
        return "fractal"
    elif entry_id.startswith("kuka/"):
        return "kuka"
    elif entry_id.startswith("ssv2_"):
        return "ssv2"
    elif entry_id.startswith("libero_"):
        return "libero"
    else:
        return "unknown"


def get_preprocessor(image_aug: bool = False, flip_vertically: bool = False):
    transforms = []

    if flip_vertically:
        transforms.append(A.VerticalFlip(p=1.0))

    transforms.append(A.LongestMaxSize(max_size=256))
    transforms.append(A.Resize(256, 256))

    if image_aug:
        transforms.extend(
            [
                A.RandomResizedCrop((256, 256), scale=(0.9, 0.9), ratio=(1.0, 1.0)),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20),
            ]
        )

    return A.Compose(transforms)


def process_images_batch(image_paths, preprocessor):
    images = [np.array(Image.open(open_file(path, "rb"))).astype(np.uint8) for path in image_paths]
    processed_images = np.array([preprocessor(image=img)["image"] for img in images])
    processed_images = (processed_images / 127.5 - 1.0).astype(np.float32)

    return processed_images


def encode_images(vqgan, images):
    # Assuming VQGAN can encode a batch of images
    encoded = jax.device_get(vqgan.encode(images))[1].astype(int)
    return encoded


class DatasetFactory(object):
    """Datset builder class."""

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = "json_vision_dynamics"

        config.dynamics_vision_text_processor = DynamicsVisionTextProcessor.get_default_config()
        config.json_dynamics_dataset = JsonDynamicsDataset.get_default_config()

        config.dynamics_vision_action_quant_processor = DynamicsVisionActionQuantProcessor.get_default_config()
        config.json_dynamics_action_quant_dataset = JsonDynamicsActionQuantDataset.get_default_config()

        config.dynamics_vision_action_cont_processor = DynamicsVisionActionContProcessor.get_default_config()
        config.json_dynamics_action_cont_dataset = JsonDynamicsActionContDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        if config.type == "json_vision_dynamics":
            vision_text_processor = DynamicsVisionTextProcessor(config.dynamics_vision_text_processor, tokenizer)
            return JsonDynamicsDataset(config.json_dynamics_dataset, tokenizer, vision_text_processor, **kwargs)
        elif config.type == "json_vision_dynamics_action_quant":
            vision_text_processor = DynamicsVisionActionQuantProcessor(
                config.dynamics_vision_action_quant_processor, tokenizer
            )
            return JsonDynamicsActionQuantDataset(
                config.json_dynamics_action_quant_dataset, tokenizer, vision_text_processor, **kwargs
            )
        elif config.type == "json_vision_dynamics_action_cont":
            vision_text_processor = DynamicsVisionActionContProcessor(
                config.dynamics_vision_action_cont_processor, tokenizer
            )
            return JsonDynamicsActionContDataset(
                config.json_dynamics_action_cont_dataset, tokenizer, vision_text_processor, **kwargs
            )
        else:
            raise ValueError(f"Unknown dataset type: {config.type}")

    def __init__(self):
        raise ValueError("DatasetFactory is a static class and should not be instantiated.")


class DynamicsVisionTextProcessor(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ""
        config.subfield_separator = " "
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ""
        config.fields_index = -1
        config.eof_token = 8192  # denotes end of each frame for video generation
        config.eov_token = 8193  # denotes end of vision generation
        config.n_tokens_per_frame = 256  # 16 x 16 VQ codes
        config.n_tokens_per_latent_action = 1  # LAQ code_seq_len
        config.n_tokens_per_latent_state = 256  # 16 x 16 VQ codes
        config.eola_token = -1  # denotes end of latent action
        config.max_n_frames = -1
        config.img_aug = False
        config.vqgan_checkpoint_path = ""
        config.vision_path = ""
        config.image_absolute_path = ""
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields_from_example != "", "fields_from_example must be specified."
        assert self.config.eola_token != -1, "eola_token must be specified."
        self.tokenizer = tokenizer
        self.vision_start = tokenizer.encode("<vision>")
        self.vision_end = tokenizer.encode("</vision>")
        self.dynamics_start = tokenizer.encode("<dynamics>")
        self.dynamics_end = tokenizer.encode("</dynamics>")
        self.preprocessor = get_preprocessor(image_aug=self.config.img_aug)
        self.vision_dict = dict()
        if self.config.vision_path != "":
            logging.info(f"Using cached vision encodings from {self.config.vision_path}")
            with open(self.config.vision_path, "r") as fh:
                for line in tqdm(fh, desc="Loading vision encodings"):
                    data = json.loads(line)
                    self.vision_dict[data["image_id"]] = data["image_encodings"]
        else:
            assert self.config.vqgan_checkpoint_path != "", "vqgan_checkpoint_path must be specified."
            logging.info("No vision encodings provided. Will encode images on the fly.")
            self.vqgan = VQGAN(self.config.vqgan_checkpoint_path, replicate=False)

    def __call__(self, example, has_aux=False, add_bos_token=True, add_eos_token=True):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        rand_state = random.Random(aux[-1] + aux[-2])  # makes augmentations deterministic by line number + epoch
        try:
            token_buffer = []
            loss_mask_buffer = []
            vision_mask = []
            latent_action_mask = []
            latent_state_mask = []

            fields = example[self.config.fields_from_example]
            if isinstance(fields, (tuple, list)):
                if self.config.fields_index >= 0:
                    fields = fields[self.config.fields_index]
                else:
                    # seed based on line number
                    fields = rand_state.choice(fields)
            fields = fields.split(",")

            if add_bos_token and self.config.add_bos_token:
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(0.0)
                vision_mask.append(False)
                latent_action_mask.append(False)
                latent_state_mask.append(False)

            for i, field in enumerate(fields):
                if field.startswith("[") and field.endswith("]"):
                    # No loss for this field.
                    field = field[1:-1].strip()
                    mask = 0.0
                else:
                    mask = 1.0

                if field == "<|bos|>":
                    token_buffer.append(self.tokenizer.bos_token_id)
                    loss_mask_buffer.append(mask)
                    vision_mask.append(False)
                    latent_action_mask.append(False)
                    latent_state_mask.append(False)
                elif field == "<|eos|>":
                    token_buffer.append(self.tokenizer.eos_token_id)
                    loss_mask_buffer.append(mask)
                    vision_mask.append(False)
                    latent_action_mask.append(False)
                    latent_state_mask.append(False)
                elif "vision" in field:
                    # Check if latent_state is present
                    if_vision_end = True
                    for key in fields:
                        if "latent_state" in key:
                            if_vision_end = False
                            break
                    dataset = extract_dataset_from_id(example["id"])
                    if self.vision_dict:
                        # Generate proper cache keys using the same logic as cache generation
                        image_ids = []
                        for img_path in example["image"]:
                            cache_key = image_path_to_cache_key(dataset, img_path)
                            image_ids.append(cache_key)
                        encoded_images = []
                        for img_id in image_ids:
                            encoded_img = [int(x) for x in self.vision_dict[img_id]]
                            encoded_images.append(encoded_img)
                        encoded_images = np.array(encoded_images)
                    else:
                        image_paths = [
                            os.path.join(self.config.image_absolute_path, dataset, img_path)
                            for img_path in example["image"]
                        ]
                        processed_images = process_images_batch(image_paths, self.preprocessor)
                        encoded_images = encode_images(self.vqgan, processed_images)
                    vision_tokens = encoded_images.flatten().tolist()
                    n_frames = int(len(vision_tokens) / self.config.n_tokens_per_frame)
                    if self.config.max_n_frames > 0 and n_frames > self.config.max_n_frames:  # uniformly select
                        idxs = np.linspace(0, n_frames - 1, self.config.max_n_frames).astype(int)
                        new_vision_tokens = []
                        for idx in idxs:
                            new_vision_tokens.extend(
                                vision_tokens[
                                    idx * self.config.n_tokens_per_frame : (idx + 1) * self.config.n_tokens_per_frame
                                ]
                            )
                        vision_tokens = new_vision_tokens
                        n_frames = self.config.max_n_frames
                    assert int(len(vision_tokens) / self.config.n_tokens_per_frame) == n_frames, (
                        int(len(vision_tokens) / self.config.n_tokens_per_frame),
                        n_frames,
                    )

                    assert n_frames > 0, len(vision_tokens)
                    tokens = list(self.vision_start)
                    for j in range(n_frames):
                        tokens.extend(
                            vision_tokens[j * self.config.n_tokens_per_frame : (j + 1) * self.config.n_tokens_per_frame]
                        )
                        if if_vision_end and j == n_frames - 1:
                            tokens.append(self.config.eov_token)
                        else:
                            tokens.append(self.config.eof_token)

                    if if_vision_end:
                        tokens.extend(self.vision_end)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    vision_mask.extend([False] * len(self.vision_start))
                    vision_mask.extend(
                        [True] * (self.config.n_tokens_per_frame * n_frames + n_frames)
                    )  # include extra eof token at the end of each frame
                    if if_vision_end:
                        vision_mask.extend([False] * len(self.vision_end))
                    latent_action_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                elif "latent_action" in field:
                    # cummulative latent action
                    latent_action_tokens = example[field + "_idxs"]
                    tokens = list(self.dynamics_start)  # start of dynamics
                    n_latent_actions = int(len(latent_action_tokens) / self.config.n_tokens_per_latent_action)
                    for idx in range(n_latent_actions):
                        tokens.extend(
                            latent_action_tokens[
                                idx
                                * self.config.n_tokens_per_latent_action : (idx + 1)
                                * self.config.n_tokens_per_latent_action
                            ]
                        )
                        tokens.append(self.config.eola_token)
                    tokens.extend(self.dynamics_end)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    latent_action_mask.extend([False] * len(self.dynamics_start))
                    latent_action_mask.extend(
                        [True] * (self.config.n_tokens_per_latent_action * n_latent_actions + n_latent_actions)
                    )  # include extra eola token
                    latent_action_mask.extend([False] * len(self.dynamics_end))
                    vision_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                elif "latent_state" in field:
                    # similar to vision
                    dataset = extract_dataset_from_id(example["id"])
                    if self.vision_dict:
                        # Generate proper cache keys using the same logic as cache generation
                        latent_image_ids = []
                        for ls_path in example[field]:
                            cache_key = image_path_to_cache_key(dataset, ls_path)
                            latent_image_ids.append(cache_key)
                        encoded_latent_images = []
                        for img_id in latent_image_ids:
                            encoded_img = [int(x) for x in self.vision_dict[img_id]]
                            encoded_latent_images.append(encoded_img)
                        encoded_latent_images = np.array(encoded_latent_images)
                    else:
                        latent_image_path = [
                            os.path.join(self.config.image_absolute_path, dataset, ls_path)
                            for ls_path in example[field]
                        ]
                        processed_latent_images = process_images_batch(latent_image_path, self.preprocessor)
                        encoded_latent_images = encode_images(self.vqgan, processed_latent_images)
                    latent_state_tokens = encoded_latent_images.flatten().tolist()
                    n_latent_states = int(len(latent_state_tokens) / self.config.n_tokens_per_latent_state)
                    tokens = []  # make sure continue the vision token sequence here and end with eov
                    for idx in range(n_latent_states):
                        tokens.extend(
                            latent_state_tokens[
                                idx
                                * self.config.n_tokens_per_latent_state : (idx + 1)
                                * self.config.n_tokens_per_latent_state
                            ]
                        )
                        if idx == n_latent_states - 1:
                            tokens.append(self.config.eov_token)
                        else:
                            tokens.append(self.config.eof_token)
                    tokens.extend(self.vision_end)  # end of vision
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    latent_state_mask.extend(
                        [True] * (self.config.n_tokens_per_latent_state * n_latent_states + n_latent_states)
                    )  # include extra eof/eov token
                    latent_state_mask.extend([False] * len(self.vision_end))
                    latent_action_mask.extend([False] * len(tokens))
                    vision_mask.extend([False] * len(tokens))
                else:
                    subfields = field.split("+")
                    text = self.config.subfield_separator.join([example[subfield] for subfield in subfields])
                    if i == 0:
                        text = self.config.prepend_text + text
                    tokens = self.tokenizer.encode(text)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    vision_mask.extend([False] * len(tokens))
                    latent_action_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))

            if add_eos_token and self.config.add_eos_token:
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(1.0)
                vision_mask.append(False)
                latent_action_mask.append(False)
                latent_state_mask.append(False)
            assert (
                len(token_buffer)
                == len(loss_mask_buffer)
                == len(vision_mask)
                == len(latent_action_mask)
                == len(latent_state_mask)
            ), (
                len(token_buffer),
                len(loss_mask_buffer),
                len(vision_mask),
                len(latent_action_mask),
                len(latent_state_mask),
            )
            keep = True
            return token_buffer, loss_mask_buffer, vision_mask, latent_action_mask, latent_state_mask, keep, *aux
        except Exception as e:
            logging.error(f"Error processing example {example.get('id', 'unknown')}: {e}")
            return [], [], [], [], [], False, *aux


class DynamicsVisionActionQuantProcessor(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ""
        config.subfield_separator = " "
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ""
        config.fields_index = -1
        config.eof_token = 8192  # denotes end of each frame for video generation
        config.eov_token = 8193  # denotes end of vision generation
        config.n_tokens_per_frame = 256  # 16 x 16 VQ codes
        config.n_tokens_per_latent_action = 1  # LAQ code_seq_len
        config.n_tokens_per_latent_state = 256  # 16 x 16 VQ codes
        config.n_tokens_per_action = 7
        config.eola_token = -1  # denotes end of latent action
        config.eoa_token = -1  # denotes end of action
        config.max_n_frames = -1
        config.img_aug = False
        config.vqgan_checkpoint_path = ""
        config.vision_path = ""
        config.image_absolute_path = ""
        config.vertical_flip = False
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields_from_example != "", "fields_from_example must be specified."
        assert self.config.eola_token != -1, "eola_token must be specified."
        assert self.config.eoa_token != -1, "eoa_token must be specified."
        self.tokenizer = tokenizer
        self.vision_start = tokenizer.encode("<vision>")
        self.vision_end = tokenizer.encode("</vision>")
        self.dynamics_start = tokenizer.encode("<dynamics>")
        self.dynamics_end = tokenizer.encode("</dynamics>")
        self.action_start = tokenizer.encode("<action>")
        self.action_end = tokenizer.encode("</action>")
        self.preprocessor = get_preprocessor(image_aug=self.config.img_aug, flip_vertically=self.config.vertical_flip)
        self.vision_dict = dict()
        if self.config.vision_path != "":
            with open(self.config.vision_path, "r") as fh:
                for line in fh:
                    data = json.loads(line)
                    self.vision_dict[data["image_id"]] = data["image_encodings"]
            logging.info(f"Using cached vision encodings from {self.config.vision_path}")
        else:
            assert self.config.vqgan_checkpoint_path != "", "vqgan_checkpoint_path must be specified."
            logging.info("No vision encodings provided. Will encode images on the fly.")
            self.vqgan = VQGAN(self.config.vqgan_checkpoint_path, replicate=False)

    def __call__(self, example, has_aux=False, add_bos_token=True, add_eos_token=True):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        rand_state = random.Random(aux[-1] + aux[-2])  # makes augmentations deterministic by line number + epoch
        try:
            token_buffer = []
            loss_mask_buffer = []
            vision_mask = []
            latent_action_mask = []
            latent_state_mask = []
            action_mask = []

            fields = example[self.config.fields_from_example]
            if isinstance(fields, (tuple, list)):
                if self.config.fields_index >= 0:
                    fields = fields[self.config.fields_index]
                else:
                    # seed based on line number
                    fields = rand_state.choice(fields)
            fields = fields.split(",")

            if add_bos_token and self.config.add_bos_token:
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(0.0)
                vision_mask.append(False)
                latent_action_mask.append(False)
                latent_state_mask.append(False)
                action_mask.append(False)

            for i, field in enumerate(fields):
                if field.startswith("[") and field.endswith("]"):
                    # No loss for this field.
                    field = field[1:-1].strip()
                    mask = 0.0
                else:
                    mask = 1.0

                if field == "<|bos|>":
                    token_buffer.append(self.tokenizer.bos_token_id)
                    loss_mask_buffer.append(mask)
                    vision_mask.append(False)
                    latent_action_mask.append(False)
                    latent_state_mask.append(False)
                    action_mask.append(False)
                elif field == "<|eos|>":
                    token_buffer.append(self.tokenizer.eos_token_id)
                    loss_mask_buffer.append(mask)
                    vision_mask.append(False)
                    latent_action_mask.append(False)
                    latent_state_mask.append(False)
                    action_mask.append(False)
                elif "vision" in field:
                    # Check if latent_state is present
                    if_vision_end = True
                    for key in fields:
                        if "latent_state" in key:
                            if_vision_end = False
                            break
                    dataset = extract_dataset_from_id(example["id"])
                    if self.vision_dict:
                        # Generate proper cache keys using the same logic as cache generation
                        image_ids = []
                        for img_path in example["image"]:
                            cache_key = image_path_to_cache_key(dataset, img_path)
                            image_ids.append(cache_key)
                        encoded_images = []
                        for img_id in image_ids:
                            encoded_img = [int(x) for x in self.vision_dict[img_id]]
                            encoded_images.append(encoded_img)
                        encoded_images = np.array(encoded_images)
                    else:
                        image_paths = [
                            os.path.join(self.config.image_absolute_path, dataset, img_path)
                            for img_path in example["image"]
                        ]
                        processed_images = process_images_batch(image_paths, self.preprocessor)
                        encoded_images = encode_images(self.vqgan, processed_images)
                    vision_tokens = encoded_images.flatten().tolist()
                    n_frames = int(len(vision_tokens) / self.config.n_tokens_per_frame)
                    if self.config.max_n_frames > 0 and n_frames > self.config.max_n_frames:  # uniformly select
                        idxs = np.linspace(0, n_frames - 1, self.config.max_n_frames).astype(int)
                        new_vision_tokens = []
                        for idx in idxs:
                            new_vision_tokens.extend(
                                vision_tokens[
                                    idx * self.config.n_tokens_per_frame : (idx + 1) * self.config.n_tokens_per_frame
                                ]
                            )
                        vision_tokens = new_vision_tokens
                        n_frames = self.config.max_n_frames
                    assert int(len(vision_tokens) / self.config.n_tokens_per_frame) == n_frames, (
                        int(len(vision_tokens) / self.config.n_tokens_per_frame),
                        n_frames,
                    )

                    assert n_frames > 0, len(vision_tokens)
                    tokens = list(self.vision_start)
                    for j in range(n_frames):
                        tokens.extend(
                            vision_tokens[j * self.config.n_tokens_per_frame : (j + 1) * self.config.n_tokens_per_frame]
                        )
                        if if_vision_end and j == n_frames - 1:
                            tokens.append(self.config.eov_token)
                        else:
                            tokens.append(self.config.eof_token)

                    if if_vision_end:
                        tokens.extend(self.vision_end)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    vision_mask.extend([False] * len(self.vision_start))
                    vision_mask.extend(
                        [True] * (self.config.n_tokens_per_frame * n_frames + n_frames)
                    )  # include extra eof token at the end of each frame
                    if if_vision_end:
                        vision_mask.extend([False] * len(self.vision_end))
                    latent_action_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))
                elif "latent_action" in field:
                    # cummulative latent action
                    latent_action_tokens = example[field + "_idxs"]
                    tokens = list(self.dynamics_start)  # start of dynamics
                    n_latent_actions = int(len(latent_action_tokens) / self.config.n_tokens_per_latent_action)
                    for idx in range(n_latent_actions):
                        tokens.extend(
                            latent_action_tokens[
                                idx
                                * self.config.n_tokens_per_latent_action : (idx + 1)
                                * self.config.n_tokens_per_latent_action
                            ]
                        )
                        tokens.append(self.config.eola_token)
                    tokens.extend(self.dynamics_end)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    latent_action_mask.extend([False] * len(self.dynamics_start))
                    latent_action_mask.extend(
                        [True] * (self.config.n_tokens_per_latent_action * n_latent_actions + n_latent_actions)
                    )  # include extra eola token
                    latent_action_mask.extend([False] * len(self.dynamics_end))
                    vision_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))
                elif "latent_state" in field:
                    # similar to vision
                    dataset = extract_dataset_from_id(example["id"])
                    if self.vision_dict:
                        # Generate proper cache keys using the same logic as cache generation
                        latent_image_ids = []
                        for ls_path in example[field]:
                            dataset = extract_dataset_from_id(example["id"])
                            cache_key = image_path_to_cache_key(dataset, ls_path)
                            latent_image_ids.append(cache_key)
                        encoded_latent_images = []
                        for img_id in latent_image_ids:
                            encoded_img = [int(x) for x in self.vision_dict[img_id]]
                            encoded_latent_images.append(encoded_img)
                        encoded_latent_images = np.array(encoded_latent_images)
                    else:
                        latent_image_path = [
                            os.path.join(self.config.image_absolute_path, dataset, ls_path)
                            for ls_path in example[field]
                        ]
                        processed_latent_images = process_images_batch(latent_image_path, self.preprocessor)
                        encoded_latent_images = encode_images(self.vqgan, processed_latent_images)
                    latent_state_tokens = encoded_latent_images.flatten().tolist()
                    n_latent_states = int(len(latent_state_tokens) / self.config.n_tokens_per_latent_state)
                    tokens = []  # make sure continue the vision token sequence here and end with eov
                    for idx in range(n_latent_states):
                        tokens.extend(
                            latent_state_tokens[
                                idx
                                * self.config.n_tokens_per_latent_state : (idx + 1)
                                * self.config.n_tokens_per_latent_state
                            ]
                        )
                        if idx == n_latent_states - 1:
                            tokens.append(self.config.eov_token)
                        else:
                            tokens.append(self.config.eof_token)
                    tokens.extend(self.vision_end)  # end of vision
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    latent_state_mask.extend(
                        [True] * (self.config.n_tokens_per_latent_state * n_latent_states + n_latent_states)
                    )  # include extra eof/eov token
                    latent_state_mask.extend([False] * len(self.vision_end))
                    latent_action_mask.extend([False] * len(tokens))
                    vision_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))
                elif "action" in field:
                    action_tokens = example[field]
                    tokens = list(self.action_start)
                    n_actions = int(len(action_tokens) / self.config.n_tokens_per_action)
                    for idx in range(n_actions):
                        tokens.extend(
                            action_tokens[
                                idx * self.config.n_tokens_per_action : (idx + 1) * self.config.n_tokens_per_action
                            ]
                        )
                        tokens.append(self.config.eoa_token)
                    tokens.extend(self.action_end)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    action_mask.extend([False] * len(self.action_start))
                    action_mask.extend(
                        [True] * (self.config.n_tokens_per_action * n_actions + n_actions)
                    )  # include extra eoa token
                    action_mask.extend([False] * len(self.action_end))
                    vision_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    latent_action_mask.extend([False] * len(tokens))
                else:
                    subfields = field.split("+")
                    text = self.config.subfield_separator.join([example[subfield] for subfield in subfields])
                    if i == 0:
                        text = self.config.prepend_text + text
                    tokens = self.tokenizer.encode(text)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    vision_mask.extend([False] * len(tokens))
                    latent_action_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))

            if add_eos_token and self.config.add_eos_token:
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(1.0)
                vision_mask.append(False)
                latent_action_mask.append(False)
                latent_state_mask.append(False)
                action_mask.append(False)
            assert (
                len(token_buffer)
                == len(loss_mask_buffer)
                == len(vision_mask)
                == len(latent_action_mask)
                == len(latent_state_mask)
                == len(action_mask)
            ), (
                len(token_buffer),
                len(loss_mask_buffer),
                len(vision_mask),
                len(latent_action_mask),
                len(latent_state_mask),
                len(action_mask),
            )
            keep = True
            return (
                token_buffer,
                loss_mask_buffer,
                vision_mask,
                latent_action_mask,
                latent_state_mask,
                action_mask,
                keep,
                *aux,
            )
        except Exception as e:
            logging.error(f"Error processing example {example.get('id', 'unknown')}: {e}")
            return [], [], [], [], [], [], False, *aux


class DynamicsVisionActionContProcessor(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ""
        config.subfield_separator = " "
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ""
        config.fields_index = -1
        config.eof_token = 8192  # denotes end of each frame for video generation
        config.eov_token = 8193  # denotes end of vision generation
        config.n_tokens_per_frame = 256  # 16 x 16 VQ codes
        config.n_tokens_per_latent_action = 1  # LAQ code_seq_len
        config.n_tokens_per_latent_state = 256  # 16 x 16 VQ codes
        config.action_dims = 8
        config.proprio_dims = 15
        config.eola_token = -1  # denotes end of latent action
        config.max_n_frames = -1
        config.img_aug = False
        config.vertical_flip = False
        config.vqgan_checkpoint_path = ""
        config.vision_path = ""
        config.image_absolute_path = ""
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields_from_example != "", "fields_from_example must be specified."
        assert self.config.eola_token != -1, "eola_token must be specified."
        self.tokenizer = tokenizer
        self.vision_start = tokenizer.encode("<vision>")
        self.vision_end = tokenizer.encode("</vision>")
        self.dynamics_start = tokenizer.encode("<dynamics>")
        self.dynamics_end = tokenizer.encode("</dynamics>")
        self.proprio_start = tokenizer.encode("<proprio>")
        self.proprio_end = tokenizer.encode("</proprio>")
        self.action_start = tokenizer.encode("<action>")
        self.action_end = tokenizer.encode("</action>")
        self.preprocessor = get_preprocessor(image_aug=self.config.img_aug, flip_vertically=self.config.vertical_flip)
        self.vision_dict = dict()
        if self.config.vision_path != "":
            with open(self.config.vision_path, "r") as fh:
                for line in fh:
                    data = json.loads(line)
                    self.vision_dict[data["image_id"]] = data["image_encodings"]
            logging.info(f"Using cached vision encodings from {self.config.vision_path}")
        else:
            assert self.config.vqgan_checkpoint_path != "", "vqgan_checkpoint_path must be specified."
            logging.info("No vision encodings provided. Will encode images on the fly.")
            self.vqgan = VQGAN(self.config.vqgan_checkpoint_path, replicate=False)

    def __call__(self, example, has_aux=False, add_bos_token=True, add_eos_token=True):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        rand_state = random.Random(aux[-1] + aux[-2])  # makes augmentations deterministic by line number + epoch
        try:
            token_buffer = []
            loss_mask_buffer = []
            vision_mask = []
            latent_action_mask = []
            latent_state_mask = []
            proprio_mask = []
            action_mask = []
            proprio = []
            action_chunk = []

            fields = example[self.config.fields_from_example]
            if isinstance(fields, (tuple, list)):
                if self.config.fields_index >= 0:
                    fields = fields[self.config.fields_index]
                else:
                    # seed based on line number
                    fields = rand_state.choice(fields)
            fields = fields.split(",")

            if add_bos_token and self.config.add_bos_token:
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(0.0)
                vision_mask.append(False)
                latent_action_mask.append(False)
                latent_state_mask.append(False)
                proprio_mask.append(False)
                action_mask.append(False)
                proprio.append([0.0] * self.config.proprio_dims)
                action_chunk.append([0.0] * self.config.action_dims)

            for i, field in enumerate(fields):
                if field.startswith("[") and field.endswith("]"):
                    # No loss for this field.
                    field = field[1:-1].strip()
                    mask = 0.0
                else:
                    mask = 1.0

                if field == "<|bos|>":
                    token_buffer.append(self.tokenizer.bos_token_id)
                    loss_mask_buffer.append(mask)
                    vision_mask.append(False)
                    latent_action_mask.append(False)
                    latent_state_mask.append(False)
                    proprio_mask.append(False)
                    action_mask.append(False)
                    proprio.append([0.0] * self.config.proprio_dims)
                    action_chunk.append([0.0] * self.config.action_dims)
                elif field == "<|eos|>":
                    token_buffer.append(self.tokenizer.eos_token_id)
                    loss_mask_buffer.append(mask)
                    vision_mask.append(False)
                    latent_action_mask.append(False)
                    latent_state_mask.append(False)
                    proprio_mask.append(False)
                    action_mask.append(False)
                    proprio.append([0.0] * self.config.proprio_dims)
                    action_chunk.append([0.0] * self.config.action_dims)
                elif "vision" in field:
                    # Check if latent_state is present
                    if_vision_end = True
                    for key in fields:
                        if "latent_state" in key:
                            if_vision_end = False
                            break
                    dataset = extract_dataset_from_id(example["id"])
                    if self.vision_dict:
                        # Generate proper cache keys using the same logic as cache generation
                        image_ids = []
                        for img_path in example["image"]:
                            cache_key = image_path_to_cache_key(dataset, img_path)
                            image_ids.append(cache_key)
                        encoded_images = []
                        for img_id in image_ids:
                            encoded_img = [int(x) for x in self.vision_dict[img_id]]
                            encoded_images.append(encoded_img)
                        encoded_images = np.array(encoded_images)
                    else:
                        image_paths = [
                            os.path.join(self.config.image_absolute_path, img_path) for img_path in example["image"]
                        ]
                        processed_images = process_images_batch(image_paths, self.preprocessor)
                        encoded_images = encode_images(self.vqgan, processed_images)
                    vision_tokens = encoded_images.flatten().tolist()
                    n_frames = int(len(vision_tokens) / self.config.n_tokens_per_frame)
                    if self.config.max_n_frames > 0 and n_frames > self.config.max_n_frames:  # uniformly select
                        idxs = np.linspace(0, n_frames - 1, self.config.max_n_frames).astype(int)
                        new_vision_tokens = []
                        for idx in idxs:
                            new_vision_tokens.extend(
                                vision_tokens[
                                    idx * self.config.n_tokens_per_frame : (idx + 1) * self.config.n_tokens_per_frame
                                ]
                            )
                        vision_tokens = new_vision_tokens
                        n_frames = self.config.max_n_frames
                    assert int(len(vision_tokens) / self.config.n_tokens_per_frame) == n_frames, (
                        int(len(vision_tokens) / self.config.n_tokens_per_frame),
                        n_frames,
                    )

                    assert n_frames > 0, len(vision_tokens)
                    tokens = list(self.vision_start)
                    for j in range(n_frames):
                        tokens.extend(
                            vision_tokens[j * self.config.n_tokens_per_frame : (j + 1) * self.config.n_tokens_per_frame]
                        )
                        if if_vision_end and j == n_frames - 1:
                            tokens.append(self.config.eov_token)
                        else:
                            tokens.append(self.config.eof_token)

                    if if_vision_end:
                        tokens.extend(self.vision_end)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    vision_mask.extend([False] * len(self.vision_start))
                    vision_mask.extend(
                        [True] * (self.config.n_tokens_per_frame * n_frames + n_frames)
                    )  # include extra eof token at the end of each frame
                    if if_vision_end:
                        vision_mask.extend([False] * len(self.vision_end))
                    latent_action_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    proprio_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))
                    proprio.extend([[0.0] * self.config.proprio_dims for _ in range(len(tokens))])
                    action_chunk.extend([[0.0] * self.config.action_dims for _ in range(len(tokens))])
                elif "latent_action" in field:
                    # cummulative latent action
                    latent_action_tokens = example[field + "_idxs"]
                    tokens = list(self.dynamics_start)  # start of dynamics
                    n_latent_actions = int(len(latent_action_tokens) / self.config.n_tokens_per_latent_action)
                    for idx in range(n_latent_actions):
                        tokens.extend(
                            latent_action_tokens[
                                idx
                                * self.config.n_tokens_per_latent_action : (idx + 1)
                                * self.config.n_tokens_per_latent_action
                            ]
                        )
                        tokens.append(self.config.eola_token)
                    tokens.extend(self.dynamics_end)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    latent_action_mask.extend([False] * len(self.dynamics_start))
                    latent_action_mask.extend(
                        [True] * (self.config.n_tokens_per_latent_action * n_latent_actions + n_latent_actions)
                    )  # include extra eola token
                    latent_action_mask.extend([False] * len(self.dynamics_end))
                    vision_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    proprio_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))
                    proprio.extend([[0.0] * self.config.proprio_dims for _ in range(len(tokens))])
                    action_chunk.extend([[0.0] * self.config.action_dims for _ in range(len(tokens))])
                elif "latent_state" in field:
                    # similar to vision
                    dataset = extract_dataset_from_id(example["id"])
                    if self.vision_dict:
                        # Generate proper cache keys using the same logic as cache generation
                        latent_image_ids = []
                        for ls_path in example[field]:
                            cache_key = image_path_to_cache_key(dataset, ls_path)
                            latent_image_ids.append(cache_key)
                        encoded_latent_images = []
                        for img_id in latent_image_ids:
                            encoded_img = [int(x) for x in self.vision_dict[img_id]]
                            encoded_latent_images.append(encoded_img)
                        encoded_latent_images = np.array(encoded_latent_images)
                    else:
                        latent_image_path = [
                            os.path.join(self.config.image_absolute_path, dataset, ls_path)
                            for ls_path in example[field]
                        ]
                        processed_latent_images = process_images_batch(latent_image_path, self.preprocessor)
                        encoded_latent_images = encode_images(self.vqgan, processed_latent_images)
                    latent_state_tokens = encoded_latent_images.flatten().tolist()
                    n_latent_states = int(len(latent_state_tokens) / self.config.n_tokens_per_latent_state)
                    tokens = []  # make sure continue the vision token sequence here and end with eov
                    for idx in range(n_latent_states):
                        tokens.extend(
                            latent_state_tokens[
                                idx
                                * self.config.n_tokens_per_latent_state : (idx + 1)
                                * self.config.n_tokens_per_latent_state
                            ]
                        )
                        if idx == n_latent_states - 1:
                            tokens.append(self.config.eov_token)
                        else:
                            tokens.append(self.config.eof_token)
                    tokens.extend(self.vision_end)  # end of vision
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    latent_state_mask.extend(
                        [True] * (self.config.n_tokens_per_latent_state * n_latent_states + n_latent_states)
                    )  # include extra eof/eov token
                    latent_state_mask.extend([False] * len(self.vision_end))
                    latent_action_mask.extend([False] * len(tokens))
                    vision_mask.extend([False] * len(tokens))
                    proprio_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))
                    proprio.extend([[0.0] * self.config.proprio_dims for _ in range(len(tokens))])
                    action_chunk.extend([[0.0] * self.config.action_dims for _ in range(len(tokens))])
                elif "action" in field:
                    action_list = [float(x) for x in example[field]]
                    tokens = list(self.action_start)
                    action_chunk.extend([[0.0] * self.config.action_dims for _ in range(len(self.action_start))])
                    n_actions = int(len(action_list) / self.config.action_dims)
                    for idx in range(n_actions):
                        action_chunk.append(
                            action_list[idx * self.config.action_dims : (idx + 1) * self.config.action_dims]
                        )
                        tokens.append(int(idx))
                    tokens.extend(self.action_end)
                    action_chunk.extend([[0.0] * self.config.action_dims for _ in range(len(self.action_end))])
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    action_mask.extend([False] * len(self.action_start))
                    action_mask.extend([True] * n_actions)  # include extra eoa token
                    action_mask.extend([False] * len(self.action_end))
                    vision_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    latent_action_mask.extend([False] * len(tokens))
                    proprio_mask.extend([False] * len(tokens))
                    proprio.extend([[0.0] * self.config.proprio_dims for _ in range(len(tokens))])
                elif "proprio" in field:
                    proprio_list = [float(x) for x in example[field]]
                    tokens = list(self.proprio_start)
                    proprio.extend([[0.0] * self.config.proprio_dims for _ in range(len(self.proprio_start))])
                    n_proprios = int(len(proprio_list) / self.config.proprio_dims)
                    for idx in range(n_proprios):
                        proprio.append(
                            proprio_list[idx * self.config.proprio_dims : (idx + 1) * self.config.proprio_dims]
                        )
                        tokens.append(int(idx))
                    tokens.extend(self.proprio_end)
                    proprio.extend([[0.0] * self.config.proprio_dims for _ in range(len(self.proprio_end))])
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    proprio_mask.extend([False] * len(self.proprio_start))
                    proprio_mask.extend([True] * n_proprios)
                    proprio_mask.extend([False] * len(self.proprio_end))
                    vision_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    latent_action_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))
                    action_chunk.extend([[0.0] * self.config.action_dims for _ in range(len(tokens))])
                else:
                    subfields = field.split("+")
                    text = self.config.subfield_separator.join([example[subfield] for subfield in subfields])
                    if i == 0:
                        text = self.config.prepend_text + text
                    tokens = self.tokenizer.encode(text)
                    token_buffer.extend(tokens)
                    loss_mask_buffer.extend([mask for _ in range(len(tokens))])
                    vision_mask.extend([False] * len(tokens))
                    latent_action_mask.extend([False] * len(tokens))
                    latent_state_mask.extend([False] * len(tokens))
                    proprio_mask.extend([False] * len(tokens))
                    action_mask.extend([False] * len(tokens))
                    proprio.extend([[0.0] * self.config.proprio_dims for _ in range(len(tokens))])
                    action_chunk.extend([[0.0] * self.config.action_dims for _ in range(len(tokens))])

            if add_eos_token and self.config.add_eos_token:
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(1.0)
                vision_mask.append(False)
                latent_action_mask.append(False)
                latent_state_mask.append(False)
                proprio_mask.append(False)
                action_mask.append(False)
                proprio.append([0.0] * self.config.proprio_dims)
                action_chunk.append([0.0] * self.config.action_dims)
            assert (
                len(token_buffer)
                == len(loss_mask_buffer)
                == len(vision_mask)
                == len(latent_action_mask)
                == len(latent_state_mask)
                == len(action_chunk)
                == len(action_mask)
                == len(proprio)
                == len(proprio_mask)
            )
            keep = True
            return (
                token_buffer,
                proprio,
                action_chunk,
                loss_mask_buffer,
                vision_mask,
                latent_action_mask,
                latent_state_mask,
                proprio_mask,
                action_mask,
                keep,
                *aux,
            )
        except Exception as e:
            logging.error(f"Error processing example {example.get('id', 'unknown')}: {e}")
            return [], [], [], [], [], [], [], [], [], False, *aux


class JsonDynamicsDataset(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""
        config.seq_length = 2048
        config.batch_size = 4
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        config.use_data_sharded_loader = True
        config.return_local_batch = False
        config.valid = False
        config.mode = "pad"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor, node_info, seed):
        self.config = self.get_default_config(config)
        assert self.config.path != ""
        self._node_info = node_info
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._seed = seed
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self.valid = self.config.valid
        self._total_tokens = 0
        self._num_complete_passes = 0  # Number of complete passes through the dataset
        self._data = self._load_and_shuffle_data()
        self._shuffle_for_pass(0)
        logging.info(
            f"Using a batch size of {self.config.batch_size} and a sequence length of {self.config.seq_length}"
        )

    def _shuffle_for_pass(self, pass_id):
        rng = random.Random((self._seed, pass_id))
        rng.shuffle(self._data)

    def _load_and_shuffle_data(self):
        data = []
        logging.info("Loading data from %s", self.config.path)
        with open(self.config.path, "r") as fin:
            line_count = 0
            while True:
                loc = fin.tell()
                line = fin.readline()
                if not line:
                    break
                data.append((line, loc))
                line_count += 1
                if line_count % 10000 == 0:
                    logging.info(f"Lines read: {line_count}")
        logging.info("Finished loading data")
        return data

    def parse_json(self, line):
        if not line or line == "\n":
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            logging.error(f"Error parsing json line:\n{line}")
            return None
        return data

    def json_iterator(self):
        current_index = self._index
        if self._num_complete_passes > 0:
            # restore proper order for the current pass
            # shuffle happens at the end of each pass
            for n_pass in range(1, self._num_complete_passes + 1):
                self._shuffle_for_pass(n_pass)
        while True:
            for index, (line, loc) in enumerate(self._data[current_index + 1 :], start=current_index + 1):
                data = self.parse_json(line)
                if data is not None:
                    yield data, loc, index, self._num_complete_passes
            current_index = -1
            self._num_complete_passes += 1
            self._shuffle_for_pass(self._num_complete_passes)

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index, num_complete_passes in self.json_iterator():
                self._file_loc = loc
                self._index = index
                yield self.text_processor((example, loc, index, num_complete_passes), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(self.json_iterator(), self.config.tokenizer_parallel_batch_size)
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator), chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator), chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        if self.config.mode == "pad":
            fn = self._iter_pad
        elif self.config.mode == "no_pad":
            fn = self._iter_no_pad
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        return fn()

    def _iter_pad(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info["dp_node_size"]
        else:
            local_batch_size = self.config.batch_size
        last_time = 0.0
        buffer = []
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for (
            tokens,
            loss_masks,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            keep,
            loc,
            index,
            num_complete_passes,
        ) in self.parallel_example_iterator():
            if not keep:
                continue
            self._file_loc = loc
            self._index = index
            buffer.append((tokens, loss_masks, vision_masks, latent_action_masks, latent_state_masks))
            while len(buffer) >= local_batch_size:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size :]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (self._total_tokens - start_tokens) / (time.time() - start_time)
                metrics = {
                    "dataset_file_loc": loc,
                    "dataset_example_index": index,
                    "dataset_total_tokens": self._total_tokens,
                    "dataset_accumulated_tps": accumulated_throughput,
                    "dataset_average_tps": average_throughput,
                }

                batch = {
                    "input_tokens": np.full(
                        (local_batch_size, self.config.seq_length), self._tokenizer.bos_token_id, dtype=np.int32
                    ),
                    "target_tokens": np.full(
                        (local_batch_size, self.config.seq_length), self._tokenizer.bos_token_id, dtype=np.int32
                    ),
                    "loss_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=np.float32),
                    "input_vision_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_vision_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "input_latent_action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_latent_action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "input_latent_state_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_latent_state_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                }
                for i in range(local_batch_size):
                    tokens, loss_masks, vision_masks, latent_action_masks, latent_state_masks = buffer[i]
                    if len(tokens) > self.config.seq_length:
                        tokens = tokens[: self.config.seq_length + 1]
                        loss_masks = loss_masks[1 : self.config.seq_length + 1]
                        vision_masks = vision_masks[: self.config.seq_length + 1]
                        latent_action_masks = latent_action_masks[: self.config.seq_length + 1]
                        latent_state_masks = latent_state_masks[: self.config.seq_length + 1]
                    input_tokens, target_tokens = tokens[:-1], tokens[1:]
                    input_vision_masks, target_vision_masks = vision_masks[:-1], vision_masks[1:]
                    input_latent_action_masks, target_latent_action_masks = (
                        latent_action_masks[:-1],
                        latent_action_masks[1:],
                    )
                    input_latent_state_masks, target_latent_state_masks = (
                        latent_state_masks[:-1],
                        latent_state_masks[1:],
                    )
                    loss_masks = loss_masks[1:]
                    batch["input_tokens"][i, : len(input_tokens)] = input_tokens
                    batch["target_tokens"][i, : len(target_tokens)] = target_tokens
                    batch["input_vision_masks"][i, : len(input_vision_masks)] = input_vision_masks
                    batch["target_vision_masks"][i, : len(target_vision_masks)] = target_vision_masks
                    batch["input_latent_action_masks"][i, : len(input_latent_action_masks)] = input_latent_action_masks
                    batch["target_latent_action_masks"][
                        i, : len(target_latent_action_masks)
                    ] = target_latent_action_masks
                    batch["input_latent_state_masks"][i, : len(input_latent_state_masks)] = input_latent_state_masks
                    batch["target_latent_state_masks"][i, : len(target_latent_state_masks)] = target_latent_state_masks
                    batch["loss_masks"][i, : len(loss_masks)] = loss_masks

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info["mesh"]
                    sp_nodes_size = max(1, mesh.shape["sp"] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {
                        k: v[:, sp_nodes_rank * seq_chunk_size : (sp_nodes_rank + 1) * seq_chunk_size]
                        for k, v in batch.items()
                    }
                    batch = host_local_array_to_global_array(batch, self._node_info["mesh"], PS(("dp", "fsdp"), "sp"))
                yield batch, metrics
                buffer = buffer[local_batch_size:]

    def _iter_no_pad(self):
        global_chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info["dp_node_size"]
        else:
            local_batch_size = self.config.batch_size
        chunk_size = local_batch_size * self.config.seq_length

        token_buffer = []
        loss_mask_buffer = []
        vision_mask_buffer = []
        latent_action_mask_buffer = []
        latent_state_mask_buffer = []

        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for (
            tokens,
            loss_masks,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            keep,
            loc,
            index,
            num_complete_passes,
        ) in self.parallel_example_iterator():
            if not keep:
                continue
            self._file_loc = loc
            self._index = index
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            vision_mask_buffer.extend(vision_masks)
            latent_action_mask_buffer.extend(latent_action_masks)
            latent_state_mask_buffer.extend(latent_state_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += global_chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size :]
                average_throughput = global_chunk_size / np.mean(step_times)
                accumulated_throughput = (self._total_tokens - start_tokens) / (time.time() - start_time)
                metrics = {
                    "dataset_file_loc": loc,
                    "dataset_example_index": index,
                    "dataset_total_tokens": self._total_tokens,
                    "dataset_accumulated_tps": accumulated_throughput,
                    "dataset_average_tps": average_throughput,
                }
                batch = {
                    "input_tokens": np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(local_batch_size, -1),
                    "target_tokens": np.array(token_buffer[1 : chunk_size + 1], dtype=np.int32).reshape(
                        local_batch_size, -1
                    ),
                    "loss_masks": np.array(loss_mask_buffer[1 : chunk_size + 1], dtype=np.float32).reshape(
                        local_batch_size, -1
                    ),
                    "input_vision_masks": np.array(vision_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_vision_masks": np.array(vision_mask_buffer[1 : chunk_size + 1], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "input_latent_action_masks": np.array(latent_action_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_latent_action_masks": np.array(
                        latent_action_mask_buffer[1 : chunk_size + 1], dtype=bool
                    ).reshape(local_batch_size, -1),
                    "input_latent_state_masks": np.array(latent_state_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_latent_state_masks": np.array(
                        latent_state_mask_buffer[1 : chunk_size + 1], dtype=bool
                    ).reshape(local_batch_size, -1),
                }

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info["mesh"]
                    sp_nodes_size = max(1, mesh.shape["sp"] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {
                        k: v[:, sp_nodes_rank * seq_chunk_size : (sp_nodes_rank + 1) * seq_chunk_size]
                        for k, v in batch.items()
                    }
                    batch = host_local_array_to_global_array(batch, self._node_info["mesh"], PS(("dp", "fsdp"), "sp"))

                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]
                vision_mask_buffer = vision_mask_buffer[chunk_size:]
                latent_action_mask_buffer = latent_action_mask_buffer[chunk_size:]
                latent_state_mask_buffer = latent_state_mask_buffer[chunk_size:]

    def _make_callback(self, v):
        return lambda index: v[index]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
            num_complete_passes=self._num_complete_passes,
        )

    def load_state_dict(self, state_dict):
        if "config" in state_dict:
            batch_size = self.config.batch_size
            self.config.update(ConfigDict(state_dict["config"]))
            logging.info(f"Found batch size {self.config.batch_size} in state dict")
            self.config.batch_size = batch_size
            logging.info(f"Using a batch size of {self.config.batch_size}")
        self._index = state_dict.get("index", self.config.example_index_at_start)
        self._file_loc = state_dict.get("file_loc", self.config.start_seek_loc)
        self._total_tokens = state_dict.get("total_tokens", self.config.tokens_count_at_start)
        self._num_complete_passes = state_dict.get("num_complete_passes", 0)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def batch_size(self):
        return self.config.batch_size

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDynamicsActionQuantDataset(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""
        config.seq_length = 2048
        config.batch_size = 4
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        config.use_data_sharded_loader = True
        config.return_local_batch = False
        config.valid = False
        config.mode = "pad"
        config.n_tokens_per_action = 7

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor, node_info):
        self.config = self.get_default_config(config)
        assert self.config.path != ""
        self._node_info = node_info
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self.valid = self.config.valid
        self._total_tokens = 0
        self._num_complete_passes = 0  # Number of complete passes through the dataset
        self._data = self._load_and_shuffle_data()
        self._shuffle_for_pass(0)
        logging.info(
            f"Using a batch size of {self.config.batch_size} and a sequence length of {self.config.seq_length}"
        )

    def _shuffle_for_pass(self, pass_id):
        rng = random.Random((self._seed, pass_id))
        rng.shuffle(self._data)

    def _load_and_shuffle_data(self):
        data = []
        logging.info("Loading data from %s", self.config.path)
        with open(self.config.path, "r") as fin:
            line_count = 0
            while True:
                loc = fin.tell()
                line = fin.readline()
                if not line:
                    break
                data.append((line, loc))
                line_count += 1
                if line_count % 10000 == 0:
                    logging.info(f"Lines read: {line_count}")
        logging.info("Finished loading data")
        return data

    def parse_json(self, line):
        if not line or line == "\n":
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            logging.error(f"Error parsing json line:\n{line}")
            return None
        return data

    def json_iterator(self):
        current_index = self._index
        if self._num_complete_passes > 0:
            # restore proper order for the current pass
            # shuffle happens at the end of each pass
            for n_pass in range(1, self._num_complete_passes + 1):
                self._shuffle_for_pass(n_pass)
        while True:
            for index, (line, loc) in enumerate(self._data[current_index + 1 :], start=current_index + 1):
                data = self.parse_json(line)
                if data is not None:
                    yield data, loc, index, self._num_complete_passes
            current_index = -1
            self._num_complete_passes += 1
            self._shuffle_for_pass(self._num_complete_passes)

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index, num_complete_passes in self.json_iterator():
                self._file_loc = loc
                self._index = index
                yield self.text_processor((example, loc, index, num_complete_passes), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(self.json_iterator(), self.config.tokenizer_parallel_batch_size)
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator), chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator), chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        if self.config.mode == "pad":
            fn = self._iter_pad
        elif self.config.mode == "no_pad":
            fn = self._iter_no_pad
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        return fn()

    def _iter_pad(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info["dp_node_size"]
        else:
            local_batch_size = self.config.batch_size
        last_time = 0.0
        buffer = []
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for (
            tokens,
            loss_masks,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            action_masks,
            keep,
            loc,
            index,
            num_complete_passes,
        ) in self.parallel_example_iterator():
            if not keep:
                continue
            self._file_loc = loc
            self._index = index
            buffer.append((tokens, loss_masks, vision_masks, latent_action_masks, latent_state_masks, action_masks))
            while len(buffer) >= local_batch_size:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size :]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (self._total_tokens - start_tokens) / (time.time() - start_time)
                metrics = {
                    "dataset_file_loc": loc,
                    "dataset_example_index": index,
                    "dataset_total_tokens": self._total_tokens,
                    "dataset_accumulated_tps": accumulated_throughput,
                    "dataset_average_tps": average_throughput,
                }

                batch = {
                    "input_tokens": np.full(
                        (local_batch_size, self.config.seq_length), self._tokenizer.bos_token_id, dtype=np.int32
                    ),
                    "target_tokens": np.full(
                        (local_batch_size, self.config.seq_length), self._tokenizer.bos_token_id, dtype=np.int32
                    ),
                    "loss_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=np.float32),
                    "input_vision_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_vision_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "input_latent_action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_latent_action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "input_latent_state_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_latent_state_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "input_action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                }
                for i in range(local_batch_size):
                    tokens, loss_masks, vision_masks, latent_action_masks, latent_state_masks, action_masks = buffer[i]
                    if len(tokens) > self.config.seq_length:
                        tokens = tokens[: self.config.seq_length + 1]
                        loss_masks = loss_masks[1 : self.config.seq_length + 1]
                        vision_masks = vision_masks[: self.config.seq_length + 1]
                        latent_action_masks = latent_action_masks[: self.config.seq_length + 1]
                        latent_state_masks = latent_state_masks[: self.config.seq_length + 1]
                        action_masks = action_masks[: self.config.seq_length + 1]
                    input_tokens, target_tokens = tokens[:-1], tokens[1:]
                    input_vision_masks, target_vision_masks = vision_masks[:-1], vision_masks[1:]
                    input_latent_action_masks, target_latent_action_masks = (
                        latent_action_masks[:-1],
                        latent_action_masks[1:],
                    )
                    input_latent_state_masks, target_latent_state_masks = (
                        latent_state_masks[:-1],
                        latent_state_masks[1:],
                    )
                    input_action_masks, target_action_masks = action_masks[:-1], action_masks[1:]
                    loss_masks = loss_masks[1:]
                    batch["input_tokens"][i, : len(input_tokens)] = input_tokens
                    batch["target_tokens"][i, : len(target_tokens)] = target_tokens
                    batch["input_vision_masks"][i, : len(input_vision_masks)] = input_vision_masks
                    batch["target_vision_masks"][i, : len(target_vision_masks)] = target_vision_masks
                    batch["input_latent_action_masks"][i, : len(input_latent_action_masks)] = input_latent_action_masks
                    batch["target_latent_action_masks"][
                        i, : len(target_latent_action_masks)
                    ] = target_latent_action_masks
                    batch["input_latent_state_masks"][i, : len(input_latent_state_masks)] = input_latent_state_masks
                    batch["target_latent_state_masks"][i, : len(target_latent_state_masks)] = target_latent_state_masks
                    batch["input_action_masks"][i, : len(input_action_masks)] = input_action_masks
                    batch["target_action_masks"][i, : len(target_action_masks)] = target_action_masks
                    batch["loss_masks"][i, : len(loss_masks)] = loss_masks

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info["mesh"]
                    sp_nodes_size = max(1, mesh.shape["sp"] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {
                        k: v[:, sp_nodes_rank * seq_chunk_size : (sp_nodes_rank + 1) * seq_chunk_size]
                        for k, v in batch.items()
                    }
                    batch = host_local_array_to_global_array(batch, self._node_info["mesh"], PS(("dp", "fsdp"), "sp"))
                yield batch, metrics
                buffer = buffer[local_batch_size:]

    def _iter_no_pad(self):
        global_chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info["dp_node_size"]
        else:
            local_batch_size = self.config.batch_size
        chunk_size = local_batch_size * self.config.seq_length

        token_buffer = []
        loss_mask_buffer = []
        vision_mask_buffer = []
        latent_action_mask_buffer = []
        latent_state_mask_buffer = []
        action_mask_buffer = []

        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for (
            tokens,
            loss_masks,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            action_masks,
            keep,
            loc,
            index,
            num_complete_passes,
        ) in self.parallel_example_iterator():
            if not keep:
                continue
            self._file_loc = loc
            self._index = index
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            vision_mask_buffer.extend(vision_masks)
            latent_action_mask_buffer.extend(latent_action_masks)
            latent_state_mask_buffer.extend(latent_state_masks)
            action_mask_buffer.extend(action_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += global_chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size :]
                average_throughput = global_chunk_size / np.mean(step_times)
                accumulated_throughput = (self._total_tokens - start_tokens) / (time.time() - start_time)
                metrics = {
                    "dataset_file_loc": loc,
                    "dataset_example_index": index,
                    "dataset_total_tokens": self._total_tokens,
                    "dataset_accumulated_tps": accumulated_throughput,
                    "dataset_average_tps": average_throughput,
                }
                batch = {
                    "input_tokens": np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(local_batch_size, -1),
                    "target_tokens": np.array(token_buffer[1 : chunk_size + 1], dtype=np.int32).reshape(
                        local_batch_size, -1
                    ),
                    "loss_masks": np.array(loss_mask_buffer[1 : chunk_size + 1], dtype=np.float32).reshape(
                        local_batch_size, -1
                    ),
                    "input_vision_masks": np.array(vision_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_vision_masks": np.array(vision_mask_buffer[1 : chunk_size + 1], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "input_action_masks": np.array(action_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_action_masks": np.array(action_mask_buffer[1 : chunk_size + 1], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "input_latent_action_masks": np.array(latent_action_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_latent_action_masks": np.array(
                        latent_action_mask_buffer[1 : chunk_size + 1], dtype=bool
                    ).reshape(local_batch_size, -1),
                    "input_latent_state_masks": np.array(latent_state_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_latent_state_masks": np.array(
                        latent_state_mask_buffer[1 : chunk_size + 1], dtype=bool
                    ).reshape(local_batch_size, -1),
                }

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info["mesh"]
                    sp_nodes_size = max(1, mesh.shape["sp"] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {
                        k: v[:, sp_nodes_rank * seq_chunk_size : (sp_nodes_rank + 1) * seq_chunk_size]
                        for k, v in batch.items()
                    }
                    batch = host_local_array_to_global_array(batch, self._node_info["mesh"], PS(("dp", "fsdp"), "sp"))

                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]
                vision_mask_buffer = vision_mask_buffer[chunk_size:]
                latent_action_mask_buffer = latent_action_mask_buffer[chunk_size:]
                latent_state_mask_buffer = latent_state_mask_buffer[chunk_size:]
                action_mask_buffer = action_mask_buffer[chunk_size:]

    def _make_callback(self, v):
        return lambda index: v[index]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
            num_complete_passes=self._num_complete_passes,
        )

    def load_state_dict(self, state_dict):
        if "config" in state_dict:
            batch_size = self.config.batch_size
            self.config.update(ConfigDict(state_dict["config"]))
            logging.info(f"Found batch size {self.config.batch_size} in state dict")
            self.config.batch_size = batch_size
            logging.info(f"Using a batch size of {self.config.batch_size}")
        self._index = state_dict.get("index", self.config.example_index_at_start)
        self._file_loc = state_dict.get("file_loc", self.config.start_seek_loc)
        self._total_tokens = state_dict.get("total_tokens", self.config.tokens_count_at_start)
        self._num_complete_passes = state_dict.get("num_complete_passes", 0)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def batch_size(self):
        return self.config.batch_size

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDynamicsActionContDataset(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""
        config.seq_length = 2048
        config.batch_size = 4
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        config.use_data_sharded_loader = True
        config.return_local_batch = False
        config.valid = False
        config.mode = "pad"
        config.action_chunk_size = 14
        config.action_dims = 16
        config.proprio_dims = 15
        config.proprio_history_len = 7

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor, node_info, seed):
        self.config = self.get_default_config(config)
        assert self.config.path != ""
        self._node_info = node_info
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._seed = seed
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self.valid = self.config.valid
        self._total_tokens = 0
        self._num_complete_passes = 0  # Number of complete passes through the dataset
        self._data = self._load_and_shuffle_data()
        self._shuffle_for_pass(0)
        logging.info(
            f"Using a batch size of {self.config.batch_size} and a sequence length of {self.config.seq_length}"
        )

    def _shuffle_for_pass(self, pass_id):
        rng = random.Random((self._seed, pass_id))
        rng.shuffle(self._data)

    def _load_and_shuffle_data(self):
        data = []
        logging.info("Loading data from %s", self.config.path)
        with open(self.config.path, "r") as fin:
            line_count = 0
            while True:
                loc = fin.tell()
                line = fin.readline()
                if not line:
                    break
                data.append((line, loc))
                line_count += 1
                if line_count % 10000 == 0:
                    logging.info(f"Lines read: {line_count}")
        logging.info("Finished loading data")
        return data

    def parse_json(self, line):
        if not line or line == "\n":
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            logging.error(f"Error parsing json line:\n{line}")
            return None
        return data

    def json_iterator(self):
        current_index = self._index
        if self._num_complete_passes > 0:
            # restore proper order for the current pass
            # shuffle happens at the end of each pass
            for n_pass in range(1, self._num_complete_passes + 1):
                self._shuffle_for_pass(n_pass)
        while True:
            for index, (line, loc) in enumerate(self._data[current_index + 1 :], start=current_index + 1):
                data = self.parse_json(line)
                if data is not None:
                    yield data, loc, index, self._num_complete_passes
            current_index = -1
            self._num_complete_passes += 1
            self._shuffle_for_pass(self._num_complete_passes)

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index, num_complete_passes in self.json_iterator():
                self._file_loc = loc
                self._index = index
                yield self.text_processor((example, loc, index, num_complete_passes), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(self.json_iterator(), self.config.tokenizer_parallel_batch_size)
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator), chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator), chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        if self.config.mode == "pad":
            fn = self._iter_pad
        elif self.config.mode == "no_pad":
            fn = self._iter_no_pad
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
        return fn()

    def _iter_pad(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info["dp_node_size"]
        else:
            local_batch_size = self.config.batch_size
        last_time = 0.0
        buffer = []
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for (
            tokens,
            proprio,
            action_chunk,
            loss_masks,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            proprio_masks,
            action_masks,
            keep,
            loc,
            index,
            num_complete_passes,
        ) in self.parallel_example_iterator():
            if not keep:
                continue
            self._file_loc = loc
            self._index = index
            buffer.append(
                (
                    tokens,
                    proprio,
                    action_chunk,
                    loss_masks,
                    vision_masks,
                    latent_action_masks,
                    latent_state_masks,
                    proprio_masks,
                    action_masks,
                )
            )
            while len(buffer) >= local_batch_size:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size :]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (self._total_tokens - start_tokens) / (time.time() - start_time)
                metrics = {
                    "dataset_file_loc": loc,
                    "dataset_example_index": index,
                    "dataset_total_tokens": self._total_tokens,
                    "dataset_accumulated_tps": accumulated_throughput,
                    "dataset_average_tps": average_throughput,
                }

                batch = {
                    "input_tokens": np.full(
                        (local_batch_size, self.config.seq_length), self._tokenizer.bos_token_id, dtype=np.int32
                    ),
                    "target_tokens": np.full(
                        (local_batch_size, self.config.seq_length), self._tokenizer.bos_token_id, dtype=np.int32
                    ),
                    "loss_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=np.float32),
                    "input_vision_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_vision_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "input_latent_action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_latent_action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "input_latent_state_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "target_latent_state_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "proprio_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "action_masks": np.zeros((local_batch_size, self.config.seq_length), dtype=bool),
                    "proprio": np.zeros(
                        (local_batch_size, self.config.seq_length, self.config.proprio_dims), dtype="f4"
                    ),
                    "actions": np.zeros(
                        (local_batch_size, self.config.seq_length, self.config.action_dims), dtype="f4"
                    ),
                }
                for i in range(local_batch_size):
                    (
                        tokens,
                        proprio,
                        action_chunk,
                        loss_masks,
                        vision_masks,
                        latent_action_masks,
                        latent_state_masks,
                        proprio_masks,
                        action_masks,
                    ) = buffer[i]
                    if len(tokens) > self.config.seq_length:
                        tokens = tokens[: self.config.seq_length + 1]
                        loss_masks = loss_masks[1 : self.config.seq_length + 1]
                        vision_masks = vision_masks[: self.config.seq_length + 1]
                        latent_action_masks = latent_action_masks[: self.config.seq_length + 1]
                        latent_state_masks = latent_state_masks[: self.config.seq_length + 1]
                        proprio_masks = proprio_masks[: self.config.seq_length + 1]
                        action_masks = action_masks[: self.config.seq_length + 1]
                        proprio = proprio[: self.config.seq_length + 1]
                        action_chunk = action_chunk[: self.config.seq_length + 1]
                    input_tokens, target_tokens = tokens[:-1], tokens[1:]
                    input_vision_masks, target_vision_masks = vision_masks[:-1], vision_masks[1:]
                    input_latent_action_masks, target_latent_action_masks = (
                        latent_action_masks[:-1],
                        latent_action_masks[1:],
                    )
                    input_latent_state_masks, target_latent_state_masks = (
                        latent_state_masks[:-1],
                        latent_state_masks[1:],
                    )
                    loss_masks = loss_masks[1:]
                    batch["input_tokens"][i, : len(input_tokens)] = input_tokens
                    batch["target_tokens"][i, : len(target_tokens)] = target_tokens
                    batch["input_vision_masks"][i, : len(input_vision_masks)] = input_vision_masks
                    batch["target_vision_masks"][i, : len(target_vision_masks)] = target_vision_masks
                    batch["input_latent_action_masks"][i, : len(input_latent_action_masks)] = input_latent_action_masks
                    batch["target_latent_action_masks"][
                        i, : len(target_latent_action_masks)
                    ] = target_latent_action_masks
                    batch["input_latent_state_masks"][i, : len(input_latent_state_masks)] = input_latent_state_masks
                    batch["target_latent_state_masks"][i, : len(target_latent_state_masks)] = target_latent_state_masks
                    batch["loss_masks"][i, : len(loss_masks)] = loss_masks
                    batch["proprio_masks"][i, : len(proprio_masks)] = proprio_masks
                    batch["proprio"][i, : len(proprio)] = np.array(proprio)
                    # action is not next token, only reconstruction
                    batch["actions"][i, : len(action_chunk)] = np.array(action_chunk)
                    batch["action_masks"][i, : len(action_masks)] = action_masks

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info["mesh"]
                    sp_nodes_size = max(1, mesh.shape["sp"] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {
                        k: v[:, sp_nodes_rank * seq_chunk_size : (sp_nodes_rank + 1) * seq_chunk_size]
                        for k, v in batch.items()
                    }
                    batch = host_local_array_to_global_array(batch, self._node_info["mesh"], PS(("dp", "fsdp"), "sp"))
                yield batch, metrics
                buffer = buffer[local_batch_size:]

    def _iter_no_pad(self):
        global_chunk_size = self.config.batch_size * self.config.seq_length
        if self.config.use_data_sharded_loader:
            local_batch_size = self.config.batch_size // self._node_info["dp_node_size"]
        else:
            local_batch_size = self.config.batch_size
        chunk_size = local_batch_size * self.config.seq_length

        token_buffer = []
        loss_mask_buffer = []
        vision_mask_buffer = []
        latent_action_mask_buffer = []
        latent_state_mask_buffer = []
        proprio_mask_buffer = []
        action_mask_buffer = []
        proprio_buffer = []
        action_chunk_buffer = []

        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for (
            tokens,
            proprio,
            action_chunk,
            loss_masks,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            proprio_masks,
            action_masks,
            keep,
            loc,
            index,
            num_complete_passes,
        ) in self.parallel_example_iterator():
            if not keep:
                continue
            self._file_loc = loc
            self._index = index
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            vision_mask_buffer.extend(vision_masks)
            latent_action_mask_buffer.extend(latent_action_masks)
            latent_state_mask_buffer.extend(latent_state_masks)
            proprio_mask_buffer.extend(proprio_masks)
            proprio_buffer.extend(proprio)
            action_mask_buffer.extend(action_masks)
            action_chunk_buffer.append(action_chunk)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += global_chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size :]
                average_throughput = global_chunk_size / np.mean(step_times)
                accumulated_throughput = (self._total_tokens - start_tokens) / (time.time() - start_time)
                metrics = {
                    "dataset_file_loc": loc,
                    "dataset_example_index": index,
                    "dataset_total_tokens": self._total_tokens,
                    "dataset_accumulated_tps": accumulated_throughput,
                    "dataset_average_tps": average_throughput,
                }
                batch = {
                    "input_tokens": np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(local_batch_size, -1),
                    "target_tokens": np.array(token_buffer[1 : chunk_size + 1], dtype=np.int32).reshape(
                        local_batch_size, -1
                    ),
                    "loss_masks": np.array(loss_mask_buffer[1 : chunk_size + 1], dtype=np.float32).reshape(
                        local_batch_size, -1
                    ),
                    "input_vision_masks": np.array(vision_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_vision_masks": np.array(vision_mask_buffer[1 : chunk_size + 1], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "input_latent_action_masks": np.array(latent_action_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_latent_action_masks": np.array(
                        latent_action_mask_buffer[1 : chunk_size + 1], dtype=bool
                    ).reshape(local_batch_size, -1),
                    "input_latent_state_masks": np.array(latent_state_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "target_latent_state_masks": np.array(
                        latent_state_mask_buffer[1 : chunk_size + 1], dtype=bool
                    ).reshape(local_batch_size, -1),
                    "proprio_masks": np.array(proprio_mask_buffer[:chunk_size], dtype=bool).reshape(
                        local_batch_size, -1
                    ),
                    "proprio": np.array(proprio_buffer[:chunk_size], dtype="f4").reshape(
                        local_batch_size, -1, self.config.proprio_dims
                    ),
                    "action_masks": np.array(action_mask_buffer[:chunk_size], dtype=bool).reshape(local_batch_size, -1),
                    "actions": np.array(action_chunk_buffer[:chunk_size], dtype="f4").reshape(
                        local_batch_size, -1, self.action_dims
                    ),
                }

                if self.config.use_data_sharded_loader and not self.config.return_local_batch:
                    mesh = self._node_info["mesh"]
                    sp_nodes_size = max(1, mesh.shape["sp"] // jax.local_device_count())
                    sp_nodes_rank = jax.process_index() % sp_nodes_size
                    assert self.config.seq_length % sp_nodes_size == 0, (self.config.seq_len, sp_nodes_size)
                    seq_chunk_size = self.config.seq_length // sp_nodes_size
                    batch = {
                        k: v[:, sp_nodes_rank * seq_chunk_size : (sp_nodes_rank + 1) * seq_chunk_size]
                        for k, v in batch.items()
                    }
                    batch = host_local_array_to_global_array(batch, self._node_info["mesh"], PS(("dp", "fsdp"), "sp"))

                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]
                vision_mask_buffer = vision_mask_buffer[chunk_size:]
                latent_action_mask_buffer = latent_action_mask_buffer[chunk_size:]
                latent_state_mask_buffer = latent_state_mask_buffer[chunk_size:]
                proprio_mask_buffer = proprio_mask_buffer[chunk_size:]
                proprio_buffer = proprio_buffer[chunk_size:]
                action_mask_buffer = action_mask_buffer[chunk_size:]
                action_chunk_buffer = action_chunk_buffer[chunk_size:]

    def _make_callback(self, v):
        return lambda index: v[index]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
            num_complete_passes=self._num_complete_passes,
        )

    def load_state_dict(self, state_dict):
        if "config" in state_dict:
            batch_size = self.config.batch_size
            self.config.update(ConfigDict(state_dict["config"]))
            logging.info(f"Found batch size {self.config.batch_size} in state dict")
            self.config.batch_size = batch_size
            logging.info(f"Using a batch size of {self.config.batch_size}")
        self._index = state_dict.get("index", self.config.example_index_at_start)
        self._file_loc = state_dict.get("file_loc", self.config.start_seek_loc)
        self._total_tokens = state_dict.get("total_tokens", self.config.tokens_count_at_start)
        self._num_complete_passes = state_dict.get("num_complete_passes", 0)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def action_chunk_size(self):
        return self.config.action_chunk_size

    @property
    def batch_size(self):
        return self.config.batch_size

    @property
    def action_dims(self):
        return self.config.action_dims

    @property
    def proprio_dims(self):
        return self.config.proprio_dims

    @property
    def proprio_history_len(self):
        return self.config.proprio_history_len

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self._tokenizer)
