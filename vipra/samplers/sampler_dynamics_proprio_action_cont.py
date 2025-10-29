from functools import cached_property

import absl.logging as logging
import albumentations
import albumentations as A
import cv2
import jax
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from PIL import Image
from policy.dynamics_proprio_action_cont_llama import FlaxVideoLLaMAForCausalLM, VideoLLaMAConfig
from policy.vqgan import VQGAN
from transformers import GenerationConfig
from tux import (
    JaxDistributedConfig,
    JaxRNG,
    StreamingCheckpointer,
    define_flags_with_default,
    get_float_dtype_by_name,
    make_shard_and_gather_fns,
    match_partition_rules,
    next_rng,
    tree_apply,
    with_sharding_constraint,
)


class CenterCropThenResize(ImageOnlyTransform):
    def __init__(self, height, width, scale, ratio, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(CenterCropThenResize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def apply(self, img, **params):
        # Calculate the size of the crop
        crop_height = int(self.height * self.scale[0])
        crop_width = int(self.width * self.ratio[0])

        # Calculate center crop coordinates
        y1 = (img.shape[0] - crop_height) // 2
        x1 = (img.shape[1] - crop_width) // 2
        y2 = y1 + crop_height
        x2 = x1 + crop_width

        # Crop the image
        cropped_img = img[y1:y2, x1:x2]

        # Resize the cropped image back to the desired dimensions
        return A.Resize(self.height, self.width).apply(cropped_img, interpolation=self.interpolation)


def get_image_preprocessor(image_aug, vertical_flip):
    transform_list = []
    if vertical_flip:
        transform_list.append(albumentations.VerticalFlip(p=1.0))
    if image_aug:
        transform_list.extend(
            [
                albumentations.LongestMaxSize(max_size=256),  # Resize the longest side to 256
                albumentations.Resize(256, 256),
                CenterCropThenResize(
                    height=256, width=256, scale=[0.9, 0.9], ratio=[1.0, 1.0], interpolation=cv2.INTER_LINEAR
                ),
            ]
        )
    else:
        transform_list.extend(
            [
                albumentations.LongestMaxSize(max_size=256),  # Resize the longest side to 256
                albumentations.Resize(256, 256),
            ]
        )
    preprocessor = albumentations.Compose(transform_list)
    return preprocessor


class ProprioActionContSampler:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.mesh = VideoLLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
        self.vqgan = VQGAN(FLAGS.vqgan_checkpoint, replicate=False)
        self.prefix_tokenizer = VideoLLaMAConfig.get_tokenizer(
            FLAGS.tokenizer, truncation_side="left", padding_side="left"
        )
        self.tokenizer = VideoLLaMAConfig.get_tokenizer(FLAGS.tokenizer)
        self.eola_token = FLAGS.eola_token
        self.eov_token = 8193
        self.num_inference_steps = FLAGS.num_inference_steps
        self.proprio_history_len = FLAGS.proprio_history_len
        self.proprio_dims = FLAGS.proprio_dims
        self.action_chunk_size = FLAGS.action_chunk_size
        self.action_dims = FLAGS.action_dims
        self.image_preprocessor = get_image_preprocessor(FLAGS.image_aug, getattr(FLAGS, "vertical_flip", False))
        self.sharded_rng = next_rng()
        self._load_model()

    @property
    def block_size(self):
        return max(self.config.scan_query_chunk_size, self.config.scan_key_chunk_size) * self.mesh.shape["sp"]

    @property
    def data_dim(self):
        return self.mesh.shape["dp"] * self.mesh.shape["fsdp"]

    def _process_frame(self, images):
        image_vqgan_list = []
        for image in images:
            img_array = np.array(image).astype(np.uint8)

            image_vqgan = self.image_preprocessor(image=img_array)["image"]
            image_vqgan = (image_vqgan / 127.5 - 1.0).astype(np.float32)
            image_vqgan_list.append(image_vqgan[None])
        image_vqgan_list = np.concatenate(image_vqgan_list, axis=0)
        return image_vqgan_list

    def encoding_to_image(self, encodings):
        if encodings.shape[1] > 256:
            encodings = encodings[:, :256]
        encodings = encodings.reshape(-1, 16, 16)
        image = self.vqgan.decode(encodings)[0]
        image = np.array(image)
        image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _read_process_vision(self, images):

        vision = self._process_frame(images)

        B = 1
        encodings = []
        for i in range(0, len(vision), 1):
            v = vision[i : i + B]
            if len(v) % B == 0:
                n_pad = 0
            else:
                n_pad = B - len(v) % B
            v = np.pad(v, ((n_pad, 0), (0, 0), (0, 0), (0, 0)))
            enc = jax.device_get(self.vqgan.encode(v))[1].astype(int)
            enc = enc[n_pad:]
            for t in range(len(enc)):
                encodings.extend(enc[t].reshape(-1).tolist())
        return encodings

    def construct_input(self, prompts):
        for i, prompt in enumerate(prompts):
            image_list = prompt["image"]
            tokens, vm = [], []
            for j, img in enumerate(image_list):
                vision = self._read_process_vision([img])
                tokens.extend(vision)
                vm.extend([True] * len(vision))
                tokens.extend([8192])
                vm.extend([True] * len([8192]))
        return {
            "input_ids": np.expand_dims(tokens, axis=0),
        }

    def _load_model(self):
        if self.FLAGS.load_llama_config != "":
            llama_config = VideoLLaMAConfig.load_config(self.FLAGS.load_llama_config)
            updates = VideoLLaMAConfig(**self.FLAGS.llama)
            llama_config.update(
                dict(
                    remat_block=updates.remat_block,
                    remat_attention=updates.remat_attention,
                    remat_mlp=updates.remat_mlp,
                    scan_attention=updates.scan_attention,
                    scan_mlp=updates.scan_mlp,
                    scan_query_chunk_size=updates.scan_query_chunk_size,
                    scan_key_chunk_size=updates.scan_key_chunk_size,
                    scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
                    scan_layers=updates.scan_layers,
                    param_scan_axis=updates.param_scan_axis,
                )
            )
        else:
            llama_config = VideoLLaMAConfig(**self.FLAGS.llama)

        if self.FLAGS.update_llama_config != "":
            llama_config.update(dict(eval(self.FLAGS.update_llama_config)))

        llama_config.update(
            dict(
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        )
        llama_config.update(dict(mesh_dim=self.FLAGS.mesh_dim))
        self.config = llama_config

        with jax.default_device(jax.devices("cpu")[0]):
            _, self.params = StreamingCheckpointer.load_trainstate_checkpoint(
                self.FLAGS.load_checkpoint, disallow_trainstate=True, max_buffer_size=32 * 2**30
            )
            self.model = FlaxVideoLLaMAForCausalLM(
                llama_config,
                input_shape=(1, 768),
                seed=self.FLAGS.seed,
                _do_init=False,
                dtype=get_float_dtype_by_name(self.FLAGS.dtype),
            )

            self.model_ps = match_partition_rules(
                VideoLLaMAConfig.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis),
                self.params,
            )
            shard_fns, _ = make_shard_and_gather_fns(self.model_ps, get_float_dtype_by_name(self.FLAGS.dtype))

            with self.mesh:
                self.params = tree_apply(shard_fns, self.params)

    @cached_property
    def _forward_generate(self):
        def fn(params, rng, batch):
            batch = with_sharding_constraint(batch, PS(("dp", "fsdp"), "sp"))
            rng_generator = JaxRNG(rng)

            self.model.config.sample_mode = "all"
            action_output = self.model.generate_action_kv_cache(
                batch["input_ids"],
                proprio=batch["proprio"],
                vision_masks=batch["vision_masks"],
                attention_mask=batch["attention_mask"],
                latent_action_masks=batch["latent_action_masks"],
                latent_state_masks=batch["latent_state_masks"],
                proprio_masks=batch["proprio_masks"],
                action_masks=batch["action_masks"],
                params=params["params"],
                prng_key=rng_generator(),
                num_inference_steps=self.num_inference_steps,
                integration_method="euler",
            )
            return action_output, rng_generator()

        return pjit(
            fn,
            in_shardings=(self.model_ps, PS(), PS()),
            out_shardings=(PS(), PS()),
        )
        # return fn

    def generate_action_pred(self, prompts, proprio, images, max_input_length):

        sharded_rng = next_rng()
        inputs = self.prefix_tokenizer(
            prompts, padding="max_length", truncation=True, max_length=max_input_length, return_tensors="np"
        )
        vision_end_proprio_start = ["</vision> <proprio>"] * len(prompts)
        vision_end_proprio_start_inputs = self.prefix_tokenizer(vision_end_proprio_start, return_tensors="np")
        dummy_proprio_ids = np.arange(self.proprio_history_len)
        dummy_proprio_ids = np.tile(dummy_proprio_ids[None, :], (len(prompts), 1))
        prefix_for_gen = ["</proprio> <action>"] * len(prompts)
        inputs_for_gen = self.prefix_tokenizer(prefix_for_gen, return_tensors="np")

        images[:, -1] = 8193  # set the last token to eov token
        batch = dict(
            input_ids=np.concatenate(
                [
                    inputs.input_ids,
                    images,
                    vision_end_proprio_start_inputs.input_ids,
                    dummy_proprio_ids,
                    inputs_for_gen.input_ids,
                ],
                axis=1,
            ),
            attention_mask=np.concatenate(
                [
                    inputs.attention_mask,
                    np.ones(images.shape, dtype=inputs.attention_mask.dtype),
                    vision_end_proprio_start_inputs.attention_mask,
                    np.ones(dummy_proprio_ids.shape, dtype=inputs.attention_mask.dtype),
                    inputs_for_gen.attention_mask,
                ],
                axis=1,
            ),
            vision_masks=np.concatenate(
                [
                    np.zeros(inputs.input_ids.shape, dtype=bool),
                    np.ones(images.shape, dtype=bool),
                    np.zeros(vision_end_proprio_start_inputs.input_ids.shape, dtype=bool),
                    np.zeros(dummy_proprio_ids.shape, dtype=bool),
                    np.zeros(inputs_for_gen.input_ids.shape, dtype=bool),
                ],
                axis=1,
            ),
            latent_action_masks=np.concatenate(
                [
                    np.zeros(inputs.input_ids.shape, dtype=bool),
                    np.zeros(images.shape, dtype=bool),
                    np.zeros(vision_end_proprio_start_inputs.input_ids.shape, dtype=bool),
                    np.zeros(dummy_proprio_ids.shape, dtype=bool),
                    np.zeros(inputs_for_gen.input_ids.shape, dtype=bool),
                ],
                axis=1,
            ),
            latent_state_masks=np.concatenate(
                [
                    np.zeros(inputs.input_ids.shape, dtype=bool),
                    np.zeros(images.shape, dtype=bool),
                    np.zeros(vision_end_proprio_start_inputs.input_ids.shape, dtype=bool),
                    np.zeros(dummy_proprio_ids.shape, dtype=bool),
                    np.zeros(inputs_for_gen.input_ids.shape, dtype=bool),
                ],
                axis=1,
            ),
            proprio_masks=np.concatenate(
                [
                    np.zeros(inputs.input_ids.shape, dtype=bool),
                    np.zeros(images.shape, dtype=bool),
                    np.zeros(vision_end_proprio_start_inputs.input_ids.shape, dtype=bool),
                    np.ones(dummy_proprio_ids.shape, dtype=bool),
                    np.zeros(inputs_for_gen.input_ids.shape, dtype=bool),
                ],
                axis=1,
            ),
            action_masks=np.concatenate(
                [
                    np.zeros(inputs.input_ids.shape, dtype=bool),
                    np.zeros(images.shape, dtype=bool),
                    np.zeros(vision_end_proprio_start_inputs.input_ids.shape, dtype=bool),
                    np.zeros(dummy_proprio_ids.shape, dtype=bool),
                    np.zeros(inputs_for_gen.input_ids.shape, dtype=bool),
                ],
                axis=1,
            ),
            proprio=np.concatenate(
                [
                    np.zeros(
                        (inputs.input_ids.shape[0], inputs.input_ids.shape[1], self.proprio_dims), dtype=np.float32
                    ),
                    np.zeros((images.shape[0], images.shape[1], self.proprio_dims), dtype=np.float32),
                    np.zeros(
                        (
                            vision_end_proprio_start_inputs.input_ids.shape[0],
                            vision_end_proprio_start_inputs.input_ids.shape[1],
                            self.proprio_dims,
                        ),
                        dtype=np.float32,
                    ),
                    proprio,
                    np.zeros(
                        (inputs_for_gen.input_ids.shape[0], inputs_for_gen.input_ids.shape[1], self.proprio_dims),
                        dtype=np.float32,
                    ),
                ],
                axis=1,
            ),
        )
        with self.mesh:
            action_output, sharded_rng = self._forward_generate(
                self.params,
                sharded_rng,
                batch,
            )
            action_output = jax.device_get(action_output)
        action_output_mutable = action_output.copy()[0]
        return action_output_mutable

    def __call__(self, prompts):
        batch = self.construct_input(prompts)
        logging.info(prompts[0]["question"])
        if isinstance(prompts[0]["image"][0], Image.Image):
            prompts[0]["image"][0].save("image1.jpg")
            prompts[0]["image"][1].save("image2.jpg")
        else:
            Image.fromarray(prompts[0]["image"][0]).save("image1.jpg")
            Image.fromarray(prompts[0]["image"][1]).save("image2.jpg")
        text_prompt = f"<s> <s> You are a helpful assistant. USER: What action should the robot take to `{prompts[0]['question']}` ASSISTANT: <vision>"
        proprio = np.array(prompts[0]["proprio"], dtype=np.float32)[
            None, :, :
        ]  # (1, proprio_history_len, proprio_dims)
        action_output = self.generate_action_pred(
            prompts=[text_prompt], proprio=proprio, images=batch["input_ids"], max_input_length=128
        )
        return action_output
