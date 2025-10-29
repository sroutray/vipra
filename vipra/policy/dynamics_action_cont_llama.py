import copy
import json
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.sharding import PartitionSpec as PS
from policy.llama import LLAMA_STANDARD_CONFIGS, FlaxLLaMABlockCollection, LLaMAConfig, RMSNorm
from transformers import GenerationConfig
from transformers.generation.flax_utils import FlaxLogitsProcessorList, FlaxSampleOutput, SampleState, logger
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from tux import load_pickle, open_file

VIDEO_LLAMA_STANDARD_CONFIGS = LLAMA_STANDARD_CONFIGS


class SinusoidalPosEmb(nn.Module):
    dim: int
    max_period: float = 10000.0

    def setup(self) -> None:
        self.half_dim = self.dim // 2
        self.inv_freq = jnp.exp(
            -jnp.log(self.max_period) * jnp.arange(self.half_dim) / (self.half_dim - 1),
        ).astype(jnp.float32)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        emb: jnp.ndarray = t[:, None] * self.inv_freq[None, :]
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


class ActionEncoder(nn.Module):
    action_dim: int
    width: int
    time_cond: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self) -> None:
        self.linear_1 = nn.Dense(
            self.width // 2,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            bias_init=jax.nn.initializers.zeros,
        )
        self.linear_2 = nn.Dense(
            self.width // 2 if not self.time_cond else self.width,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            bias_init=jax.nn.initializers.zeros,
        )
        self.linear_3 = nn.Dense(
            self.width,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            bias_init=jax.nn.initializers.zeros,
        )
        self.nonlinearity = nn.silu  # Swish activation

    def __call__(self, action: jnp.ndarray, time_emb: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # [Batch_Size, Seq_Len, Width]
        emb = self.linear_1(action)
        if self.time_cond and time_emb is not None:
            # repeat time embedding for seq_len
            # [Batch_Size, Seq_Len, Width]
            time_emb_full = jnp.expand_dims(time_emb, axis=1)
            time_emb_full = jnp.tile(time_emb_full, (1, action.shape[1], 1))
            emb = jnp.concatenate([time_emb_full, emb], axis=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb


class VideoLLaMAConfig(LLaMAConfig):
    model_type = "video_llama"

    def __init__(
        self,
        vision_vocab_size=8448,
        tie_vision_embeddings=False,
        latent_action_vocab_size=32,
        action_dims=8,
        action_chunk_size=14,
        sample_mode="all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_vocab_size = vision_vocab_size  # 8192 + 256
        self.tie_vision_embeddings = tie_vision_embeddings
        self.sample_mode = sample_mode
        self.latent_action_vocab_size = latent_action_vocab_size  # last token is separator token
        self.action_dims = action_dims
        self.action_chunk_size = action_chunk_size

    @staticmethod
    def get_partition_rules(scan_layers=False, scan_axis=0):
        """Parition rules for GPTJ. Note that these rules are orderd, so that
        the beginning rules match first. It is important to use
        PartitionSpec() instead of None here because JAX does not treat
        None as a pytree leaf.
        """
        if scan_layers:
            if scan_axis == 0:
                return (
                    # embeddings
                    ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                    ("transformer/vte/embedding", PS("tp", ("fsdp", "sp"))),
                    ("transformer/late/embedding", PS("tp", ("fsdp", "sp"))),
                    # action encoder
                    ("transformer/action_encoder/linear_1/kernel", PS(None, "tp")),
                    ("transformer/action_encoder/linear_1/bias", PS("tp")),
                    ("transformer/action_encoder/linear_2/kernel", PS("tp", None)),
                    ("transformer/action_encoder/linear_2/bias", PS(None)),
                    ("transformer/action_encoder/linear_3/kernel", PS(None, ("fsdp", "sp"))),
                    ("transformer/action_encoder/linear_3/bias", PS(("fsdp", "sp"))),
                    # attention
                    ("attention/(wq|wk|wv)/kernel", PS(None, ("fsdp", "sp"), "tp")),
                    ("attention/wo/kernel", PS(None, "tp", ("fsdp", "sp"))),
                    # mlp
                    ("feed_forward/w1/kernel", PS(None, ("fsdp", "sp"), "tp")),
                    ("feed_forward/w2/kernel", PS(None, "tp", ("fsdp", "sp"))),
                    ("feed_forward/w3/kernel", PS(None, ("fsdp", "sp"), "tp")),
                    # layer norms
                    ("attention_norm/kernel", PS(None, None)),
                    ("ffn_norm/kernel", PS(None, None)),
                    # output head
                    ("transformer/ln_f/kernel", PS(None)),
                    ("lm_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ("vision_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ("latent_actions_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ("action_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ("action_head/bias", PS("tp")),
                    (".*", PS(None)),
                )
            elif scan_axis == 1:
                return (
                    # embeddings
                    ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                    ("transformer/vte/embedding", PS("tp", ("fsdp", "sp"))),
                    ("transformer/late/embedding", PS("tp", ("fsdp", "sp"))),
                    # action encoder
                    ("transformer/action_encoder/linear_1/kernel", PS(None, "tp")),
                    ("transformer/action_encoder/linear_1/bias", PS("tp")),
                    ("transformer/action_encoder/linear_2/kernel", PS("tp", None)),
                    ("transformer/action_encoder/linear_2/bias", PS(None)),
                    ("transformer/action_encoder/linear_3/kernel", PS(None, ("fsdp", "sp"))),
                    ("transformer/action_encoder/linear_3/bias", PS(("fsdp", "sp"))),
                    # attention
                    ("attention/(wq|wk|wv)/kernel", PS(("fsdp", "sp"), None, "tp")),
                    ("attention/wo/kernel", PS("tp", None, ("fsdp", "sp"))),
                    # mlp
                    ("feed_forward/w1/kernel", PS(("fsdp", "sp"), None, "tp")),
                    ("feed_forward/w2/kernel", PS("tp", None, ("fsdp", "sp"))),
                    ("feed_forward/w3/kernel", PS(("fsdp", "sp"), None, "tp")),
                    # layer norms
                    ("attention_norm/kernel", PS(None, None)),
                    ("ffn_norm/kernel", PS(None, None)),
                    # output head
                    ("transformer/ln_f/kernel", PS(None)),
                    ("lm_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ("vision_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ("latent_actions_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ("action_head/kernel", PS(("fsdp", "sp"), "tp")),
                    ("action_head/bias", PS("tp")),
                    (".*", PS(None)),
                )
            else:
                raise ValueError(f"Invalid scan_axis {scan_axis}")
        else:
            return (
                # embeddings
                ("transformer/wte/embedding", PS("tp", ("fsdp", "sp"))),
                ("transformer/vte/embedding", PS("tp", ("fsdp", "sp"))),
                ("transformer/late/embedding", PS("tp", ("fsdp", "sp"))),
                # action encoder
                ("transformer/action_encoder/linear_1/kernel", PS(None, "tp")),
                ("transformer/action_encoder/linear_1/bias", PS("tp")),
                ("transformer/action_encoder/linear_2/kernel", PS("tp", None)),
                ("transformer/action_encoder/linear_2/bias", PS(None)),
                ("transformer/action_encoder/linear_3/kernel", PS(None, ("fsdp", "sp"))),
                ("transformer/action_encoder/linear_3/bias", PS(("fsdp", "sp"))),
                # attention
                ("attention/(wq|wk|wv)/kernel", PS(("fsdp", "sp"), "tp")),
                ("attention/wo/kernel", PS("tp", ("fsdp", "sp"))),
                # mlp
                ("feed_forward/w1/kernel", PS(("fsdp", "sp"), "tp")),
                ("feed_forward/w2/kernel", PS("tp", ("fsdp", "sp"))),
                ("feed_forward/w3/kernel", PS(("fsdp", "sp"), "tp")),
                # layer norms
                ("attention_norm/kernel", PS(None)),
                ("ffn_norm/kernel", PS(None)),
                # output head
                ("transformer/ln_f/kernel", PS(None)),
                ("lm_head/kernel", PS(("fsdp", "sp"), "tp")),
                ("vision_head/kernel", PS(("fsdp", "sp"), "tp")),
                ("latent_actions_head/kernel", PS(("fsdp", "sp"), "tp")),
                ("action_head/kernel", PS(("fsdp", "sp"), "tp")),
                ("action_head/bias", PS("tp")),
                (".*", PS(None)),
            )

    @classmethod
    def load_config(cls, path):
        if path in VIDEO_LLAMA_STANDARD_CONFIGS:
            return cls.from_dict(VIDEO_LLAMA_STANDARD_CONFIGS[path])
        load_type, load_path = path.split("::", 1)
        if load_type == "pickle":
            return cls.from_dict(load_pickle(load_path)["llama_config"])
        elif load_type == "json":
            with open_file(load_path, "r") as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f"Unsupported load config type: {load_type}")


class FlaxVideoLLaMAPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VideoLLaMAConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: VideoLLaMAConfig,
        input_shape: Tuple = (4, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_cache(self, batch_size, max_length):
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        time = jnp.zeros(((batch_size,)), dtype="f4")
        actions = jnp.zeros((batch_size, max_length, self.config.action_dims), dtype="f4")
        attention_mask = jnp.ones_like(input_ids)
        segment_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        vision_masks = jnp.ones((batch_size, max_length), dtype=bool)
        latent_action_masks = jnp.ones((batch_size, max_length), dtype=bool)
        latent_state_masks = jnp.ones((batch_size, max_length), dtype=bool)
        action_masks = jnp.ones((batch_size, max_length), dtype=bool)

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            actions,
            time,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            action_masks,
            attention_mask,
            segment_ids,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return init_variables["cache"]

    def init_weights(self, rng, input_shape, params=None):
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        time = jnp.zeros(((input_shape[0],)), dtype="f4")
        actions = jnp.zeros((input_shape[0], input_shape[1], self.config.action_dims), dtype="f4")
        attention_mask = jnp.ones_like(input_ids)
        vision_masks = jnp.ones(input_ids.shape, dtype=bool)
        latent_action_masks = jnp.ones(input_ids.shape, dtype=bool)
        latent_state_masks = jnp.ones(input_ids.shape, dtype=bool)
        action_masks = jnp.ones(input_ids.shape, dtype=bool)
        segment_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(
            rngs,
            input_ids,
            actions,
            time,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            action_masks,
            attention_mask,
            segment_ids,
            position_ids,
            return_dict=False,
        )["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        actions,
        time,
        vision_masks,
        latent_action_masks,
        latent_state_masks,
        action_masks,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        init_cache: bool = False,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        if segment_ids is None:
            segment_ids = jnp.zeros((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTJAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(actions, dtype="f4"),
            jnp.array(time, dtype="f4"),
            jnp.array(vision_masks, dtype="f4"),
            jnp.array(latent_action_masks, dtype="f4"),
            jnp.array(latent_state_masks, dtype="f4"),
            jnp.array(action_masks, dtype="f4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(segment_ids, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            init_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxVideoLLaMAModule(nn.Module):
    config: VideoLLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.embed_dim = self.config.hidden_size

        self.vte = nn.Embed(
            self.config.vision_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.late = nn.Embed(
            self.config.latent_action_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # Replace action_head with flow matching network
        self.time_emb = SinusoidalPosEmb(self.config.hidden_size // 2)
        self.action_encoder = ActionEncoder(
            self.config.action_dims,
            self.config.hidden_size,
            time_cond=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )

        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxLLaMABlockCollection(
            self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
        )
        self.ln_f = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype
        )

    def __call__(
        self,
        input_ids,
        actions,
        time,
        vision_masks,
        latent_action_masks,
        latent_state_masks,
        action_masks,
        attention_mask,
        segment_ids,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_ids = input_ids.astype("i4")

        if input_ids.shape[1] == 1 or (input_ids.shape[1] != 1 and self.config.sample_mode == "action"):
            if self.config.sample_mode == "text":
                input_embeds = self.wte(input_ids)
            elif self.config.sample_mode == "vision":
                input_embeds = self.vte(input_ids)
            elif self.config.sample_mode == "latent_action":
                input_embeds = self.late(input_ids)
            elif self.config.sample_mode == "latent_state":
                input_embeds = self.vte(input_ids)
            elif self.config.sample_mode == "action":
                input_embeds = self.action_encoder(actions, self.time_emb(time))
            elif self.config.sample_mode == "all":
                raise NotImplementedError
            else:
                raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")
        else:
            input_text_embeds = self.wte(jnp.where(vision_masks, 0, input_ids))
            input_vision_embeds = self.vte(jnp.where(vision_masks, input_ids, 0))
            input_latent_action_embeds = self.late(jnp.where(latent_action_masks, input_ids, 0))
            input_latent_state_embeds = self.vte(
                jnp.where(latent_state_masks, input_ids, 0)
            )  # use vision embeddings for latent states
            noisy_action_embeds = self.action_encoder(actions, self.time_emb(time))

            vision_masks = vision_masks[..., None].astype("f4")  # 1 is vision, 0 is text
            latent_action_masks = latent_action_masks[..., None].astype("f4")  # 1 is latent actions, 0 is others
            latent_state_masks = latent_state_masks[..., None].astype("f4")  # 1 is latent states, 0 is others
            action_masks = action_masks[..., None].astype("f4")  # 1 is action, 0 is others

            input_embeds = input_text_embeds * (1 - vision_masks) + input_vision_embeds * vision_masks
            input_embeds = input_embeds * (1 - latent_action_masks) + input_latent_action_embeds * latent_action_masks
            input_embeds = input_embeds * (1 - latent_state_masks) + input_latent_state_embeds * latent_state_masks
            input_embeds = input_embeds * (1 - action_masks) + noisy_action_embeds * action_masks

        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            segment_ids,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxDynamicsActionContLLaMAForCausalLMModule(nn.Module):
    config: VideoLLaMAConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]] = None

    def setup(self):
        self.transformer = FlaxVideoLLaMAModule(
            self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision
        )
        self.vision_head = nn.Dense(
            self.config.vision_vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
        self.latent_actions_head = nn.Dense(
            self.config.latent_action_vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
        self.action_head = nn.Dense(
            self.config.action_dims,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            bias_init=jax.nn.initializers.zeros,
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids,
        actions,
        time,
        vision_masks,
        latent_action_masks,
        latent_state_masks,
        action_masks,
        attention_mask=None,
        segment_ids=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if segment_ids is None:
            segment_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0), (batch_size, seq_length)
            )

        outputs = self.transformer(
            input_ids,
            actions,
            time,
            vision_masks,
            latent_action_masks,
            latent_state_masks,
            action_masks,
            attention_mask,
            segment_ids,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_vision_embeddings:
            shared_kernel = self.transformer.variables["params"]["vte"]["embedding"].T
            vision_logits = self.vision_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            vision_logits = self.vision_head(hidden_states)

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if self.config.tie_vision_embeddings:
            shared_kernel = self.transformer.variables["params"]["late"]["embedding"].T
            latent_action_logits = self.latent_actions_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            latent_action_logits = self.latent_actions_head(hidden_states)

        # latent states are predicted using vision embeddings
        if self.config.tie_vision_embeddings:
            shared_kernel = self.transformer.variables["params"]["vte"]["embedding"].T
            latent_state_logits = self.vision_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            latent_state_logits = self.vision_head(hidden_states)

        action_logits = self.action_head(hidden_states)

        if self.config.sample_mode == "all":
            if not return_dict:
                return (vision_logits, lm_logits, latent_action_logits, latent_state_logits, action_logits) + outputs[
                    1:
                ]

            return FlaxCausalLMOutput(
                logits=(vision_logits, lm_logits, latent_action_logits, latent_state_logits, action_logits),
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        elif self.config.sample_mode == "vision":
            if not return_dict:
                return (vision_logits,) + outputs[1:]

            return FlaxCausalLMOutput(
                logits=vision_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )
        elif self.config.sample_mode == "text":
            if not return_dict:
                return (lm_logits,) + outputs[1:]

            return FlaxCausalLMOutput(
                logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )
        elif self.config.sample_mode == "latent_action":
            if not return_dict:
                return (latent_action_logits,) + outputs[1:]

            return FlaxCausalLMOutput(
                logits=latent_action_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )
        elif self.config.sample_mode == "latent_state":
            if not return_dict:
                return (latent_state_logits,) + outputs[1:]

            return FlaxCausalLMOutput(
                logits=latent_state_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )
        elif self.config.sample_mode == "action":
            if not return_dict:
                return (action_logits,) + outputs[1:]

            return FlaxCausalLMOutput(
                logits=action_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )
        else:
            raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")


@add_start_docstrings("", "")
class FlaxVideoLLaMAForCausalLM(FlaxVideoLLaMAPreTrainedModel):
    module_class = FlaxDynamicsActionContLLaMAForCausalLMModule

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length,
        attention_mask: Optional[jax.Array] = None,
        vision_masks=None,
        latent_action_masks=None,
        latent_state_masks=None,
        action_masks=None,
        actions=None,
        time=None,
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPTJ uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
            "vision_masks": vision_masks,
            "latent_action_masks": latent_action_masks,
            "latent_state_masks": latent_state_masks,
            "action_masks": action_masks,
            "actions": actions,
            "time": time,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        return {
            "past_key_values": model_outputs.past_key_values,
            "position_ids": model_kwargs["position_ids"][:, -1:] + 1,
            "attention_mask": model_kwargs["attention_mask"],
            "vision_masks": model_kwargs["vision_masks"],
            "latent_action_masks": model_kwargs["latent_action_masks"],
            "latent_state_masks": model_kwargs["latent_state_masks"],
            "action_masks": model_kwargs["action_masks"],
            "actions": model_kwargs["actions"],
            "time": model_kwargs["time"],
        }

    def _sample(
        self,
        input_ids: None,
        n_tokens_per_sample: Optional[int] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        cfg_scales: jnp.ndarray = 1.0,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        assert n_tokens_per_sample is not None
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape
        initial_len = cur_len

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_p, top_k, temperature
            logits = logits_warper(logits, logits, state.cur_len)

            next_token = jax.random.categorical(prng_key, logits, axis=-1)
            if self.config.sample_mode == "vision" or self.config.sample_mode == "latent_state":
                next_token = jax.lax.cond(
                    (state.cur_len - initial_len + 1) % n_tokens_per_sample == 0,
                    lambda: jnp.full_like(next_token, pad_token_id),
                    lambda: next_token,
                )

            next_is_sent_finished = state.is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(sample_search_cond_fn, sample_search_body_fn, state)
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

        return FlaxSampleOutput(sequences=state.sequences)

    def generate(
        self,
        input_ids: jnp.ndarray,
        sample_mode: str,
        n_tokens_per_sample: int,
        cfg_scales: jnp.ndarray = 1.0,
        generation_config: Optional[GenerationConfig] = None,
        prng_key: Optional[jnp.ndarray] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        **kwargs,
    ):
        # Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        self.config.sample_mode = sample_mode

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
                self.generation_config
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        logits_processor = logits_processor if logits_processor is not None else FlaxLogitsProcessorList()

        # set init values
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask") is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        if generation_config.decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError("`decoder_start_token_id` has to be defined for encoder-decoder generation.")

        # decoder-only models should use left-padding for generation (can't be checked with `trace=True`)
        if not self.config.is_encoder_decoder and not trace:
            if (
                generation_config.pad_token_id is not None
                and jnp.sum(input_ids[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        batch_size = input_ids.shape[0]

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)
            # prepare decoder_input_ids for generation
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                model_kwargs=model_kwargs,
            )

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing`max_new_tokens`."
            )

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            logits_processor=logits_processor,
        )

        if not generation_config.do_sample and generation_config.num_beams == 1:
            logits_warper = self._get_logits_warper(generation_config=generation_config)
            return self._sample(
                input_ids,
                n_tokens_per_sample,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                cfg_scales=cfg_scales,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif generation_config.do_sample and generation_config.num_beams == 1:
            logits_warper = self._get_logits_warper(generation_config=generation_config)
            return self._sample(
                input_ids,
                n_tokens_per_sample,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                cfg_scales=cfg_scales,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif not generation_config.do_sample and generation_config.num_beams > 1:
            raise NotImplementedError
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")

    def generate_action_naive(
        self,
        condition_input_ids: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
        num_inference_steps: int = 20,
        prng_key: Optional[jnp.ndarray] = None,
        trace: bool = False,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        integration_method: Literal["euler", "rk2"] = "euler",
        **kwargs,
    ):
        """
        Naive action generation without caching.
        Performs full forward passes at each Euler step.

        Args:
            condition_input_ids: (B, T_cond)
            actions: (B, T_action, D) or None
            num_inference_steps: Integration steps
            prng_key: JAX RNG key
            trace: Use `lax.scan` or unrolled loop
            params: Optional model params
            integration_method: Integration method to use ('euler' or 'rk2')
            kwargs: attention_mask, vision_masks, etc.

        Returns:
            Final actions (B, T_action, D)
        """
        model = self
        self.config.sample_mode = "all"
        batch_size = condition_input_ids.shape[0]
        condition_length = condition_input_ids.shape[1]
        action_dim = self.config.action_dims

        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)

        # Init actions if not provided
        if actions is None:
            action_length = self.config.action_chunk_size
            prng_key, noise_key = jax.random.split(prng_key)
            actions = jax.random.normal(noise_key, (batch_size, action_length, action_dim), dtype=jnp.float32)
        else:
            action_length = actions.shape[1]

        max_length = condition_length + action_length
        position_ids = jnp.arange(max_length)[None, :].repeat(batch_size, axis=0)
        action_ids = jnp.broadcast_to(jnp.arange(action_length)[None, :], (batch_size, action_length))

        # Prepare full sequence tokens (condition + dummy)
        full_input_ids = jnp.concatenate([condition_input_ids, action_ids], axis=1)

        # Expand masks (assumes kwargs has the masks for condition part)
        def expand_and_concat(original, fill=0):
            new = jnp.full((batch_size, action_length), fill, dtype=original.dtype)
            return jnp.concatenate([original, new], axis=1)

        attention_mask = expand_and_concat(kwargs["attention_mask"], fill=1)
        vision_masks = expand_and_concat(kwargs["vision_masks"], fill=0)
        latent_action_masks = expand_and_concat(kwargs["latent_action_masks"], fill=0)
        latent_state_masks = expand_and_concat(kwargs["latent_state_masks"], fill=0)
        action_masks = expand_and_concat(kwargs["action_masks"], fill=1)

        # Init time
        time = jnp.zeros((batch_size,), dtype=jnp.float32)
        delta_t = 1.0 / num_inference_steps

        actions = lax.dynamic_update_slice(
            jnp.zeros((batch_size, max_length, action_dim)), actions, (0, condition_length, 0)
        )

        def model_step(actions_in, time_in):
            return model(
                full_input_ids,
                actions=actions_in,
                time=time_in,
                vision_masks=vision_masks,
                latent_action_masks=latent_action_masks,
                latent_state_masks=latent_state_masks,
                action_masks=action_masks,
                attention_mask=attention_mask,
                position_ids=position_ids,
                params=params,
                return_dict=True,
                init_cache=False,
            ).logits[-1][:, condition_length:, :]

        # Integration Methods
        def euler_step(state, _):
            current_action, current_time, key = state
            vel = model_step(current_action, current_time)
            updated = current_action[:, condition_length:, :] + delta_t * vel
            next_action = lax.dynamic_update_slice(current_action, updated, (0, condition_length, 0))
            return (next_action, current_time + delta_t, jax.random.split(key)[0]), None

        def rk2_step(state, _):
            current_action, current_time, key = state

            vel1 = model_step(current_action, current_time)
            tentative = current_action[:, condition_length:, :] + delta_t * vel1
            tentative_action = lax.dynamic_update_slice(current_action, tentative, (0, condition_length, 0))

            vel2 = model_step(tentative_action, current_time + delta_t)
            averaged_vel = 0.5 * (vel1 + vel2)
            updated = current_action[:, condition_length:, :] + delta_t * averaged_vel
            next_action = lax.dynamic_update_slice(current_action, updated, (0, condition_length, 0))

            return (next_action, current_time + delta_t, jax.random.split(key)[0]), None

        step_fn = rk2_step if integration_method == "rk2" else euler_step
        initial_state = (actions, time, prng_key)

        if trace:
            final_state, _ = jax.lax.scan(step_fn, initial_state, None, length=num_inference_steps)
        else:
            state = initial_state
            for _ in range(num_inference_steps):
                state, _ = step_fn(state, None)
            final_state = state

        final_actions = final_state[0][:, condition_length:, :]
        return final_actions

    def generate_action_kv_cache(
        self,
        condition_input_ids: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
        num_inference_steps: int = 10,
        prng_key: Optional[jnp.ndarray] = None,
        trace: bool = False,
        params: Optional[dict] = None,
        integration_method: Literal["euler", "rk2"] = "euler",
        **kwargs,
    ):
        """
        Action generation with KV cached condition inputs.
        Performs chunked forward passes at each Euler step.

        Args:
            condition_input_ids: (B, T_cond)
            actions: (B, T_action, D) or None
            num_inference_steps: Integration steps
            prng_key: JAX RNG key
            trace: Use `lax.scan` or unrolled loop
            params: Optional model params
            integration_method: Integration method to use ('euler' or 'rk2')
            kwargs: attention_mask, vision_masks, etc.

        Returns:
            Final actions (B, T_action, D)
        """
        model = self
        self.config.sample_mode = "all"
        batch_size = condition_input_ids.shape[0]
        condition_length = condition_input_ids.shape[1]
        action_dim = self.config.action_dims

        # Init actions if not provided
        if actions is None:
            action_length = self.config.action_chunk_size
            prng_key, noise_key = jax.random.split(prng_key)
            actions = jax.random.normal(noise_key, (batch_size, action_length, action_dim), dtype=jnp.float32)
        else:
            action_length = actions.shape[1]

        max_length = condition_length + action_length
        action_ids = jnp.broadcast_to(jnp.arange(action_length)[None, :], (batch_size, action_length))

        # Prepare full sequence tokens (condition + dummy)
        full_input_ids = jnp.concatenate([condition_input_ids, action_ids], axis=1)

        # Expand masks (assumes kwargs has the masks for condition part)
        def expand_and_concat(original, fill=0):
            new = jnp.full((batch_size, action_length), fill, dtype=original.dtype)
            return jnp.concatenate([original, new], axis=1)

        full_attention_mask = expand_and_concat(kwargs["attention_mask"], fill=1)
        full_vision_masks = expand_and_concat(kwargs["vision_masks"], fill=0)
        full_latent_action_masks = expand_and_concat(kwargs["latent_action_masks"], fill=0)
        full_latent_state_masks = expand_and_concat(kwargs["latent_state_masks"], fill=0)
        full_action_masks = expand_and_concat(kwargs["action_masks"], fill=1)

        # Init time
        time = jnp.zeros((batch_size,), dtype=jnp.float32)
        delta_t = 1.0 / num_inference_steps

        full_actions = lax.dynamic_update_slice(
            jnp.zeros((batch_size, max_length, action_dim)), actions, (0, condition_length, 0)
        )

        # Prepare the cache, masks, and position_ids
        init_state = self.prepare_inputs_for_generation(
            condition_input_ids,
            max_length=max_length,
            attention_mask=full_attention_mask,
            vision_masks=full_vision_masks,
            latent_action_masks=full_latent_action_masks,
            latent_state_masks=full_latent_state_masks,
            action_masks=full_action_masks,
            actions=actions,
            time=time,
        )
        full_position_ids = init_state["position_ids"]
        # jax.debug.breakpoint(num_frames=1)

        # Run first forward pass to get KV cache
        init_out = model(
            full_input_ids,
            actions=full_actions,
            time=time,
            vision_masks=full_vision_masks,
            latent_action_masks=full_latent_action_masks,
            latent_state_masks=full_latent_state_masks,
            action_masks=full_action_masks,
            attention_mask=full_attention_mask,
            position_ids=full_position_ids,
            init_cache=True,
            return_dict=True,
            params=params,
            past_key_values=init_state["past_key_values"],
        )
        vel = init_out.logits[-1][:, condition_length:, :]
        actions = actions + delta_t * vel
        time = time + delta_t
        prng_key, _ = jax.random.split(prng_key)

        # Adjust the cache_index back to condition_length
        orig_index = init_out.past_key_values["transformer"]["h"]["scan_decoder"]["attention"]["cache_index"]
        context_index = orig_index - action_length

        temp = unfreeze(init_out.past_key_values)
        temp["transformer"]["h"]["scan_decoder"]["attention"]["cache_index"] = context_index
        condition_cache = freeze(temp)
        condition_cache_index = context_index

        # Switch to action-only mode
        original_mode = self.config.sample_mode
        self.config.sample_mode = "action"

        gen_vision_masks = full_vision_masks[:, condition_length:]
        gen_latent_action_masks = full_latent_action_masks[:, condition_length:]
        gen_latent_state_masks = full_latent_state_masks[:, condition_length:]
        gen_action_masks = full_action_masks[:, condition_length:]
        gen_position_ids = full_position_ids[:, condition_length:]

        def euler_step(state, _):
            cur_actions, cur_time, key = state

            # Reset cache_index so we overwrite slots N..N+G-1
            past = unfreeze(condition_cache)
            past["transformer"]["h"]["scan_decoder"]["attention"]["cache_index"] = condition_cache_index
            past = freeze(past)

            out = self(
                action_ids,
                actions=cur_actions,
                time=cur_time,
                vision_masks=gen_vision_masks,
                latent_action_masks=gen_latent_action_masks,
                latent_state_masks=gen_latent_state_masks,
                action_masks=gen_action_masks,
                attention_mask=full_attention_mask,
                position_ids=gen_position_ids,
                init_cache=False,
                return_dict=True,
                params=params,
                past_key_values=past,
            )

            vel = out.logits  # (B, G, action_dims)
            next_actions = cur_actions + delta_t * vel
            next_time = cur_time + delta_t
            key, _ = jax.random.split(key)
            return (next_actions, next_time, key), None

        init_state = (actions, time, prng_key)
        if trace:
            (final_actions, _, _), _ = jax.lax.scan(euler_step, init_state, None, length=num_inference_steps - 1)
        else:
            state = init_state
            for _ in range(num_inference_steps - 1):
                state, _ = euler_step(state, None)
            final_actions, _, _ = state

        # Restore mode
        self.config.sample_mode = original_mode

        return final_actions
