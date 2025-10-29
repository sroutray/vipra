import os
import pprint
import random
import time

import absl.logging as logging
import flax
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import tux
from absl.app import run
from flax.serialization import from_bytes, from_state_dict, to_state_dict
from flax.training.train_state import TrainState
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from policy.data import DatasetFactory
from policy.dynamics_action_cont_llama import FlaxDynamicsActionContLLaMAForCausalLMModule, VideoLLaMAConfig
from policy.dynamics_action_quant_llama import FlaxDynamicsActionQuantLLaMAForCausalLMModule, VideoLLaMAConfig
from policy.dynamics_llama import FlaxDynamicsLLaMAForCausalLMModule, VideoLLaMAConfig
from policy.dynamics_proprio_action_cont_llama import (
    FlaxDynamicsProprioActionContLLaMAForCausalLMModule,
    VideoLLaMAConfig,
)
from policy.fm_sampler import FlowMatchingSamplerConfig, create_fm_time_sampler
from policy.llama import FlaxLLaMAForCausalLMModule, LLaMAConfig
from policy.utils import check_and_log_frozen_params, check_invalid_params, l1_loss, l2_loss, log_invalid_params
from tqdm import tqdm, trange
from tux import (
    JaxDistributedConfig,
    JaxRNG,
    OptimizerFactory,
    StreamingCheckpointer,
    average_metrics,
    cross_entropy_loss_and_accuracy,
    define_flags_with_default,
    get_float_dtype_by_name,
    get_mask,
    global_norm,
    make_shard_and_gather_fns,
    match_partition_rules,
    next_rng,
    set_random_seed,
    with_sharding_constraint,
)
from tux.utils import open_file

FLAGS, FLAGS_DEF = define_flags_with_default(
    modality="text",
    use_data_sharded_loader=True,
    seed=42,
    mesh_dim="1,-1,1,1",
    dtype="fp32",
    total_steps=10000,
    load_llama_config="",
    update_llama_config="",
    frozen_layers="",
    load_checkpoint="",
    load_dataset_state="",
    log_freq=10,
    eval_log_freq=10,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    flow_matching_sampler=FlowMatchingSamplerConfig.get_default_config(),
    tokenizer=VideoLLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    unseen_eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=VideoLLaMAConfig.get_default_config(),
    logger=tux.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    autoresume=False,
    freeze=0,
    mse_loss=1,
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = tux.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = tux.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    logger = tux.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    if jax.process_index() == 0:
        output_dir = logger.output_dir
    else:
        output_dir = os.path.join(logger.output_dir, logger.experiment_id)

    if FLAGS.modality == "vision,text,latent_action,latent_state":
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxDynamicsLLaMAForCausalLMModule
    elif FLAGS.modality == "vision,text,latent_action,latent_state,action_quant":
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxDynamicsActionQuantLLaMAForCausalLMModule
    elif FLAGS.modality == "vision,text,latent_action,latent_state,action_cont":
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxDynamicsActionContLLaMAForCausalLMModule
    elif FLAGS.modality == "vision,text,proprio,latent_action,latent_state,action_cont":
        config_cls = VideoLLaMAConfig
        llama_cls = FlaxDynamicsProprioActionContLLaMAForCausalLMModule
    else:
        raise ValueError(f"Unsupported modality: {FLAGS.modality}")

    mesh = config_cls.get_jax_mesh(FLAGS.mesh_dim)
    node_info = config_cls.get_ranks_and_size(mesh)

    tokenizer = config_cls.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer, node_info=node_info, seed=FLAGS.seed)
    dataset_resume_path = f"{output_dir}/dataset.pkl"
    if FLAGS.autoresume and tux.check_exists(dataset_resume_path):
        logging.info(f"Found existing output. Resuming dataset from latest checkpoint: {dataset_resume_path}")
        dataset.load_state_dict(tux.load_pickle(dataset_resume_path))
    elif FLAGS.load_dataset_state != "":
        logging.info(f"Loading dataset state from {FLAGS.load_dataset_state}")
        dataset.load_state_dict(tux.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(FLAGS.eval_dataset, dataset.tokenizer, node_info=node_info)
        eval_iterator = iter(eval_dataset)
        unseen_eval_dataset = DatasetFactory.load_dataset(
            FLAGS.unseen_eval_dataset, dataset.tokenizer, node_info=node_info
        )
        unseen_eval_iterator = iter(unseen_eval_dataset)

    seq_length = dataset.seq_length
    action_chunk_size = getattr(dataset, "action_chunk_size", 1)
    action_dims = getattr(dataset, "action_dims", 0)
    proprio_dims = getattr(dataset, "proprio_dims", 0)

    if FLAGS.load_llama_config != "":
        llama_config = config_cls.load_config(FLAGS.load_llama_config)
        updates = config_cls(**FLAGS.llama)
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
        llama_config = config_cls(**FLAGS.llama)

    if FLAGS.update_llama_config != "":
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(
        dict(
            bos_token_id=dataset.tokenizer.bos_token_id,
            eos_token_id=dataset.tokenizer.eos_token_id,
        )
    )
    if llama_config.vocab_size < dataset.vocab_size:
        llama_config.update(dict(vocab_size=dataset.vocab_size))
    llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    model = llama_cls(llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype))

    sample_fm_time = create_fm_time_sampler(FLAGS.flow_matching_sampler, dataset.batch_size)

    if FLAGS.frozen_layers != "":
        frozen_param_keys = FLAGS.frozen_layers.split(",")
        # Create mask: True = trainable, False = frozen
        # mask has to follow optax.masked(optax.set_to_zero(), frozen_param_mask)
        frozen_param_mask = get_mask(frozen_param_keys, tf_map={True: False, False: True})
    else:
        frozen_param_mask = None
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_mask(config_cls.get_weight_decay_exclusions()),
        frozen_param_mask=frozen_param_mask,
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        batch = dataset.batch_size
        if FLAGS.modality == "vision,text,latent_action,latent_state":
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                latent_action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                latent_state_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == "vision,text,latent_action,latent_state,action_quant":
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                latent_action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                latent_state_masks=jnp.zeros((batch, seq_length), dtype=bool),
                action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == "vision,text,latent_action,latent_state,action_cont":
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                actions=jnp.zeros((batch, seq_length, action_dims), dtype=jnp.float32),
                time=jnp.zeros((batch,), dtype=jnp.float32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                latent_action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                latent_state_masks=jnp.zeros((batch, seq_length), dtype=bool),
                action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        elif FLAGS.modality == "vision,text,proprio,latent_action,latent_state,action_cont":
            params = model.init(
                input_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                proprio=jnp.zeros((batch, seq_length, proprio_dims), dtype=jnp.float32),
                actions=jnp.zeros((batch, seq_length, action_dims), dtype=jnp.float32),
                time=jnp.zeros((batch,), dtype=jnp.float32),
                vision_masks=jnp.zeros((batch, seq_length), dtype=bool),
                latent_action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                latent_state_masks=jnp.zeros((batch, seq_length), dtype=bool),
                proprio_masks=jnp.zeros((batch, seq_length), dtype=bool),
                action_masks=jnp.zeros((batch, seq_length), dtype=bool),
                position_ids=jnp.zeros((batch, seq_length), dtype=jnp.int32),
                attention_mask=jnp.ones((batch, seq_length), dtype=jnp.int32),
                rngs=rng_generator(llama_config.rng_keys()),
            )
        else:
            raise ValueError(f"Unsupported modality: {FLAGS.modality}")
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp"), "sp"))

        def loss_and_accuracy(params):
            if FLAGS.modality == "vision,text,latent_action,latent_state":
                vision_logits, text_logits, latent_action_logits, latent_state_logits = model.apply(
                    params,
                    batch["input_tokens"],
                    batch["input_vision_masks"],
                    batch["input_latent_action_masks"],
                    batch["input_latent_state_masks"],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                latent_action_loss, latent_action_acc = cross_entropy_loss_and_accuracy(
                    latent_action_logits,
                    jnp.where(batch["target_latent_action_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_latent_action_masks"],
                )
                latent_state_loss, latent_state_acc = cross_entropy_loss_and_accuracy(
                    latent_state_logits,
                    jnp.where(batch["target_latent_state_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_latent_state_masks"],
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits,
                    jnp.where(batch["target_vision_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_vision_masks"],
                )
                target_text_masks = (
                    (1.0 - batch["target_vision_masks"])
                    * (1.0 - batch["target_latent_action_masks"])
                    * (1.0 - batch["target_latent_state_masks"])
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits,
                    jnp.where(target_text_masks, batch["target_tokens"], 0),
                    batch["loss_masks"] * target_text_masks,
                )
                loss = 0.48 * latent_action_loss + 0.48 * latent_state_loss + 0.02 * text_loss

                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                    latent_action_loss=latent_action_loss,
                    latent_action_acc=latent_action_acc,
                    latent_state_loss=latent_state_loss,
                    latent_state_acc=latent_state_acc,
                )
            elif FLAGS.modality == "vision,text,latent_action,latent_state,action_quant":
                vision_logits, text_logits, latent_action_logits, latent_state_logits, action_logits = model.apply(
                    params,
                    batch["input_tokens"],
                    batch["input_vision_masks"],
                    batch["input_latent_action_masks"],
                    batch["input_latent_state_masks"],
                    batch["input_action_masks"],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits
                latent_action_loss, latent_action_acc = cross_entropy_loss_and_accuracy(
                    latent_action_logits,
                    jnp.where(batch["target_latent_action_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_latent_action_masks"],
                )
                latent_state_loss, latent_state_acc = cross_entropy_loss_and_accuracy(
                    latent_state_logits,
                    jnp.where(batch["target_latent_state_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_latent_state_masks"],
                )
                action_loss, action_acc = cross_entropy_loss_and_accuracy(
                    action_logits,
                    jnp.where(batch["target_action_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_action_masks"],
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits,
                    jnp.where(batch["target_vision_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_vision_masks"],
                )
                target_text_masks = (
                    (1.0 - batch["target_vision_masks"])
                    * (1.0 - batch["target_latent_action_masks"])
                    * (1.0 - batch["target_latent_state_masks"])
                    * (1.0 - batch["target_action_masks"])
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits,
                    jnp.where(target_text_masks, batch["target_tokens"], 0),
                    batch["loss_masks"] * target_text_masks,
                )
                loss = latent_state_loss + action_loss

                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                    latent_action_loss=latent_action_loss,
                    latent_action_acc=latent_action_acc,
                    latent_state_loss=latent_state_loss,
                    latent_state_acc=latent_state_acc,
                    action_loss=action_loss,
                    action_acc=action_acc,
                )
            elif FLAGS.modality == "vision,text,latent_action,latent_state,action_cont":
                flow_sig_min = FLAGS.flow_matching_sampler.flow_sig_min
                # Create a noise tensor with the same shape as actions
                time = sample_fm_time(rng_generator())
                time = with_sharding_constraint(time, PS(("dp", "fsdp")))
                actions = batch["actions"]
                noise = jax.random.normal(rng_generator(), shape=actions.shape, dtype=actions.dtype)
                noisy_actions = (1 - (1 - flow_sig_min) * time[:, None, None]) * noise + time[:, None, None] * actions

                vision_logits, text_logits, latent_action_logits, latent_state_logits, action_logits = model.apply(
                    params,
                    batch["input_tokens"],
                    noisy_actions,
                    time,
                    batch["input_vision_masks"],
                    batch["input_latent_action_masks"],
                    batch["input_latent_state_masks"],
                    batch["action_masks"],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits

                pred_flow = action_logits
                gt_flow = actions - (1 - flow_sig_min) * noise
                action_loss = l2_loss(pred_flow, gt_flow, batch["action_masks"])

                latent_action_loss, latent_action_acc = cross_entropy_loss_and_accuracy(
                    latent_action_logits,
                    jnp.where(batch["target_latent_action_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_latent_action_masks"],
                )
                latent_state_loss, latent_state_acc = cross_entropy_loss_and_accuracy(
                    latent_state_logits,
                    jnp.where(batch["target_latent_state_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_latent_state_masks"],
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits,
                    jnp.where(batch["target_vision_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_vision_masks"],
                )
                target_text_masks = (
                    (1.0 - batch["target_vision_masks"])
                    * (1.0 - batch["target_latent_action_masks"])
                    * (1.0 - batch["target_latent_state_masks"])
                    * (1.0 - batch["action_masks"])
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits,
                    jnp.where(target_text_masks, batch["target_tokens"], 0),
                    batch["loss_masks"] * target_text_masks,
                )
                loss = action_loss + latent_state_loss

                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                    latent_action_loss=latent_action_loss,
                    latent_action_acc=latent_action_acc,
                    latent_state_loss=latent_state_loss,
                    latent_state_acc=latent_state_acc,
                    action_loss=action_loss,
                )
            elif FLAGS.modality == "vision,text,proprio,latent_action,latent_state,action_cont":
                flow_sig_min = FLAGS.flow_matching_sampler.flow_sig_min
                # Create a noise tensor with the same shape as actions
                time = sample_fm_time(rng_generator())
                time = with_sharding_constraint(time, PS(("dp", "fsdp")))
                actions = batch["actions"]
                noise = jax.random.normal(rng_generator(), shape=actions.shape, dtype=actions.dtype)
                noisy_actions = (1 - (1 - flow_sig_min) * time[:, None, None]) * noise + time[:, None, None] * actions

                vision_logits, text_logits, latent_action_logits, latent_state_logits, action_logits = model.apply(
                    params,
                    batch["input_tokens"],
                    batch["proprio"],
                    noisy_actions,
                    time,
                    batch["input_vision_masks"],
                    batch["input_latent_action_masks"],
                    batch["input_latent_state_masks"],
                    batch["proprio_masks"],
                    batch["action_masks"],
                    deterministic=False,
                    rngs=rng_generator(llama_config.rng_keys()),
                ).logits

                pred_flow = action_logits
                gt_flow = actions - (1 - flow_sig_min) * noise
                action_loss = l2_loss(pred_flow, gt_flow, batch["action_masks"])

                latent_action_loss, latent_action_acc = cross_entropy_loss_and_accuracy(
                    latent_action_logits,
                    jnp.where(batch["target_latent_action_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_latent_action_masks"],
                )
                latent_state_loss, latent_state_acc = cross_entropy_loss_and_accuracy(
                    latent_state_logits,
                    jnp.where(batch["target_latent_state_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_latent_state_masks"],
                )
                vision_loss, vision_acc = cross_entropy_loss_and_accuracy(
                    vision_logits,
                    jnp.where(batch["target_vision_masks"], batch["target_tokens"], 0),
                    batch["loss_masks"] * batch["target_vision_masks"],
                )
                target_text_masks = (
                    (1.0 - batch["target_vision_masks"])
                    * (1.0 - batch["target_latent_action_masks"])
                    * (1.0 - batch["target_latent_state_masks"])
                    * (1.0 - batch["action_masks"])
                )
                text_loss, text_acc = cross_entropy_loss_and_accuracy(
                    text_logits,
                    jnp.where(target_text_masks, batch["target_tokens"], 0),
                    batch["loss_masks"] * target_text_masks,
                )
                loss = action_loss + latent_state_loss

                metrics = dict(
                    vision_loss=vision_loss,
                    vision_acc=vision_acc,
                    text_loss=text_loss,
                    text_acc=text_acc,
                    latent_action_loss=latent_action_loss,
                    latent_action_acc=latent_action_acc,
                    latent_state_loss=latent_state_loss,
                    latent_state_acc=latent_state_acc,
                    action_loss=action_loss,
                )
            else:
                raise ValueError(f"Unsupported modality: {FLAGS.modality}")
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, loss_metrics), grads = grad_fn(train_state.params)

        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            learning_rate=optimizer_info["learning_rate_schedule"](train_state.step),
            param_norm=global_norm(train_state.params),
            gradient_norm=global_norm(grads),
            **loss_metrics,
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp"), "sp"))
        if FLAGS.modality == "vision,text,latent_action,latent_state":
            raise NotImplementedError
        elif FLAGS.modality == "vision,text,latent_action,latent_state,action_quant":
            raise NotImplementedError
        elif FLAGS.modality == "vision,text,latent_action,latent_state,action_cont":
            raise NotImplementedError
        return rng_generator(), metrics

    def unseen_eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp"), "sp"))
        if FLAGS.modality == "vision,text,latent_action,latent_state":
            raise NotImplementedError
        elif FLAGS.modality == "vision,text,latent_action,latent_state,action_quant":
            raise NotImplementedError
        elif FLAGS.modality == "vision,text,latent_action,latent_state,action_cont":
            raise NotImplementedError
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        config_cls.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(train_state_partition, train_state_shapes)
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer,
        logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(init_fn, in_shardings=PS(), out_shardings=train_state_partition)

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params,),
        out_shardings=train_state_partition,
        donate_argnums=(0,),
    )

    if FLAGS.use_data_sharded_loader:
        batch_spec = PS(("dp", "fsdp"), "sp")
    else:
        batch_spec = PS()

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    sharded_unseen_eval_step = pjit(
        unseen_eval_step,
        in_shardings=(train_state_partition, PS(), batch_spec),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None, max_buffer_size=0):
        if shard_fns is not None:
            shard_fns = flatten_dict(to_state_dict(shard_fns))
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)
        flattend_train_state = {}
        with open_file(path) as fin:
            # 83886080 bytes = 80 MB, which is 16 blocks on GCS
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=max_buffer_size)
            for key, value in unpacker:
                key = tuple(key)
                if remove_dict_prefix is not None:
                    if key[: len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix) :]
                    else:
                        continue
                tensor = from_bytes(None, value)
                if shard_fns is not None:
                    tensor = shard_fns[key](tensor)
                flattend_train_state[key] = tensor

        if target is not None:
            flattened_target = flatten_dict(to_state_dict(target), keep_empty_nodes=True)
            for key, value in flattened_target.items():
                if key not in flattend_train_state and value == empty_node:
                    flattend_train_state[key] = value
                elif key not in flattend_train_state:
                    if len(value.shape) < 2:
                        # bias terms, zero init
                        tensor = jax.nn.initializers.zeros(jax.random.PRNGKey(0), value.shape, dtype=value.dtype)
                        tensor = jax.random.normal(jax.random.PRNGKey(0), value.shape, dtype=value.dtype) * 0.01
                    else:
                        initializer = jax.nn.initializers.lecun_normal()  # Example initializer
                        tensor = initializer(jax.random.PRNGKey(0), value.shape, dtype=value.dtype)

                    flattend_train_state[key] = tensor

        train_state = unflatten_dict(flattend_train_state)
        if target is None:
            return train_state

        return from_state_dict(target, train_state)

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    with mesh:
        train_state, restored_params = None, None
        using_autoresume = False
        train_state_resume_path = f"{output_dir}/streaming_train_state"

        if FLAGS.autoresume and tux.check_exists(train_state_resume_path):
            logging.info(f"Found existing output. Resuming model from latest checkpoint: {train_state_resume_path}")
            using_autoresume = True
            resume_path = f"trainstate::{train_state_resume_path}"
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                resume_path, train_state_shapes, shard_fns, max_buffer_size=32 * 2**30
            )
        elif FLAGS.load_checkpoint != "":
            logging.info(f"Loading model from checkpoint: {FLAGS.load_checkpoint}")
            params_target = train_state_shapes.params["params"]
            params_shard_fns = shard_fns.params["params"]
            load_type, load_path = FLAGS.load_checkpoint.split("::", 1)
            train_state = None
            restored_params = None
            if load_type == "trainstate":
                train_state = load_checkpoint(
                    path=load_path,
                    target=train_state_shapes,
                    shard_fns=shard_fns,
                    max_buffer_size=32 * 2**30,
                )
            elif load_type == "trainstate_params":
                # Load the params part of the train state in the streaming format
                restored_params = load_checkpoint(
                    path=load_path,
                    target=params_target,
                    shard_fns=params_shard_fns,
                    remove_dict_prefix=("params", "params"),
                    max_buffer_size=32 * 2**30,
                )
                restored_params = flax.core.frozen_dict.freeze({"params": restored_params})
            elif load_type == "params":
                # Load the params in the streaming format
                restored_params = load_checkpoint(
                    path=load_path, target=params_target, shard_fns=params_shard_fns, max_buffer_size=32 * 2**30
                )
                restored_params = flax.core.frozen_dict.freeze({"params": restored_params})

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            # Run the checks on the restored_params
            logging.info("Checking loaded parameters for NaN or Inf...")
            invalid_params = check_invalid_params(restored_params["params"])
            log_invalid_params(restored_params["params"])

            # Print the overall result
            if jax.tree_util.tree_reduce(lambda x, y: x or y, invalid_params):
                logging.info("Warning: Invalid parameter values detected in checkpoint!")
            else:
                logging.info("All parameters are valid (no NaN or Inf detected).")
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        check_and_log_frozen_params(train_state.params, frozen_param_mask)

        start_step = int(jax.device_get(train_state.step))

        # Ensure all compilation and weight sharding is finished before training
        train_state = jax.block_until_ready(train_state)
        import gc

        gc.collect()

        if FLAGS.save_model_freq > 0 and not using_autoresume:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)
        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            train_state, sharded_rng, metrics = sharded_train_step(train_state, sharded_rng, batch)
            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0 and step % FLAGS.eval_log_freq == 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(train_state, sharded_rng, eval_batch)
                        eval_metrics = jax.device_get(eval_metrics)
                        eval_batch, _ = next(unseen_eval_iterator)
                        sharded_rng, eval_metrics2 = sharded_unseen_eval_step(train_state, sharded_rng, eval_batch)
                        eval_metrics2 = jax.device_get(eval_metrics2)
                        # concat two dict
                        eval_metrics.update(eval_metrics2)
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    run(main)
