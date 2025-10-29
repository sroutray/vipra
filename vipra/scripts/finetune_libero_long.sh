export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

export absolute_path=$PROJECT_DIR
export llama_tokenizer_path="$absolute_path/lwm/tokenizer.model"
export vqgan_path="$absolute_path/lwm/vqgan"
# export vqgan_path=""
# export vision_path="/fsx2/shared/sroutray/libero_all_dynamics14_vqgan_vision.jsonl"
export output_dir="$absolute_path/outputs"

export project_id='vipra-policy'
export experiment_note='vipra_finetune_libero_long_ca14_lr4e-4_aug_run3'

export dataset_path="$absolute_path/libero_10_modified/libero_10_dynamics14_norm_v2.jsonl"
export image_absolute_path="$absolute_path"  # necessary if vision_path does not exist
export experiment_id='vipra_finetune_libero_long_ca14_lr4e-4_aug_run3'

python3 -u -m policy.train \
    --modality='vision,text,latent_action,latent_state,action_cont' \
    --mesh_dim='!-1,8,1,1' \
    --dtype='bf16' \
    --total_steps=30001 \
    --log_freq=1 \
    --eval_steps=0 \
    --save_model_freq=500 \
    --eval_log_freq=100 \
    --save_milestone_freq=10000 \
    --load_llama_config='7b' \
    --load_checkpoint="params::$absolute_path/vipra_checkpoints/vipra_params" \
    --update_llama_config="dict(action_dims=7,latent_action_vocab_size=9,theta=50000000,max_sequence_length=768,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=768,scan_key_chunk_size=768,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" \
    --tokenizer.vocab_file="$llama_tokenizer_path" \
    --optimizer.type='adamw' \
    --llama.latent_action_vocab_size=9 \
    --llama.action_dims=7 \
    --llama.action_chunk_size=14 \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0 \
    --optimizer.adamw_optimizer.lr=4e-4 \
    --optimizer.adamw_optimizer.end_lr=4e-4 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 \
    --optimizer.adamw_optimizer.lr_decay_steps=100 \
    --use_data_sharded_loader=True \
    --train_dataset.type='json_vision_dynamics_action_cont' \
    --train_dataset.dynamics_vision_action_cont_processor.image_absolute_path="$image_absolute_path" \
    --train_dataset.dynamics_vision_action_cont_processor.eola_token=8 \
    --train_dataset.dynamics_vision_action_cont_processor.fields_from_example='fields_ca' \
    --train_dataset.dynamics_vision_action_cont_processor.n_tokens_per_latent_action=16 \
    --train_dataset.dynamics_vision_action_cont_processor.n_tokens_per_latent_state=256 \
    --train_dataset.dynamics_vision_action_cont_processor.action_dims=7 \
    --train_dataset.dynamics_vision_action_cont_processor.max_n_frames=-1 \
    --train_dataset.dynamics_vision_action_cont_processor.img_aug=True \
    --train_dataset.dynamics_vision_action_cont_processor.vision_path="$vision_path" \
    --train_dataset.dynamics_vision_action_cont_processor.vqgan_checkpoint_path="$vqgan_path" \
    --train_dataset.json_dynamics_action_cont_dataset.action_dims=7 \
    --train_dataset.json_dynamics_action_cont_dataset.action_chunk_size=14 \
    --train_dataset.json_dynamics_action_cont_dataset.mode="pad" \
    --train_dataset.json_dynamics_action_cont_dataset.path="$dataset_path" \
    --train_dataset.json_dynamics_action_cont_dataset.seq_length=768 \
    --train_dataset.json_dynamics_action_cont_dataset.batch_size=512 \
    --train_dataset.json_dynamics_action_cont_dataset.tokenizer_processes=1 \
    --train_dataset.json_dynamics_action_cont_dataset.tokenizer_parallel_chunk_size=8 \
    --train_dataset.json_dynamics_action_cont_dataset.tokenizer_parallel_batch_size=1024 \
    --train_dataset.json_dynamics_action_cont_dataset.use_data_sharded_loader=True \
    --checkpointer.save_optimizer_state=True \
    --autoresume=True \
    --logger.append_uuid=False \
    --logger.online=True \
    --logger.project_id="$project_id" \
    --logger.experiment_id="$experiment_id" \
    --logger.experiment_note="$experiment_note" \
    --logger.output_dir="$output_dir" \
    --logger.wandb_dir="$absolute_path/experiment_output/$project_id"
