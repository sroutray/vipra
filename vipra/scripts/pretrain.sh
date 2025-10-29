export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

export absolute_path=$PROJECT_DIR
export llama_tokenizer_path="$absolute_path/lwm/tokenizer.model"
# export vqgan_path="$absolute_path/lwm/vqgan"  # necessary if vision_path does not exist
export vqgan_path=""
export vision_path="$absolute_path/vision_cache/cotrain_vqgan_vision_cache.jsonl"
export output_dir="$absolute_path/outputs"

export project_id='vipra-policy'
export experiment_note='vipra_pretrain_cotrain_ls14_la14_run1'

export dataset_path="$absolute_path/cotrain_data/cotrain_dynamics14_processed.jsonl"
export image_absolute_path=""  # necessary if vision_path does not exist
export experiment_id='vipra_pretrain_cotrain_ls14_la14_run1'

python3 -u -m policy.train \
    --modality='vision,text,latent_action,latent_state' \
    --mesh_dim='!-1,8,1,1' \
    --dtype='bf16' \
    --total_steps=50001 \
    --log_freq=1 \
    --eval_steps=0 \
    --save_model_freq=500 \
    --eval_log_freq=100 \
    --save_milestone_freq=50000 \
    --load_llama_config='7b' \
    --load_checkpoint="params::$absolute_path/lwm/params" \
    --update_llama_config="dict(latent_action_vocab_size=9,theta=50000000,max_sequence_length=1536,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=1536,scan_key_chunk_size=1536,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" \
    --tokenizer.vocab_file="$llama_tokenizer_path" \
    --optimizer.type='adamw' \
    --llama.latent_action_vocab_size=9 \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0 \
    --optimizer.adamw_optimizer.lr=4e-5 \
    --optimizer.adamw_optimizer.end_lr=4e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 \
    --optimizer.adamw_optimizer.lr_decay_steps=100 \
    --use_data_sharded_loader=True \
    --train_dataset.type='json_vision_dynamics' \
    --train_dataset.dynamics_vision_text_processor.image_absolute_path="$image_absolute_path" \
    --train_dataset.dynamics_vision_text_processor.eola_token=8 \
    --train_dataset.dynamics_vision_text_processor.fields_from_example='fields_ls_la' \
    --train_dataset.dynamics_vision_text_processor.n_tokens_per_latent_action=16 \
    --train_dataset.dynamics_vision_text_processor.n_tokens_per_latent_state=256 \
    --train_dataset.dynamics_vision_text_processor.max_n_frames=-1 \
    --train_dataset.dynamics_vision_text_processor.img_aug=False \
    --train_dataset.dynamics_vision_text_processor.vision_path="$vision_path" \
    --train_dataset.dynamics_vision_text_processor.vqgan_checkpoint_path="$vqgan_path" \
    --train_dataset.json_dynamics_dataset.mode="pad" \
    --train_dataset.json_dynamics_dataset.path="$dataset_path" \
    --train_dataset.json_dynamics_dataset.seq_length=1536 \
    --train_dataset.json_dynamics_dataset.batch_size=320 \
    --train_dataset.json_dynamics_dataset.tokenizer_processes=1 \
    --train_dataset.json_dynamics_dataset.tokenizer_parallel_chunk_size=8 \
    --train_dataset.json_dynamics_dataset.tokenizer_parallel_batch_size=1024 \
    --train_dataset.json_dynamics_dataset.use_data_sharded_loader=True \
    --checkpointer.save_optimizer_state=True \
    --autoresume=True \
    --logger.append_uuid=False \
    --logger.online=True \
    --logger.project_id="$project_id" \
    --logger.experiment_id="$experiment_id" \
    --logger.experiment_note="$experiment_note" \
    --logger.output_dir="$output_dir" \
    --logger.wandb_dir="$absolute_path/experiment_output/$project_id"
