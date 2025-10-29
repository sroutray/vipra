from pathlib import Path

DATA_CFG = dict(
    image_size=224,
    max_frames=8,
)


DATASET_CFG = dict(
    ssv2=dict(
        root_dir=Path("/fsx2/shared/sroutray/OpenX/ssv2"),
        split="trainval",
        stepsize=2,
    ),
    fractal=dict(
        root_dir=Path("/fsx2/shared/sroutray/OpenX/fractal"),
        split="trainval",
        num_trajs=dict(trainval=70000, val=7000),
        stepsize=1,
    ),
    bridge=dict(
        root_dir=Path("/fsx2/shared/sroutray/OpenX/bridge"),
        split="trainval",
        num_trajs=dict(trainval=25460, val=2546),
        stepsize=1,
    ),
    kuka=dict(
        root_dir=Path("/fsx2/shared/sroutray/OpenX/kuka"),
        split="trainval",
        num_trajs=dict(trainval=70000, val=7000),
        stepsize=1,
    ),
    libero=dict(
        root_dir=Path("/fsx2/shared/sroutray/LIBERO"),
        split="trainval",
        num_trajs=dict(trainval=1.0, val=0.1),
        stepsize=1,
    ),
)


VAL_DATASET_CFG = {dataset: {**config, "split": "val"} for dataset, config in DATASET_CFG.items()}


MODEL_CFG = dict(
    dim=768,
    image_size=DATA_CFG["image_size"],
    patch_size=14,
    channels=3,
    enc_depth=6,
    dec_depth=8,
    dim_head=64,
    heads=16,
    attn_dropout=0.0,
    ff_dropout=0.1,
    quant_dim=32,
    codebook_size=8,
    code_seq_len=16,
    encode_deltas=False,
    discarding_threshold=0.015,
    max_codebook_update_step=120_000,
    spatial_enc_type="dino_reg_base",
    use_lpips_loss=True,
    lpips_loss_weight=0.5,
    use_flow_loss=True,
    flow_loss_weight=0.1,
    flow_loss_kickin_step=60_000,
    flow_loss_warmup_steps=10_000,
)


TRAIN_CFG = dict(
    num_train_steps=300_000,
    batch_size=18,
    val_batch_size=8,
    lr=1e-4,
    pretrained_init_lr_mult_factor=0.1,
    weight_decay=0.0,
    grad_accum_every=1,
    max_grad_norm=6.0,
    save_model_every=1000,
    save_milestone_every=10000,
    val_every_n_steps=2000,
    num_val_batches_to_log=4,
    num_val_samples_to_save=6,
    num_workers=16,
    prefetch_factor=24,
    pin_memory=True,
)
