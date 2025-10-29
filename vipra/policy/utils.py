import absl.logging as logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import empty_node, flatten_dict, unflatten_dict


# Check for NaN or Inf in the loaded checkpoint parameters
def check_invalid_params(tree):
    """Recursively checks for NaN or Inf values in a pytree of params."""

    def check(x):
        if jnp.any(jnp.isnan(x)):
            print("NaN detected!")
            return True
        if jnp.any(jnp.isinf(x)):
            print("Inf detected!")
            return True
        return False

    invalid = jax.tree_util.tree_map(check, tree)
    return invalid


# Log invalid params
def log_invalid_params(tree, prefix=""):
    """Logs any parameter arrays that contain NaN or Inf values."""
    for key, value in tree.items() if isinstance(tree, dict) else enumerate(tree):
        if isinstance(value, (dict, list, tuple)):
            log_invalid_params(value, prefix=f"{prefix}/{key}")
        elif isinstance(value, (jnp.ndarray, np.ndarray, float, int)):  # Only check numerical types
            if jnp.any(jnp.isnan(value)):
                logging.info(f"NaN in parameter: {prefix}/{key}")
            if jnp.any(jnp.isinf(value)):
                logging.info(f"Inf in parameter: {prefix}/{key}")
        else:
            logging.info(f"Skipping non-numerical parameter: {prefix}/{key}")


def check_and_log_frozen_params(params, frozen_param_mask):
    if frozen_param_mask is None:
        logging.info("[Freeze Mask Verification] Skipped: No frozen_param_mask provided.")
        return

    param_mask = frozen_param_mask(params)
    flat_mask = flatten_dict(param_mask)

    frozen_count = 0
    trainable_count = 0
    logging.info("[Freeze Mask Verification]")
    for key, is_frozen in flat_mask.items():
        full_name = "/".join(key)
        if is_frozen:
            logging.info(f"Frozen:     {full_name}")
            frozen_count += 1
        else:
            logging.info(f"Trainable:  {full_name}")
            trainable_count += 1

    logging.info(f"[Summary] Trainable params: {trainable_count}, Frozen params: {frozen_count}")


def l1_loss(predicted_logits, true_tokens, valid=None):
    # Get the predicted tokens by taking the argmax over logits
    predicted_tokens = jnp.argmax(predicted_logits, axis=-1)

    # Calculate the L1 loss as the sum of absolute differences between predicted and true tokens
    loss = jnp.abs(predicted_tokens - true_tokens)

    # Mask the loss with the valid mask if provided
    if valid is not None:
        loss = loss * valid
    loss = jnp.mean(jnp.sum(loss, axis=-1))

    return loss


def l2_loss(val, target, valid=None):
    """
    val: predicted values [B, S, D]
    target: target values [B, S, D]
    valid: mask [B, S, 1]
    """
    if valid is None:
        valid = jnp.ones((*target.shape[:2], 1))
    valid = valid.astype(jnp.float32)
    loss = jnp.sum(jnp.where(valid > 0.0, jnp.mean(jnp.square(val - target), -1), 0.0)) / jnp.maximum(
        jnp.sum(valid), 1e-5
    )
    return loss
