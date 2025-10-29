import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from ml_collections import ConfigDict


class FlowMatchingSamplerConfig:
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict(
            {
                "flow_sampling": "beta",  # Options: "uniform", "beta"
                "flow_alpha": 1.5,  # Alpha parameter for Beta distribution
                "flow_beta": 1.0,  # Beta parameter for Beta distribution
                "flow_sig_min": 0.001,  # Minimum sigma value for time scaling
            }
        )
        if updates:
            config.update(updates)
        return config


def create_fm_time_sampler(config=None, batch_size=1):
    """
    Creates a flow matching time sampler function based on configuration.

    Args:
        config: Dictionary with configuration parameters

    Returns:
        A function fm_time_sampler(batch, rng) for time sampling
    """

    # Set up default configuration
    if config is None:
        config = {}

    flow_sampling = config.get("flow_sampling", "beta")
    flow_alpha = config.get("flow_alpha", 1.5)
    flow_beta = config.get("flow_beta", 1.0)
    flow_sig_min = config.get("flow_sig_min", 0.001)

    # Validate configuration
    if flow_sampling not in ["uniform", "beta"]:
        raise ValueError(f"Invalid flow sampling mode: {flow_sampling}")

    # Pre-compute constants (as static values outside the returned function)
    flow_t_max = 1.0 - flow_sig_min
    is_uniform = flow_sampling == "uniform"

    # Create beta distribution if needed
    beta_dist = None
    if flow_sampling == "beta":
        beta_dist = tfd.Beta(flow_alpha, flow_beta)

    # Define helper functions that will be used by our sampler
    def _uniform_sampler(rng):
        rng, sample_rng = jax.random.split(rng)
        eps = 1e-5
        rand_offset = jax.random.uniform(sample_rng, (1,))
        t = (rand_offset + jnp.arange(batch_size) / batch_size) % (1.0 - eps)
        return t

    def _beta_sampler(rng):
        rng, sample_rng = jax.random.split(rng)
        samples = beta_dist.sample(seed=sample_rng, sample_shape=(batch_size,))
        t = flow_t_max * (1.0 - samples)
        return t

    if is_uniform:
        return _uniform_sampler

    return _beta_sampler
