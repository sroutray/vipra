import logging

import torch
import torch.distributed as dist
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist
import torch.nn as nn
from numpy import indices

logger = logging.getLogger(__name__)


class NSVQ(nn.Module):
    """
    Noise-Substitution Vector Quantizer (NSVQ), DDP-safe:
    - Training forward: hard quantize + noise-substitution residual
    - Global perplexity via all_reduce
    - Local usage counters, synchronized for replacement
    - Separate no_grad replace_unused_codebooks with global thresholding
    - Inference: Hard quantization

    Args:
        num_embeddings (int): number of codebook entries.
        embedding_dim (int): dimensionality of each entry.
        discarding_threshold (float): fraction threshold to replace unused codes.
        initialization (str): 'normal' or 'uniform'.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        discarding_threshold: float = 0.01,
        initialization: str = "normal",
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-8

        # initialize codebook (on default device)
        if initialization == "normal":
            init_cb = torch.randn(num_embeddings, embedding_dim)
        elif initialization == "uniform":
            low, high = -1.0 / num_embeddings, 1.0 / num_embeddings
            init_cb = uniform_dist.Uniform(low, high).sample((num_embeddings, embedding_dim))
        else:
            raise ValueError("initialization must be 'normal' or 'uniform'")
        self.codebooks = nn.Parameter(init_cb)

        # usage counter buffer
        self.register_buffer("codebooks_used", torch.zeros(num_embeddings, dtype=torch.long))

    def forward(self, input_data: torch.Tensor):
        """
        Args:
            input_data: (N, D) tensor
        Returns:
            quantized_input: (N, D)
            perplexity: scalar (float)
            num_unique_indices: scalar (int)
        """
        # input_data: (N, D)
        N, D = input_data.shape
        device = input_data.device

        # compute distances: (N, K)
        cb = self.codebooks
        d2 = input_data.pow(2).sum(dim=1, keepdim=True) - 2 * input_data @ cb.t() + cb.pow(2).sum(dim=1).unsqueeze(0)
        min_idx = torch.argmin(d2, dim=1)

        # hard quantized vectors
        hard_q = cb[min_idx]
        # noise-substitution residual
        rand = normal_dist.Normal(0, 1).sample(input_data.shape).to(device)
        r_norm = torch.linalg.norm(input_data - hard_q, dim=1, keepdim=True)
        n_norm = torch.linalg.norm(rand, dim=1, keepdim=True)
        vq_err = (r_norm / (n_norm + self.eps)) * rand
        quantized_input = input_data + vq_err

        with torch.no_grad():
            # Local counter update
            local_counts = torch.bincount(min_idx, minlength=self.num_embeddings)
            self.codebooks_used += local_counts.to(self.codebooks_used.dtype)

            # Global usage counts
            usage_counts = local_counts.to(torch.float32)
            if dist.is_initialized():
                dist.all_reduce(usage_counts, op=dist.ReduceOp.SUM)

            # Perplexity computation
            total = usage_counts.sum().clamp_min(1.0)
            avg_probs = usage_counts / total
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps))).item()

            logger.debug(f"Perplexity: {perplexity}")
            logger.debug(f"Codebook usage: {usage_counts.cpu().tolist()}")
            logger.debug(f"Average usage: {avg_probs.cpu().tolist()}")
            # print(f"Codes: {min_idx.clone().cpu().tolist()}")

            # Count number of unique indices (used for monitoring entropy)
            num_unique_indices = int((usage_counts > 0).sum().item())

        return quantized_input, perplexity, num_unique_indices

    @torch.no_grad()
    def replace_unused_codebooks(self):
        """
        Replace codes whose global usage fraction < threshold.
        Should be called periodically every `num_batches`.
        """
        # gather global usage counts
        usage = self.codebooks_used.to(self.codebooks.device).float()
        if dist.is_initialized():
            dist.all_reduce(usage, op=dist.ReduceOp.SUM)
        frac = usage / (usage.sum() + self.eps)
        unused = torch.nonzero(frac < self.discarding_threshold, as_tuple=False).view(-1)
        used = torch.nonzero(frac >= self.discarding_threshold, as_tuple=False).view(-1)

        cb_data = self.codebooks.data
        new_cb_data_to_sync = cb_data.clone()  # Initialize for all ranks

        is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0

        if is_rank_0:
            # Perform replacement logic only on rank 0
            if used.numel() == 0:
                noise = self.eps * torch.randn_like(cb_data)
                new_cb_data_to_sync = cb_data + noise  # Modify the tensor that will be broadcast
            else:
                used_vecs = cb_data[used]
                k_un = unused.numel()
                if k_un > 0:  # Only proceed if there are unused codes to replace
                    if used_vecs.size(0) < k_un:
                        reps = (k_un + used_vecs.size(0) - 1) // used_vecs.size(0)  # Ceiling division
                        used_vecs = used_vecs.repeat(reps, 1)

                    perm = torch.randperm(used_vecs.size(0), device=cb_data.device)
                    selected = used_vecs[perm[:k_un]]

                    # Ensure 'selected' has the right shape for assignment
                    if selected.shape[0] == k_un:
                        new_cb_data_to_sync[unused] = selected + self.eps * torch.randn_like(selected)
                    # else: handle cases where not enough vectors could be selected, though logic should prevent this.

        if dist.is_initialized():
            dist.broadcast(new_cb_data_to_sync, src=0)

        self.codebooks.data.copy_(new_cb_data_to_sync)
        self.codebooks_used.zero_()
        logger.info(f"#### Replaced {unused.numel()} unused codebooks with new vectors. ####")
        logger.info(f"#### Codebook usage: {frac.cpu().tolist()} ####")
        logger.info(f"#### Discarded codebooks: {unused.cpu().tolist()} ####")

    @torch.no_grad()
    def inference(self, input_data: torch.Tensor):
        """
        Inference: Hard quantization.
        """
        N, D = input_data.shape
        device = input_data.device
        cb = self.codebooks.to(device)
        d2 = input_data.pow(2).sum(dim=1, keepdim=True) - 2 * input_data @ cb.t() + cb.pow(2).sum(dim=1).unsqueeze(0)
        min_idx = torch.argmin(d2, dim=1)
        return cb[min_idx], min_idx

    @torch.no_grad()
    def convert_idx_to_embeddings(self, idxs: torch.Tensor):
        """
        Convert indices to embeddings.
        """
        out_shape = idxs.shape + (self.embedding_dim,)
        idxs = idxs.reshape(-1)
        embeddings = self.codebooks[idxs]
        return embeddings.reshape(out_shape)
