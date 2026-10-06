"""Small causal transformers for the approved specialization experiment.

Architecture follows standard decoder-only transformer components. Experiment
motivation: Branch-Train-Merge (2208.03306) and c-BTM (2303.14177).
This is an original small implementation, not the authors' released code.
AI assistance: implemented by Codex; user authorship/mastery is not implied.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    width: int
    heads: int
    mlp_width: int
    vocabulary: int = 8192
    context: int = 256

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("All model dimensions must be positive integers")
        if self.width % self.heads:
            raise ValueError("Width must divide evenly into attention heads")

    @property
    def parameters(self):
        d, m, L = self.width, self.mlp_width, self.layers
        return (self.vocabulary + self.context) * d + L * (
            4 * d * d + 9 * d + (2 * d + 1) * m
        ) + 2 * d

    @property
    def train_matmul_flops_per_token(self):
        d, m, L = self.width, self.mlp_width, self.layers
        # Forward + backward estimate. Excludes normalization, optimizer and
        # other elementwise/preparation/selection operations.
        return 6 * (self.vocabulary * d + L * (4 * d * d + 2 * d * m)) + (
            12 * L * self.context * d
        )


BASE_MODELS = {
    "small": ModelConfig(6, 320, 5, 1280),
    "medium": ModelConfig(8, 384, 6, 1536),
}
CALIBRATORS = {
    "small_2": ModelConfig(7, 448, 7, 1728),
    "small_4": ModelConfig(12, 512, 8, 1920),
    "small_8": ModelConfig(19, 576, 9, 2304),
    "medium_2": ModelConfig(12, 448, 7, 1984),
    "medium_4": ModelConfig(16, 576, 9, 2368),
    "medium_8": ModelConfig(22, 704, 11, 2880),
}


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(tensor):
            return tensor.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
        result = F.scaled_dot_product_attention(
            heads(q), heads(k), heads(v), dropout_p=0.0, is_causal=True
        )
        return self.output(result.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, config.mlp_width),
            nn.GELU(),
            nn.Linear(config.mlp_width, config.width),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocabulary, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.apply(self._initialize)
        # Scale residual branch initializations equally in every experimental arm.
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, std=0.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.mlp[2].weight, std=0.02 / math.sqrt(2 * config.layers))
        count = sum(p.numel() for p in self.parameters())
        if count != config.parameters:
            raise ValueError(f"Parameter count mismatch: {count} != {config.parameters}")

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.config.context:
            raise ValueError("Expected [batch, time] input within the context limit")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.tokens(token_ids) + self.positions(positions)
        for block in self.blocks:
            x = block(x)
        # The vocabulary embedding is also the output matrix: no duplicate head.
        return F.linear(self.norm(x), self.tokens.weight)


def token_loss(logits, targets, *, score_from=0):
    """Already shifted targets: logits at position t predict token t+1."""
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError("Logits and shifted targets must share batch/time dimensions")
    if not 0 <= score_from < targets.shape[1]:
        raise ValueError("Invalid scoring boundary")
    scores = logits[:, score_from:].float()
    labels = targets[:, score_from:]
    total = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), labels.reshape(-1), reduction="sum")
    return total, labels.numel()


def mixture_log_probabilities(logits, weights):
    """Mix probabilities in log space; do not average logits or checkpoint weights."""
    if logits.ndim != 4:
        raise ValueError("Expected [expert, batch, time, vocabulary]")
    if weights.ndim == 1:
        weights = weights[:, None].expand(-1, logits.shape[1])
    if weights.shape != logits.shape[:2] or not torch.isfinite(weights).all():
        raise ValueError("Weights must be finite [expert, batch] values")
    if (weights < 0).any() or (weights.sum(dim=0) <= 0).any():
        raise ValueError("Weights must be nonnegative with positive total mass")
    weights = weights.float() / weights.float().sum(dim=0, keepdim=True)
    return torch.logsumexp(
        logits.float().log_softmax(dim=-1) + weights.log()[:, :, None, None], dim=0
    )
