"""Token-weighted updates and exact local checkpoint recovery.

AI assistance: implemented by Codex. Software checks are not research results.
"""

from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
import random

import numpy as np
import torch

from .model import ModelConfig, TinyGPT, token_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_for(model, *, learning_rate=3e-4, fused=False):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused,
    )


def phase_learning_rate(step, steps, *, peak=3e-4, floor=3e-5):
    if not 0 <= step < steps:
        raise ValueError("Step outside the phase")
    warmup = max(1, math.ceil(0.02 * steps))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup - 1)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def update(model, optimizer, blocks, *, microbatch=8, scaler=None, max_grad_norm=1.0):
    """Train on [batch, context+1] integer blocks, preserving token mean on accumulation."""
    if blocks.ndim != 2 or blocks.shape[1] != model.config.context + 1:
        raise ValueError("Blocks need context+1 tokens for explicit target shifting")
    if blocks.shape[0] <= 0 or microbatch <= 0:
        raise ValueError("Empty batch or invalid microbatch")
    device = next(model.parameters()).device
    if scaler is not None and device.type != "cuda":
        raise ValueError("The FP16 scaler is only used on CUDA in this experiment")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    token_count = blocks.shape[0] * model.config.context
    loss_total = 0.0
    try:
        for start in range(0, blocks.shape[0], microbatch):
            batch = blocks[start:start + microbatch].to(device=device, dtype=torch.long)
            precision = torch.autocast("cuda", dtype=torch.float16) if scaler is not None else nullcontext()
            with precision:
                logits = model(batch[:, :-1])
                loss_sum, _ = token_loss(logits, batch[:, 1:])
                loss = loss_sum / token_count
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss_total += float(loss_sum.detach())
            (scaler.scale(loss) if scaler is not None else loss).backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        applied = [False]
        hook = optimizer.register_step_post_hook(lambda *_: applied.__setitem__(0, True))
        try:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
        finally:
            hook.remove()
        return {"loss": loss_total / token_count, "tokens": token_count,
                "grad_norm": float(norm), "optimizer_step_applied": applied[0]}
    finally:
        optimizer.zero_grad(set_to_none=True)


def save_checkpoint(path, model, optimizer, *, scaler=None, progress=None, identity=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1, "config": asdict(model.config),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": progress or {}, "identity": identity or {},
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, device="cpu", expected_identity=None, fused=False):
    """Load only a checkpoint created by this trusted local runner (pickle payload)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    if expected_identity is not None and state["identity"] != expected_identity:
        raise ValueError("Checkpoint code/data/config identity mismatch")
    model = TinyGPT(ModelConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    optimizer = optimizer_for(model, fused=fused)
    optimizer.load_state_dict(state["optimizer"])
    scaler = None
    if state["scaler"] is not None:
        if torch.device(device).type != "cuda":
            raise ValueError("Cannot silently discard an FP16 scaler on CPU resume")
        scaler = torch.amp.GradScaler("cuda")
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return model, optimizer, scaler, state["progress"]
