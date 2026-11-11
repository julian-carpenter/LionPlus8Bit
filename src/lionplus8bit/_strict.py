"""Strict recurrence with bounded momentum scratch and independent block scales.

Flattening a noncontiguous gradient can still require a full contiguous copy.
"""

from contextlib import nullcontext
import bitsandbytes as bnb
import torch


@torch.no_grad()
def strict_update(target, grad, state, config, current, default):
    codes = state["state1"]
    quantized = codes.dtype == torch.uint8
    # Align chunks with quantization blocks; momentum scratch is bounded by this size.
    chunk_size = 65536
    flat_target, flat_grad = target.view(-1), grad.reshape(-1)
    flat_state = codes.view(-1)
    beta1, beta2 = config["betas"]
    for start in range(0, target.numel(), chunk_size):
        end = min(start + chunk_size, target.numel())
        g = flat_grad[start:end].float()
        if quantized:
            scales = state["absmax1"][start // 256 : (end + 255) // 256]
            m = state["qmap1"][flat_state[start:end].long()]
            m = m * scales.repeat_interleave(256)[: end - start]
        else:
            m = flat_state[start:end]
        direction = (m * beta1 + g * (1 - beta1)).sign()
        updated = m * beta2 + g * (1 - beta2)
        flat_target[start:end].mul_(1 - config["lr"] * config["weight_decay"]).add_(
            direction, alpha=-config["lr"]
        )
        if quantized:
            pad = (-updated.numel()) % 256
            blocks = torch.nn.functional.pad(updated, (0, pad)).view(-1, 256)
            peaks = blocks.abs().amax(1)
            divisors = torch.where(peaks > 0, peaks, torch.ones_like(peaks))
            normalized = (blocks / divisors[:, None]).flatten().contiguous()
            if current is not None and current != default:
                default.wait_stream(current)
            context = torch.cuda.stream(default) if default is not None else nullcontext()
            with context:
                result, qs = bnb.functional.quantize_blockwise(
                    normalized, code=state["qmap1"], blocksize=256
                )
            if current is not None:
                if current != default:
                    current.wait_stream(default)
                normalized.record_stream(default)
                state["qmap1"].record_stream(default)
                result.record_stream(current)
                qs.absmax.record_stream(current)
            flat_state[start:end].copy_(result[: end - start])
            scales.copy_(qs.absmax * divisors)
        else:
            flat_state[start:end].copy_(updated)
    state["step"] += 1
