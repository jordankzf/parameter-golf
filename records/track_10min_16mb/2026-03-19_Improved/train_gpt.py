"""
Improved baseline for parameter-golf competition.
Key changes from naive baseline:
- Layer recurrence: 5 unique layers x 2 loops = 10 effective layers
- Per-loop LoRA adapters on Q/K/V/O for loop specialization
- SwiGLU MLP (replaces ReLU^2)
- Multi-token prediction with scheduled fadeout
- Sliding window evaluation (stride-64)
- Shifted softcap from modded-nanogpt
- seq_len=4096 (from PR#65)
- Tuned Muon: momentum=0.99, lower LRs (from PR#65)
- Smaller batch 393K for more steps/min (from PR#65)
- fp16 tied embedding export (from PR#66)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 393_216))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape - layer recurrence with SwiGLU
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", 5))
    num_recurrence_loops = int(os.environ.get("NUM_RECURRENCE_LOOPS", 2))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 544))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    # SwiGLU uses 8/3 * dim for hidden, we round to multiple of 64
    swiglu_mult = float(os.environ.get("SWIGLU_MULT", 2.667))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    # Shifted softcap from modded-nanogpt: A * sigmoid((logits + B) / C)
    softcap_A = float(os.environ.get("SOFTCAP_A", 23.0))
    softcap_B = float(os.environ.get("SOFTCAP_B", 5.0))
    softcap_C = float(os.environ.get("SOFTCAP_C", 7.5))

    # LoRA rank for per-loop adapters
    lora_rank = int(os.environ.get("LORA_RANK", 16))

    # Multi-token prediction
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 3))

    # Sliding window eval
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # fp16 tied embedding export (keeps embedding in fp16 during quantization)
    embed_fp16_export = bool(int(os.environ.get("EMBED_FP16_EXPORT", "1")))

    # Optimizer hyperparameters (tuned from PR#65)
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    lora_lr = float(os.environ.get("LORA_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    # For sliding window eval, we keep all tokens (don't truncate to seq_len multiples)
    return tokens


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Sliding window evaluation with configurable stride for better BPB."""
    seq_len = args.train_seq_len
    stride = args.eval_stride
    total_tokens = val_tokens.numel()

    # Generate all window start positions
    starts = list(range(0, total_tokens - seq_len, stride))
    if not starts:
        starts = [0]

    # Split across ranks
    rank_starts = starts[rank::world_size]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Track which token positions have been scored (only count each position once)
    scored = torch.zeros(total_tokens, dtype=torch.bool)

    model.eval()
    with torch.inference_mode():
        for start in rank_starts:
            end = start + seq_len + 1
            if end > total_tokens:
                break
            local = val_tokens[start:end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].unsqueeze(0)  # (1, seq_len)
            y = local[1:].unsqueeze(0)   # (1, seq_len)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x, targets=None)  # (1, seq_len, vocab_size)

            # Only score tokens in the last (seq_len - stride) positions for middle windows,
            # or all tokens for the first window
            if start == 0:
                score_start = 0
            else:
                score_start = seq_len - stride

            score_positions = slice(score_start, seq_len)
            score_logits = logits[0, score_positions, :]
            score_targets = y[0, score_positions]

            # Apply softcap
            score_logits = args.softcap_A * torch.sigmoid(
                (score_logits + args.softcap_B) / args.softcap_C
            )

            per_token_loss = F.cross_entropy(
                score_logits.float(), score_targets, reduction="none"
            )

            # Mark which global positions we scored
            global_positions = torch.arange(
                start + 1 + score_start, start + 1 + seq_len, device="cpu"
            )
            # Avoid double-counting
            mask = ~scored[global_positions]
            scored[global_positions] = True

            mask_device = mask.to(device)
            per_token_loss = per_token_loss * mask_device.float()
            num_scored = mask_device.sum().item()

            val_loss_sum += per_token_loss.to(torch.float64).sum()
            val_token_count += num_scored

            # Byte counting for scored tokens
            prev_ids = x[0, score_positions].reshape(-1)
            tgt_ids = score_targets.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += (token_bytes.to(torch.float64) * mask_device.to(torch.float64)).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_ttt(
    args: Hyperparameters,
    base_model: nn.Module,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Test-Time Training eval: adapt the model on already-scored tokens.
    Every ttt_interval windows, do a few gradient steps on recently seen tokens.
    This is the neural equivalent of PAQ/cmix online weight updates."""
    seq_len = args.train_seq_len
    stride = args.eval_stride
    total_tokens = val_tokens.numel()
    ttt_interval = int(os.environ.get("TTT_INTERVAL", "200"))  # adapt every N windows
    ttt_lr = float(os.environ.get("TTT_LR", "1e-4"))
    ttt_steps = int(os.environ.get("TTT_STEPS", "3"))

    starts = list(range(0, total_tokens - seq_len, stride))
    if not starts:
        starts = [0]
    rank_starts = starts[rank::world_size]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    scored = torch.zeros(total_tokens, dtype=torch.bool)

    # TTT optimizer: only adapt embedding + final norm + last few block params
    # Keep it lightweight to avoid blowing up eval time
    ttt_params = [p for p in base_model.tok_emb.parameters()]
    ttt_params.extend(base_model.final_norm.parameters())
    # Add the last block's MLP parameters (small, high impact on output)
    last_block = base_model.blocks[-1]
    ttt_params.extend(last_block.mlp.parameters())
    ttt_optimizer = torch.optim.Adam(ttt_params, lr=ttt_lr, betas=(0.9, 0.99))

    recent_chunks: list[Tensor] = []  # buffer of recent token sequences for TTT
    window_count = 0

    model.eval()
    for start in rank_starts:
        end = start + seq_len + 1
        if end > total_tokens:
            break
        local = val_tokens[start:end].to(device=device, dtype=torch.int64, non_blocking=True)
        x = local[:-1].unsqueeze(0)
        y = local[1:].unsqueeze(0)

        # Score this window (no gradients for scoring)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x, targets=None)

        if start == 0:
            score_start = 0
        else:
            score_start = seq_len - stride

        score_positions = slice(score_start, seq_len)
        score_logits = logits[0, score_positions, :]
        score_targets = y[0, score_positions]
        score_logits = args.softcap_A * torch.sigmoid(
            (score_logits + args.softcap_B) / args.softcap_C
        )
        per_token_loss = F.cross_entropy(
            score_logits.float(), score_targets, reduction="none"
        )
        global_positions = torch.arange(
            start + 1 + score_start, start + 1 + seq_len, device="cpu"
        )
        mask = ~scored[global_positions]
        scored[global_positions] = True
        mask_device = mask.to(device)
        per_token_loss = per_token_loss * mask_device.float()
        num_scored = mask_device.sum().item()
        val_loss_sum += per_token_loss.to(torch.float64).sum()
        val_token_count += num_scored
        prev_ids = x[0, score_positions].reshape(-1)
        tgt_ids = score_targets.reshape(-1)
        token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
        val_byte_count += (token_bytes.to(torch.float64) * mask_device.to(torch.float64)).sum()

        # Buffer this window for TTT
        recent_chunks.append(local.detach())
        if len(recent_chunks) > 10:
            recent_chunks.pop(0)

        # Periodically adapt model on recent tokens
        window_count += 1
        if window_count % ttt_interval == 0 and len(recent_chunks) >= 3:
            model.train()
            for _ in range(ttt_steps):
                ttt_optimizer.zero_grad()
                # Pick a random recent chunk
                chunk = recent_chunks[torch.randint(len(recent_chunks), (1,)).item()]
                ttt_x = chunk[:-1].unsqueeze(0)
                ttt_y = chunk[1:].unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    ttt_loss = model(ttt_x, ttt_y)
                ttt_loss.backward()
                ttt_optimizer.step()
            model.eval()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# K-MEANS CODEBOOK QUANTIZATION
# -----------------------------

def quantize_tensor_kmeans(t: Tensor, n_clusters: int = 16) -> tuple[Tensor, Tensor, tuple]:
    """Quantize a tensor using k-means clustering for non-uniform quantization.
    Returns (indices, codebook, original_shape).
    indices: uint8 tensor of cluster assignments
    codebook: float16 tensor of cluster centroids
    """
    shape = t.shape
    flat = t.float().reshape(-1)
    n = flat.numel()

    # Fast k-means initialization: use quantile-based init for speed
    init_centroids = torch.quantile(flat, torch.linspace(0, 1, n_clusters))

    # Run k-means (few iterations suffice for weight quantization)
    centroids = init_centroids.clone()
    for _ in range(20):
        # Assign each value to nearest centroid
        dists = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters)
        for c in range(n_clusters):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = flat[mask].mean()
                counts[c] = mask.sum()
            else:
                new_centroids[c] = centroids[c]
        centroids = new_centroids

    # Final assignment
    dists = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1).to(torch.uint8)

    return indices.reshape(shape), centroids.to(torch.float16), shape


def dequantize_tensor_kmeans(indices: Tensor, codebook: Tensor, dtype: torch.dtype) -> Tensor:
    """Reconstruct tensor from k-means indices and codebook."""
    return codebook[indices.long()].to(dtype)


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


# K-means codebook quantization: 4-bit (16 centroids) per row
KMEANS_NUM_CENTROIDS = int(os.environ.get("KMEANS_CENTROIDS", "16"))
KMEANS_ITERS = int(os.environ.get("KMEANS_ITERS", "20"))

def _kmeans_1d(values: Tensor, k: int, n_iters: int) -> tuple[Tensor, Tensor]:
    """Run 1-D k-means on a flat float tensor. Returns (centroids [k], assignments [N])."""
    N = values.numel()
    if N == 0:
        return torch.zeros(k, dtype=values.dtype), torch.zeros(0, dtype=torch.int64)
    # Init centroids via quantile spacing
    q = torch.linspace(0, 1, k, device=values.device)
    centroids = torch.quantile(values, q)
    for _ in range(n_iters):
        # Assign each value to nearest centroid
        dists = (values.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # [N, k]
        assignments = dists.argmin(dim=1)
        # Update centroids
        for j in range(k):
            mask = assignments == j
            if mask.any():
                centroids[j] = values[mask].mean()
    return centroids, assignments

def _pack_4bit(indices: Tensor) -> Tensor:
    """Pack uint8 indices (0-15) into 4-bit pairs. Input length must be even."""
    assert indices.dtype == torch.uint8
    flat = indices.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    high = flat[0::2] << 4
    low = flat[1::2] & 0x0F
    return (high | low).contiguous()

def _unpack_4bit(packed: Tensor, orig_numel: int) -> Tensor:
    """Unpack 4-bit packed tensor back to uint8."""
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    interleaved = torch.stack([high, low], dim=-1).reshape(-1)
    return interleaved[:orig_numel].contiguous()

def quantize_tensor_kmeans(t: Tensor, k: int = KMEANS_NUM_CENTROIDS, n_iters: int = KMEANS_ITERS) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize a tensor using k-means codebook. Returns (packed_indices, codebook, orig_shape).
    For 2D: per-row k-means with 4-bit packed indices (half the storage of int8).
    For 1D: global k-means with 4-bit packing."""
    assert k <= 16, f"4-bit packing supports at most 16 centroids, got {k}"
    t32 = t.float()
    orig_shape = torch.tensor(list(t.shape), dtype=torch.int32)
    if t32.ndim == 2:
        rows, cols = t32.shape
        all_indices = torch.zeros(rows, cols, dtype=torch.uint8)
        all_codebooks = torch.zeros(rows, k, dtype=torch.float16)
        for r in range(rows):
            centroids, assignments = _kmeans_1d(t32[r], k, n_iters)
            all_indices[r] = assignments.to(torch.uint8)
            all_codebooks[r] = centroids.to(torch.float16)
        # Pack each row's indices into 4-bit
        packed_rows = []
        for r in range(rows):
            packed_rows.append(_pack_4bit(all_indices[r]))
        packed = torch.stack(packed_rows)
        return packed.contiguous(), all_codebooks.contiguous(), orig_shape
    else:
        centroids, assignments = _kmeans_1d(t32.flatten(), k, n_iters)
        packed = _pack_4bit(assignments.to(torch.uint8))
        return packed.contiguous(), centroids.to(torch.float16).contiguous(), orig_shape

def dequantize_tensor_kmeans(packed: Tensor, codebook: Tensor, orig_shape: Tensor) -> Tensor:
    """Dequantize k-means quantized tensor from 4-bit packed format."""
    shape = tuple(orig_shape.tolist())
    if codebook.ndim == 2:
        rows = codebook.shape[0]
        cols = shape[1]
        out = torch.zeros(rows, cols, dtype=torch.float32)
        for r in range(rows):
            indices = _unpack_4bit(packed[r], cols)
            out[r] = codebook[r].float()[indices.long()]
        return out
    else:
        numel = 1
        for s in shape:
            numel *= s
        indices = _unpack_4bit(packed, numel)
        return codebook.float()[indices.long()].reshape(shape)

USE_KMEANS_QUANT = bool(int(os.environ.get("USE_KMEANS", "1")))

def quantize_state_dict_int8(state_dict: dict[str, Tensor], fp16_passthrough_names: set[str] | None = None):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    if fp16_passthrough_names is None:
        fp16_passthrough_names = set()

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Force fp16 passthrough for specified tensors (e.g. tied embedding)
        if name in fp16_passthrough_names:
            kept = t.to(dtype=torch.float16).contiguous()
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1

        if USE_KMEANS_QUANT and t.ndim >= 1 and t.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            # K-means 4-bit quantization: packed indices (half size) + codebook
            packed, codebook, orig_shape = quantize_tensor_kmeans(t, k=KMEANS_NUM_CENTROIDS)
            quantized[name] = packed
            scales[name] = codebook
            scheme = "kmeans_per_row" if t.ndim == 2 else "kmeans_global"
            qmeta[name] = {"scheme": scheme, "k": KMEANS_NUM_CENTROIDS, "orig_shape": orig_shape.tolist()}
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_payload_bytes"] += tensor_nbytes(packed) + tensor_nbytes(codebook)
        else:
            q, s = quantize_float_tensor(t)
            if s.ndim > 0:
                qmeta[name] = {"scheme": "per_row", "axis": 0}
            quantized[name] = q
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "kmeans_int4_v1" if USE_KMEANS_QUANT else "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        scheme = qmeta.get(name, {}).get("scheme", "")
        if scheme.startswith("kmeans"):
            # K-means dequantization: s is the codebook, q is packed 4-bit indices
            orig_shape_list = qmeta[name]["orig_shape"]
            orig_shape = torch.tensor(orig_shape_list, dtype=torch.int32)
            out[name] = dequantize_tensor_kmeans(q, s, orig_shape).to(dtype=dtype).contiguous()
        elif scheme == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class LoRALinear(nn.Module):
    """Low-rank adapter applied on top of a base CastedLinear."""
    def __init__(self, base: CastedLinear, rank: int):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        # Initialize A with small random values, B with zeros (so LoRA starts as identity)
        nn.init.normal_(self.lora_A, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return base_out + lora_out


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, lora_adapters: dict | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape

        # Apply LoRA adapters if provided
        c_q = lora_adapters["q"] if lora_adapters and "q" in lora_adapters else self.c_q
        c_k = lora_adapters["k"] if lora_adapters and "k" in lora_adapters else self.c_k
        c_v = lora_adapters["v"] if lora_adapters and "v" in lora_adapters else self.c_v
        proj = lora_adapters["o"] if lora_adapters and "o" in lora_adapters else self.proj

        q = c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return proj(y)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP: gate(x) * up(x), then project down."""
    def __init__(self, dim: int, mult: float):
        super().__init__()
        hidden = int(mult * dim)
        # Round to nearest multiple of 64 for efficiency
        hidden = ((hidden + 63) // 64) * 64
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.down = CastedLinear(hidden, dim, bias=False)
        self.down._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        swiglu_mult: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = SwiGLUMLP(dim, swiglu_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, lora_adapters: dict | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), lora_adapters=lora_adapters)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_unique_layers: int,
        num_recurrence_loops: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        swiglu_mult: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        softcap_A: float,
        softcap_B: float,
        softcap_C: float,
        rope_base: float,
        qk_gain_init: float,
        lora_rank: int,
        mtp_num_heads: int,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.softcap_A = softcap_A
        self.softcap_B = softcap_B
        self.softcap_C = softcap_C
        self.num_unique_layers = num_unique_layers
        self.num_recurrence_loops = num_recurrence_loops
        self.num_effective_layers = num_unique_layers * num_recurrence_loops
        self.mtp_num_heads = mtp_num_heads

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # U-net skip connections across effective layers
        self.num_encoder_layers = self.num_effective_layers // 2
        self.num_decoder_layers = self.num_effective_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Unique transformer blocks (shared across loops)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, swiglu_mult, rope_base, qk_gain_init)
            for _ in range(num_unique_layers)
        ])

        # Per-loop LoRA adapters for Q, K, V, O
        # Loop 0 uses base weights (no LoRA), loops 1+ get adapters
        self.lora_adapters = nn.ModuleDict()
        for loop_idx in range(1, num_recurrence_loops):
            for layer_idx in range(num_unique_layers):
                prefix = f"loop{loop_idx}_layer{layer_idx}"
                attn = self.blocks[layer_idx].attn
                self.lora_adapters[f"{prefix}_q"] = LoRALinear(attn.c_q, lora_rank)
                self.lora_adapters[f"{prefix}_k"] = LoRALinear(attn.c_k, lora_rank)
                self.lora_adapters[f"{prefix}_v"] = LoRALinear(attn.c_v, lora_rank)
                self.lora_adapters[f"{prefix}_o"] = LoRALinear(attn.proj, lora_rank)

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # Multi-token prediction heads (lightweight linear projections)
        # Head 0 is the main next-token prediction (uses lm_head / tied embed)
        # Heads 1..mtp_num_heads-1 predict tokens 2..mtp_num_heads steps ahead
        if mtp_num_heads > 1:
            self.mtp_heads = nn.ModuleList([
                CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads - 1)
            ])
            for head in self.mtp_heads:
                head._zero_init = True
        else:
            self.mtp_heads = None

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _get_lora_adapters(self, loop_idx: int, layer_idx: int) -> dict | None:
        if loop_idx == 0:
            return None
        prefix = f"loop{loop_idx}_layer{layer_idx}"
        return {
            "q": self.lora_adapters[f"{prefix}_q"],
            "k": self.lora_adapters[f"{prefix}_k"],
            "v": self.lora_adapters[f"{prefix}_v"],
            "o": self.lora_adapters[f"{prefix}_o"],
        }

    def _apply_softcap(self, logits: Tensor) -> Tensor:
        return self.softcap_A * torch.sigmoid((logits + self.softcap_B) / self.softcap_C)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None,
                mtp_weights: list[float] | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        effective_layer_idx = 0
        for loop_idx in range(self.num_recurrence_loops):
            for layer_idx in range(self.num_unique_layers):
                # U-net skip connections
                if effective_layer_idx < self.num_encoder_layers:
                    lora = self._get_lora_adapters(loop_idx, layer_idx)
                    x = self.blocks[layer_idx](x, x0, lora_adapters=lora)
                    skips.append(x)
                else:
                    decoder_idx = effective_layer_idx - self.num_encoder_layers
                    if decoder_idx < self.num_skip_weights and skips:
                        x = x + self.skip_weights[decoder_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    lora = self._get_lora_adapters(loop_idx, layer_idx)
                    x = self.blocks[layer_idx](x, x0, lora_adapters=lora)
                effective_layer_idx += 1

        x = self.final_norm(x)

        if target_ids is None:
            # Inference mode: return logits
            if self.tie_embeddings:
                logits = F.linear(x, self.tok_emb.weight)
            else:
                logits = self.lm_head(x)
            return logits

        # Training mode: compute loss
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits = self.lm_head(x_flat)
        logits = self._apply_softcap(logits)
        loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        # Multi-token prediction auxiliary losses
        if self.mtp_heads is not None and mtp_weights is not None and len(mtp_weights) > 1:
            total_weight = mtp_weights[0]
            weighted_loss = loss * mtp_weights[0]
            seq_len = input_ids.size(1)
            for head_idx, mtp_head in enumerate(self.mtp_heads):
                k = head_idx + 2  # predicting k steps ahead
                if k > seq_len:
                    break
                w = mtp_weights[head_idx + 1] if (head_idx + 1) < len(mtp_weights) else 0.0
                if w <= 0.0:
                    continue
                # Shift targets: predict token at position t+k from hidden state at position t
                mtp_targets = target_ids[:, k-1:].reshape(-1)
                mtp_hidden = x[:, :-(k-1), :].reshape(-1, x.size(-1)) if k > 1 else x_flat
                mtp_logits = mtp_head(mtp_hidden)
                mtp_logits = self._apply_softcap(mtp_logits)
                mtp_loss = F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                weighted_loss += w * mtp_loss
                total_weight += w
            loss = weighted_loss / total_weight

        return loss


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # DISTRIBUTED + CUDA SETUP
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # TOKENIZER + VALIDATION SETUP
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # MODEL + OPTIMIZER SETUP
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_unique_layers=args.num_unique_layers,
        num_recurrence_loops=args.num_recurrence_loops,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        swiglu_mult=args.swiglu_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        softcap_A=args.softcap_A,
        softcap_B=args.softcap_B,
        softcap_C=args.softcap_C,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        lora_rank=args.lora_rank,
        mtp_num_heads=args.mtp_num_heads,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    # LoRA parameters
    lora_params = list(base_model.lora_adapters.parameters())

    # MTP head parameters
    mtp_params = []
    if base_model.mtp_heads is not None:
        mtp_params = list(base_model.mtp_heads.parameters())

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    if lora_params:
        optimizer_lora = torch.optim.Adam(
            [{"params": lora_params, "lr": args.lora_lr, "base_lr": args.lora_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.append(optimizer_lora)

    if mtp_params:
        optimizer_mtp = torch.optim.Adam(
            [{"params": mtp_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.append(optimizer_mtp)

    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.append(optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"unique_layers:{args.num_unique_layers} loops:{args.num_recurrence_loops} "
         f"effective_layers:{args.num_unique_layers * args.num_recurrence_loops}")
    log0(f"lora_rank:{args.lora_rank} lora_params:{sum(p.numel() for p in lora_params)}")
    log0(f"mtp_heads:{args.mtp_num_heads}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"vocab_size:{args.vocab_size} model_dim:{args.model_dim}")

    # DATA LOADER & WARMUP
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def get_mtp_weights(step: int, total_steps: int) -> list[float]:
        """Scheduled MTP weight fadeout: 3-token -> 2-token -> 1-token."""
        if args.mtp_num_heads <= 1:
            return [1.0]
        frac = step / max(total_steps, 1)
        if frac < 0.33:
            # Phase 1: all heads active, gradually fade head 3
            t = frac / 0.33
            return [1.0, 0.5, 0.25 * (1.0 - t)]
        elif frac < 0.67:
            # Phase 2: head 3 gone, gradually fade head 2
            t = (frac - 0.33) / 0.34
            return [1.0, 0.5 * (1.0 - t)]
        else:
            # Phase 3: only next-token prediction
            return [1.0]

    # Warmup for torch.compile
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y, mtp_weights=[1.0, 0.5, 0.25])
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # MAIN TRAINING LOOP
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        mtp_weights = get_mtp_weights(step, args.iterations)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, mtp_weights=mtp_weights)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # SERIALIZATION + ROUNDTRIP VALIDATION
    # Remove MTP heads before saving (they're training-only)
    # Keep tied embedding in fp16 (most quantization-sensitive tensor, per PR#66)
    save_state = {}
    for name, tensor in base_model.state_dict().items():
        if "mtp_heads" not in name:
            save_state[name] = tensor

    if master_process:
        torch.save(save_state, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    # fp16 embedding export: keep embedding in fp16 (not int8) during quantization.
    # The tied embedding is the most quantization-sensitive tensor (PR#66).
    fp16_passthrough_names = set()
    if args.embed_fp16_export and args.tie_embeddings:
        fp16_passthrough_names.add("tok_emb.weight")
    quant_obj, quant_stats = quantize_state_dict_int8(save_state, fp16_passthrough_names=fp16_passthrough_names)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    # Roundtrip validation - reload quantized model
    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    deq_state = dequantize_state_dict_int8(quant_state)
    # Re-add MTP heads as zeros for loading (they won't be used in eval)
    if base_model.mtp_heads is not None:
        for name, param in base_model.named_parameters():
            if "mtp_heads" in name and name not in deq_state:
                deq_state[name] = torch.zeros_like(param, device="cpu")
    base_model.load_state_dict(deq_state, strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # TTT evaluation: reload fresh quantized weights, then adapt during eval
    # In distributed mode, all ranks process the same windows sequentially
    # so TTT updates are consistent (each rank sees same data, same gradients)
    use_ttt = bool(int(os.environ.get("USE_TTT", "1")))
    if use_ttt:
        base_model.load_state_dict(deq_state, strict=True)
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_val_ttt(
            args, base_model, compiled_model if not distributed else model,
            rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(
            f"final_ttt_eval val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"final_ttt_eval_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
