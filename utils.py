"""
Lab 2 - Utility functions for the Pruning assignment.

Covers: CIFAR-10 data loading, a CIFAR-sized MobileNetV2, a minimal
train/eval loop, model size / sparsity / latency measurement, and
wrappers around `torch.nn.utils.prune` for unstructured and (naive)
structured pruning.

The dependency-graph-aware structured pruning (via the `torch-pruning`
library) is intentionally left in the notebook itself rather than here,
since building the pruner and choosing which layers to ignore is a core
part of what this lab is teaching.
"""
from __future__ import annotations

import copy
import io
import math
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as T


# --------------------------------------------------------------------------
# Device / data
# --------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """Count nonzero parameters — meaningful after *unstructured* pruning,
    where pruned weights are zeroed but tensors keep their original shape."""
    return sum(int(torch.count_nonzero(p)) for p in model.parameters())


def global_sparsity(model: nn.Module) -> float:
    """Fraction of zero-valued parameters across the whole model."""
    total = count_parameters(model)
    nonzero = count_nonzero_parameters(model)
    return 1.0 - (nonzero / total)


def sparsity_report(model: nn.Module, layer_types=(nn.Conv2d, nn.Linear)) -> list[dict]:
    """Per-layer sparsity for all layers of the given types."""
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, layer_types) and hasattr(module, "weight"):
            w = module.weight.data
            n_total = w.numel()
            n_zero = int((w == 0).sum())
            rows.append({
                "layer": name,
                "shape": tuple(w.shape),
                "sparsity": n_zero / n_total if n_total else 0.0,
            })
    return rows


def model_size_mb(model: nn.Module) -> float:
    """Serialized size of a model's state_dict, in megabytes.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024 ** 2)


def save_checkpoint(model: nn.Module, path: str | Path, metadata=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.state_dict() if metadata is None else {**metadata, "model_state_dict": model.state_dict()}
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str | Path, map_location=None) -> nn.Module:
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict.get("model_state_dict", state_dict))
    return model


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """Returns (avg_loss, accuracy in percent)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def fit(
    model, train_loader, val_loader, device,
    epochs: int = 3, lr: float = 1e-3, weight_decay: float = 1e-4,
    verbose: bool = True, checkpoint_path=None, checkpoint_metadata=None,
) -> nn.Module:
    """Minimal training loop, used both for the initial baseline and for
    post-pruning fine-tuning."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    best_loss = float("inf")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        if not math.isfinite(val_loss):
            raise RuntimeError("Validation loss is non-finite; refusing to select a checkpoint")
        if val_loss < best_loss:
            best_loss = val_loss
            if checkpoint_path is not None:
                save_checkpoint(model, checkpoint_path, {
                    **(checkpoint_metadata or {}), "epoch": epoch,
                    "val_loss": val_loss, "val_accuracy": val_acc,
                })
        if verbose:
            print(
                f"epoch {epoch}/{epochs} - "
                f"train_loss: {train_loss:.4f} - "
                f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}%"
            )
    return model


@torch.no_grad()
def benchmark_latency(
    model: nn.Module, input_size=(1, 3, 224, 224), device=None,
    n_warmup: int = 10, n_runs: int = 50,
) -> float:
    """Average single-batch inference latency, in ms."""
    device = device or get_device()
    model.eval().to(device)
    dummy = torch.randn(*input_size, device=device)

    for _ in range(n_warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0


# --------------------------------------------------------------------------
# Pruning (torch.nn.utils.prune) — unstructured & naive structured
# --------------------------------------------------------------------------

def _prunable_layers(model: nn.Module, layer_types=(nn.Conv2d, nn.Linear)):
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, layer_types)
    ]


def apply_global_unstructured_pruning(
    model: nn.Module, amount: float, layer_types=(nn.Conv2d, nn.Linear),
) -> nn.Module:
    """Global magnitude-based (L1) unstructured pruning.

    Prunes ``amount`` (0-1) fraction of the *smallest-magnitude* weights
    across all matching layers combined — not per-layer — which usually
    beats per-layer pruning at the same overall sparsity. Weights are
    masked (set to zero) but the tensor shapes are unchanged; call
    ``remove_pruning_reparam`` once you're done pruning/fine-tuning to
    make the masks permanent.
    """
    params_to_prune = _prunable_layers(model, layer_types)
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def apply_structured_pruning_torch(
    model: nn.Module, amount: float, layer_types=(nn.Conv2d,), dim: int = 0,
) -> nn.Module:
    """Naive structured pruning using ``torch.nn.utils.prune.ln_structured``,
    applied independently to each matching layer.

    This zeroes out whole output channels (``dim=0``) ranked by L2 norm,
    per layer. Important caveat to discuss in your report: because this
    is applied layer-by-layer with no knowledge of cross-layer
    dependencies (e.g. skip connections, grouped/depthwise convs), the
    channels are *masked*, not physically removed — tensor shapes are
    unchanged and there is no real speedup. Compare this against the
    DepGraph-based structured pruning in Section 4, which does remove
    channels and their dependents.
    """
    for module in model.modules():
        if isinstance(module, layer_types):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=dim)
    return model


def remove_pruning_reparam(model: nn.Module, layer_types=(nn.Conv2d, nn.Linear)) -> nn.Module:
    """Make pruning masks permanent (removes the ``weight_orig``/``weight_mask``
    reparametrization so the model behaves like an ordinary module again)."""
    for module in model.modules():
        if isinstance(module, layer_types) and prune.is_pruned(module):
            prune.remove(module, "weight")
    return model


def clone_model(model: nn.Module) -> nn.Module:
    """Deep-copy a model — handy for comparing multiple pruning strategies
    starting from the same trained baseline."""
    return copy.deepcopy(model)
