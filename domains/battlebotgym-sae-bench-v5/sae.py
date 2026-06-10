# SAE Architecture — agents edit this file
#
# The default engine uses sae_lens's built-in BatchTopKTrainingSAE.
# To use a custom architecture:
#   1. Define your SAE class and config class here
#   2. Set sae_class: YourClassName in config.yaml
#   3. Your config class needs a from_dict(cfg, total_steps) classmethod
#
# The encoder/decoder interface:
#   - forward(x) -> (sae_out, feature_acts, loss, loss_dict)
#   - encode(x) -> feature_acts
#   - decode(feature_acts) -> sae_out
#
# Useful imports:
#   from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig
#   from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from sae_lens.saes.sae import (
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
)
from sae_lens.saes.topk_sae import TopK, TopKTrainingSAE, TopKTrainingSAEConfig, act_times_W_dec
from sae_lens.saes.batchtopk_sae import BatchTopK, BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig


# ─── PureISTASAE (agent0) ────────────────────────────────────────────
# Stripped-down: just TopK + multi-step ISTA + k-annealing. No matryoshka,
# no freq_sort, no term_tilt. Tests whether ISTA alone drives F1.

@dataclass
class PureISTASAEConfig(TopKTrainingSAEConfig):
    n_ista_steps: int = 5
    ista_step_size: float = 0.3
    ista_decay: float = 0.7
    initial_k: int = 100
    k_schedule: str = "cosine"
    total_steps: int = 50000

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "PureISTASAEConfig":
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            n_ista_steps=int(cfg.get('n_ista_steps', 5)),
            ista_step_size=float(cfg.get('ista_step_size', 0.3)),
            ista_decay=float(cfg.get('ista_decay', 0.7)),
            initial_k=int(cfg.get('initial_k', 100)),
            k_schedule=cfg.get('k_schedule', 'cosine'),
            total_steps=total_steps,
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class PureISTASAE(TopKTrainingSAE):
    """Minimal SAE: TopK + multi-step decaying ISTA + cosine k-annealing. No extras."""
    cfg: PureISTASAEConfig
    _step: int

    def __init__(self, cfg: PureISTASAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self._step = 0

    def _current_k(self) -> int:
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac))

    def _topk_with_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk_values, topk_indices = torch.topk(x, k=k, dim=-1, sorted=False)
        values = topk_values.relu()
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, values)
        return result

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(self.cfg.k)

    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k()
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # Multi-step decaying ISTA
        if self.cfg.n_ista_steps > 0 and self.training:
            step_size = self.cfg.ista_step_size
            for _ in range(self.cfg.n_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)
                step_size *= self.cfg.ista_decay

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        per_item_mse = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse.sum(dim=-1).mean()

        aux_losses = self.calculate_aux_loss(
            step_input=step_input, feature_acts=feature_acts,
            hidden_pre=hidden_pre, sae_out=sae_out,
        )

        total_loss = mse_loss
        losses = {"mse_loss": mse_loss}
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)
            for v in aux_losses.values():
                total_loss = total_loss + v
        else:
            total_loss = total_loss + aux_losses

        return TrainStepOutput(
            sae_in=step_input.sae_in, sae_out=sae_out, feature_acts=feature_acts,
            hidden_pre=hidden_pre, loss=total_loss, losses=losses,
        )


# ─── ReferenceStyleSAE ───────────────────────────────────────────────
# Combines: matryoshka multi-scale loss, ISTA refinement, k-annealing,
# frequency-based feature sorting, and term-tilt sparsity.

@dataclass
class ReferenceStyleSAEConfig(TopKTrainingSAEConfig):
    # Matryoshka widths for multi-scale training
    matryoshka_widths: list[int] = field(default_factory=lambda: [32, 128, 512, 1024, 2048, 4096])
    detach_matryoshka: bool = True
    inner_loss_weight: float = 0.5

    # ISTA refinement
    n_ista_steps: int = 1
    ista_step_size: float = 0.25

    # K-annealing: start with initial_k, anneal to k
    initial_k: int = 80

    # Frequency-based sorting
    use_freq_sort: bool = True

    # Term tilt (L1-like sparsity penalty)
    term_tilt: float = 0.01

    # Total training steps (set by from_dict)
    total_steps: int = 50000

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "ReferenceStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.01)),
            total_steps=total_steps,
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class ReferenceStyleSAE(TopKTrainingSAE):
    """
    Advanced SAE with matryoshka training, ISTA refinement, k-annealing,
    frequency sorting, and term-tilt sparsity.
    """
    cfg: ReferenceStyleSAEConfig
    _step: int
    _feature_counts: torch.Tensor

    def __init__(self, cfg: ReferenceStyleSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self._step = 0
        # Track feature activation frequency for sorting
        self.register_buffer('_feature_counts', torch.zeros(cfg.d_sae, device=cfg.device))

    def _current_k(self) -> int:
        """Anneal k from initial_k down to target k over first 60% of training."""
        if self._step >= self.cfg.total_steps * 0.6:
            return self.cfg.k
        frac = self._step / (self.cfg.total_steps * 0.6)
        return int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(self.cfg.k)

    def _topk_with_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply TopK with a specific k value."""
        topk_values, topk_indices = torch.topk(x, k=k, dim=-1, sorted=False)
        values = topk_values.relu()
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, values)
        return result

    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        # Use annealed k during training
        current_k = self._current_k()
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA refinement steps
        if self.cfg.n_ista_steps > 0 and self.training:
            for _ in range(self.cfg.n_ista_steps):
                # Compute residual
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                # Gradient step on the code
                grad = residual @ self.W_enc  # d_sae
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @torch.no_grad()
    def _sort_features_by_freq(self):
        """Sort encoder/decoder columns by descending activation frequency."""
        if not self.cfg.use_freq_sort:
            return
        # Sort every 1000 steps
        if self._step % 1000 != 0 or self._step == 0:
            return
        _, sort_idx = self._feature_counts.sort(descending=True)
        # Reorder weights
        self.W_enc.data = self.W_enc.data[:, sort_idx]
        self.b_enc.data = self.b_enc.data[sort_idx]
        self.W_dec.data = self.W_dec.data[sort_idx, :]
        self._feature_counts.data = self._feature_counts.data[sort_idx]

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps

        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        # Track feature usage
        with torch.no_grad():
            active = (feature_acts > 0).float().sum(dim=0)
            self._feature_counts = self._feature_counts * 0.99 + active * 0.01

        # Sort features by frequency periodically
        self._sort_features_by_freq()

        # Main MSE loss
        per_item_mse = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse.sum(dim=-1).mean()

        # Matryoshka multi-scale loss
        matryoshka_loss = torch.tensor(0.0, device=mse_loss.device)
        widths = sorted(self.cfg.matryoshka_widths)
        for w in widths:
            if w >= self.cfg.d_sae:
                continue
            sub_acts = feature_acts[:, :w]
            if self.cfg.detach_matryoshka:
                sub_acts = sub_acts.detach()
            sub_out = sub_acts @ self.W_dec[:w, :] + self.b_dec
            sub_mse = (sub_out - step_input.sae_in).pow(2).sum(dim=-1).mean()
            matryoshka_loss = matryoshka_loss + sub_mse

        matryoshka_loss = matryoshka_loss / max(len(widths) - 1, 1)

        # Term tilt (L1-like sparsity on activations)
        term_loss = self.cfg.term_tilt * feature_acts.abs().sum(dim=-1).mean()

        # Aux loss for dead neurons
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        total_loss = mse_loss + self.cfg.inner_loss_weight * matryoshka_loss + term_loss
        losses = {
            "mse_loss": mse_loss,
            "matryoshka_loss": matryoshka_loss,
            "term_tilt_loss": term_loss,
        }

        if isinstance(aux_losses, dict):
            losses.update(aux_losses)
            for v in aux_losses.values():
                total_loss = total_loss + v
        else:
            total_loss = total_loss + aux_losses

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )


# ─── AnnealedTermRefStyleSAE ─────────────────────────────────────────
# Same as ReferenceStyleSAE but with annealed term_tilt (decreasing over training).

@dataclass
class AnnealedTermRefStyleSAEConfig(ReferenceStyleSAEConfig):
    term_start: float = 0.015
    term_end: float = 0.008

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "AnnealedTermRefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.008)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class AnnealedTermRefStyleSAE(ReferenceStyleSAE):
    """ReferenceStyleSAE with annealed term_tilt: linearly decreases from term_start to term_end."""
    cfg: AnnealedTermRefStyleSAEConfig

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        # Temporarily override term_tilt with annealed value
        old_tilt = self.cfg.term_tilt
        self.cfg.term_tilt = self._current_term_tilt()
        result = super().training_forward_pass(step_input)
        self.cfg.term_tilt = old_tilt
        return result


# ─── EMARefStyleSAE ──────────────────────────────────────────────────
# ReferenceStyleSAE + EMA decoder for stable matryoshka targets.

@dataclass
class EMARefStyleSAEConfig(ReferenceStyleSAEConfig):
    ema_decay: float = 0.999
    ema_start_frac: float = 0.1

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "EMARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            ema_decay=float(cfg.get('ema_decay', 0.999)),
            ema_start_frac=float(cfg.get('ema_start_frac', 0.1)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class EMARefStyleSAE(ReferenceStyleSAE):
    """ReferenceStyleSAE with EMA-smoothed decoder for matryoshka loss stability."""
    cfg: EMARefStyleSAEConfig

    def __init__(self, cfg: EMARefStyleSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # EMA copy of decoder
        self.register_buffer('W_dec_ema', self.W_dec.data.clone())
        self.register_buffer('b_dec_ema', self.b_dec.data.clone())

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps

        # Update EMA after warmup
        ema_start = int(self.cfg.ema_start_frac * self.cfg.total_steps)
        if self._step > ema_start:
            d = self.cfg.ema_decay
            self.W_dec_ema.data.mul_(d).add_(self.W_dec.data, alpha=1 - d)
            self.b_dec_ema.data.mul_(d).add_(self.b_dec.data, alpha=1 - d)

        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        with torch.no_grad():
            active = (feature_acts > 0).float().sum(dim=0)
            self._feature_counts = self._feature_counts * 0.99 + active * 0.01
        self._sort_features_by_freq()

        mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in).sum(dim=-1).mean()

        # Matryoshka using EMA decoder for stable targets
        matryoshka_loss = torch.tensor(0.0, device=mse_loss.device)
        use_ema = self._step > ema_start
        dec_w = self.W_dec_ema if use_ema else self.W_dec
        dec_b = self.b_dec_ema if use_ema else self.b_dec

        widths = sorted(self.cfg.matryoshka_widths)
        for w in widths:
            if w >= self.cfg.d_sae:
                continue
            sub_acts = feature_acts[:, :w]
            if self.cfg.detach_matryoshka:
                sub_acts = sub_acts.detach()
            sub_out = sub_acts @ dec_w[:w, :] + dec_b
            sub_mse = (sub_out - step_input.sae_in).pow(2).sum(dim=-1).mean()
            matryoshka_loss = matryoshka_loss + sub_mse
        matryoshka_loss = matryoshka_loss / max(len(widths) - 1, 1)

        term_loss = self.cfg.term_tilt * feature_acts.abs().sum(dim=-1).mean()

        aux_losses = self.calculate_aux_loss(
            step_input=step_input, feature_acts=feature_acts,
            hidden_pre=hidden_pre, sae_out=sae_out,
        )

        total_loss = mse_loss + self.cfg.inner_loss_weight * matryoshka_loss + term_loss
        losses = {"mse_loss": mse_loss, "matryoshka_loss": matryoshka_loss, "term_tilt_loss": term_loss}
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)
            for v in aux_losses.values():
                total_loss = total_loss + v
        else:
            total_loss = total_loss + aux_losses

        return TrainStepOutput(
            sae_in=step_input.sae_in, sae_out=sae_out, feature_acts=feature_acts,
            hidden_pre=hidden_pre, loss=total_loss, losses=losses,
        )


# ─── WarmupTermRefStyleSAE ───────────────────────────────────────────
# Term tilt that warms up from 0 to target value.

@dataclass
class WarmupTermRefStyleSAEConfig(ReferenceStyleSAEConfig):
    term_warmup_frac: float = 0.2

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "WarmupTermRefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.01)),
            total_steps=total_steps,
            term_warmup_frac=float(cfg.get('term_warmup_frac', 0.2)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class WarmupTermRefStyleSAE(ReferenceStyleSAE):
    """Term tilt warms up from 0 to target over the first warmup_frac of training."""
    cfg: WarmupTermRefStyleSAEConfig

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        warmup_steps = int(self.cfg.term_warmup_frac * self.cfg.total_steps)
        if self._step < warmup_steps and warmup_steps > 0:
            frac = self._step / warmup_steps
            old_tilt = self.cfg.term_tilt
            self.cfg.term_tilt = old_tilt * frac
            result = super().training_forward_pass(step_input)
            self.cfg.term_tilt = old_tilt
            return result
        return super().training_forward_pass(step_input)


# ─── LowRankCorrRefStyleSAE ─────────────────────────────────────────
# Adds a low-rank correction to the decoder for fine-grained reconstruction.

@dataclass
class LowRankCorrRefStyleSAEConfig(ReferenceStyleSAEConfig):
    correction_rank: int = 16

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "LowRankCorrRefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            correction_rank=int(cfg.get('correction_rank', 16)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class LowRankCorrRefStyleSAE(ReferenceStyleSAE):
    """Adds low-rank correction matrices A, B to decoder: out = acts @ (W_dec + A @ B) + b_dec."""
    cfg: LowRankCorrRefStyleSAEConfig

    def __init__(self, cfg: LowRankCorrRefStyleSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        r = cfg.correction_rank
        self.corr_A = nn.Parameter(torch.zeros(cfg.d_sae, r, device=cfg.device))
        self.corr_B = nn.Parameter(torch.zeros(r, 768, device=cfg.device))
        nn.init.orthogonal_(self.corr_A)
        self.corr_B.data *= 0.01

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        base_out = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm)
        correction = feature_acts @ self.corr_A @ self.corr_B
        sae_out_pre = base_out + correction + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


# ─── AdaptiveISTARefStyleSAE ────────────────────────────────────────
# ISTA with learned per-feature step sizes.

@dataclass
class AdaptiveISTARefStyleSAEConfig(ReferenceStyleSAEConfig):
    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "AdaptiveISTARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class AdaptiveISTARefStyleSAE(ReferenceStyleSAE):
    """ISTA with learned per-feature step sizes instead of a fixed scalar."""
    cfg: AdaptiveISTARefStyleSAEConfig

    def __init__(self, cfg: AdaptiveISTARefStyleSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Learnable per-feature ISTA step size (initialized around ista_step_size)
        self.ista_step = nn.Parameter(
            torch.full((cfg.d_sae,), cfg.ista_step_size, device=cfg.device)
        )

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k()
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.cfg.n_ista_steps > 0 and self.training:
            step_sizes = self.ista_step.abs()  # ensure positive
            for _ in range(self.cfg.n_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_sizes * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── MomentumISTASAE (agent1) ─────────────────────────────────────
# Key improvements over ReferenceStyleSAE:
# 1. Momentum in ISTA steps (Nesterov-like acceleration)
# 2. Cosine k-schedule (smoother annealing)
# 3. Annealed term tilt (high→low)
# 4. More ISTA steps (3) with momentum accumulation

@dataclass
class MomentumISTASAEConfig(ReferenceStyleSAEConfig):
    ista_momentum: float = 0.9
    term_start: float = 0.015
    term_end: float = 0.005
    k_schedule: str = "cosine"  # "cosine" or "linear"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "MomentumISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 3)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            ista_momentum=float(cfg.get('ista_momentum', 0.9)),
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.005)),
            k_schedule=cfg.get('k_schedule', 'cosine'),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class MomentumISTASAE(ReferenceStyleSAE):
    """ISTA with Nesterov-like momentum + cosine k-schedule + annealed term tilt."""
    cfg: MomentumISTASAEConfig

    def _current_k(self) -> int:
        """Cosine or linear k-annealing from initial_k to target k."""
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac))

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k()
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # Momentum-ISTA: accumulate velocity across steps
        if self.cfg.n_ista_steps > 0 and self.training:
            mu = self.cfg.ista_momentum
            velocity = torch.zeros_like(hidden_pre)
            for i in range(self.cfg.n_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                velocity = mu * velocity + self.cfg.ista_step_size * grad
                updated = hidden_pre + velocity
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        old_tilt = self.cfg.term_tilt
        self.cfg.term_tilt = self._current_term_tilt()
        result = super().training_forward_pass(step_input)
        self.cfg.term_tilt = old_tilt
        return result


# ─── GatedRefStyleSAE (agent3) ───────────────────────────────────
# Key innovation: separate gating path for feature selection.
# The gate determines WHICH features fire (topk on gate scores),
# the magnitude path determines HOW MUCH. This decouples selection
# from magnitude estimation for better feature recovery.
# Also includes: decoder column normalization, cosine k-schedule.

@dataclass
class GatedRefStyleSAEConfig(ReferenceStyleSAEConfig):
    gate_bias_init: float = -1.0  # initial gate bias (negative = conservative)
    normalize_decoder: bool = True  # unit-norm decoder columns
    k_schedule: str = "cosine"  # cosine or linear k annealing
    term_start: float = 0.015
    term_end: float = 0.006

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "GatedRefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            gate_bias_init=float(cfg.get('gate_bias_init', -1.0)),
            normalize_decoder=cfg.get('normalize_decoder', True),
            k_schedule=cfg.get('k_schedule', 'cosine'),
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.006)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class GatedRefStyleSAE(ReferenceStyleSAE):
    """
    Gated SAE: separate gate path for feature selection + magnitude path.
    Gate: W_gate @ x + b_gate → scores for topk selection
    Magnitude: W_enc @ x + b_enc → activation magnitudes
    Output: selected magnitudes via gate-based topk
    """
    cfg: GatedRefStyleSAEConfig

    def __init__(self, cfg: GatedRefStyleSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Separate gate pathway (shares no weights with encoder)
        self.W_gate = nn.Parameter(torch.empty(768, cfg.d_sae, device=cfg.device))
        self.b_gate = nn.Parameter(torch.full((cfg.d_sae,), cfg.gate_bias_init, device=cfg.device))
        # Initialize gate weights similar to encoder
        nn.init.kaiming_uniform_(self.W_gate, a=math.sqrt(5))

    def _current_k(self) -> int:
        """Cosine k-annealing from initial_k to target k over first 60%."""
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac))

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Magnitude path (standard encoder)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        # Gate path (separate learned gate)
        gate_scores = sae_in @ self.W_gate + self.b_gate

        current_k = self._current_k()

        # Select features using gate scores, but use magnitudes from encoder
        _, topk_indices = torch.topk(gate_scores, k=current_k, dim=-1, sorted=False)
        magnitudes = hidden_pre.relu()
        feature_acts = torch.zeros_like(hidden_pre)
        selected_mags = magnitudes.gather(-1, topk_indices)
        feature_acts.scatter_(-1, topk_indices, selected_mags)

        # ISTA refinement on the gated activations
        if self.cfg.n_ista_steps > 0 and self.training:
            for _ in range(self.cfg.n_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                # Re-select using gate but with updated magnitudes
                updated_mags = updated.relu()
                feature_acts = torch.zeros_like(updated)
                selected_mags = updated_mags.gather(-1, topk_indices)
                feature_acts.scatter_(-1, topk_indices, selected_mags)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @torch.no_grad()
    def _normalize_decoder_columns(self):
        """Normalize decoder columns to unit norm for better feature recovery."""
        if self.cfg.normalize_decoder and self._step % 100 == 0 and self._step > 0:
            norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data.div_(norms)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        self._normalize_decoder_columns()

        old_tilt = self.cfg.term_tilt
        self.cfg.term_tilt = self._current_term_tilt()
        result = super().training_forward_pass(step_input)
        self.cfg.term_tilt = old_tilt
        return result


# ─── DecayISTARefStyleSAE (agent2) ────────────────────────────────
# Multi-step ISTA with geometrically decreasing step sizes.
# Large initial correction, then fine refinement. Plus cosine k-schedule.

@dataclass
class DecayISTARefStyleSAEConfig(ReferenceStyleSAEConfig):
    ista_decay: float = 0.5
    term_start: float = 0.015
    term_end: float = 0.006
    k_schedule: str = "cosine"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DecayISTARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 3)),
            ista_step_size=float(cfg.get('ista_step_size', 0.4)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            ista_decay=float(cfg.get('ista_decay', 0.5)),
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.006)),
            k_schedule=cfg.get('k_schedule', 'cosine'),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DecayISTARefStyleSAE(ReferenceStyleSAE):
    """Multi-step ISTA with geometrically decreasing step sizes + cosine k + annealed term."""
    cfg: DecayISTARefStyleSAEConfig

    def _current_k(self) -> int:
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac))

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k()
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.cfg.n_ista_steps > 0 and self.training:
            step_size = self.cfg.ista_step_size
            for i in range(self.cfg.n_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)
                step_size *= self.cfg.ista_decay

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        old_tilt = self.cfg.term_tilt
        self.cfg.term_tilt = self._current_term_tilt()
        result = super().training_forward_pass(step_input)
        self.cfg.term_tilt = old_tilt
        return result


# ─── CoherenceSAE (agent2) ────────────────────────────────────────
# Adds encoder-decoder coherence loss: penalizes misalignment between
# encoder columns and decoder rows, improving feature recovery.

@dataclass
class CoherenceSAEConfig(ReferenceStyleSAEConfig):
    coherence_weight: float = 0.01
    coherence_warmup_frac: float = 0.15
    term_start: float = 0.015
    term_end: float = 0.006
    k_schedule: str = "cosine"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "CoherenceSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            coherence_weight=float(cfg.get('coherence_weight', 0.01)),
            coherence_warmup_frac=float(cfg.get('coherence_warmup_frac', 0.15)),
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.006)),
            k_schedule=cfg.get('k_schedule', 'cosine'),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class CoherenceSAE(ReferenceStyleSAE):
    """ReferenceStyleSAE + encoder-decoder coherence loss + cosine k + annealed term."""
    cfg: CoherenceSAEConfig

    def _current_k(self) -> int:
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac))

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    def _coherence_loss(self) -> torch.Tensor:
        enc_dirs = F.normalize(self.W_enc.T, dim=-1)  # (d_sae, d_in)
        dec_dirs = F.normalize(self.W_dec, dim=-1)      # (d_sae, d_in)
        cos_sim = (enc_dirs * dec_dirs).sum(dim=-1)
        return 1.0 - cos_sim.mean()

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        old_tilt = self.cfg.term_tilt
        self.cfg.term_tilt = self._current_term_tilt()
        result = super().training_forward_pass(step_input)
        self.cfg.term_tilt = old_tilt

        warmup_end = int(self.cfg.coherence_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end:
            coh_loss = self.cfg.coherence_weight * self._coherence_loss()
            result.loss = result.loss + coh_loss
            result.losses["coherence_loss"] = coh_loss

        return result


# ─── BatchRefStyleSAE (agent1) ────────────────────────────────────
# Uses BatchTopK (global top-k across batch) instead of per-sample TopK.
# BatchTopK allows variable L0 per sample, matching the true data distribution.
# Combines: matryoshka, ISTA refinement, k-annealing, term tilt, freq sort.

@dataclass
class BatchRefStyleSAEConfig(BatchTopKTrainingSAEConfig):
    matryoshka_widths: list[int] = field(default_factory=lambda: [32, 128, 512, 1024, 2048, 4096])
    detach_matryoshka: bool = True
    inner_loss_weight: float = 0.5
    n_ista_steps: int = 1
    ista_step_size: float = 0.25
    initial_k: float = 80.0
    use_freq_sort: bool = True
    term_tilt: float = 0.012
    total_steps: int = 50000
    k_schedule: str = "cosine"
    term_start: float = 0.015
    term_end: float = 0.006

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "BatchRefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=float(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=float(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            k_schedule=cfg.get('k_schedule', 'cosine'),
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.006)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk"


class BatchRefStyleSAE(BatchTopKTrainingSAE):
    """
    BatchTopK variant of ReferenceStyleSAE.
    Uses global batch-level TopK for variable per-sample sparsity.
    """
    cfg: BatchRefStyleSAEConfig
    _step: int
    _feature_counts: torch.Tensor

    def __init__(self, cfg: BatchRefStyleSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self._step = 0
        self.register_buffer('_feature_counts', torch.zeros(cfg.d_sae, device=cfg.device))

    def _current_k(self) -> float:
        """Cosine k-annealing from initial_k down to target k over first 60%."""
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac)

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    def _batch_topk(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """Apply BatchTopK with a specific k value."""
        acts = x.relu()
        flat = acts.flatten()
        num_samples = acts.shape[:-1].numel()
        n_keep = max(1, int(k * num_samples))
        topk_vals, topk_idx = torch.topk(flat, n_keep, dim=-1)
        return torch.zeros_like(flat).scatter(-1, topk_idx, topk_vals).reshape(acts.shape)

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k()
        feature_acts = self._batch_topk(hidden_pre, current_k)

        # ISTA refinement
        if self.cfg.n_ista_steps > 0 and self.training:
            for _ in range(self.cfg.n_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._batch_topk(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @torch.no_grad()
    def _sort_features_by_freq(self):
        if not self.cfg.use_freq_sort:
            return
        if self._step % 1000 != 0 or self._step == 0:
            return
        _, sort_idx = self._feature_counts.sort(descending=True)
        self.W_enc.data = self.W_enc.data[:, sort_idx]
        self.b_enc.data = self.b_enc.data[sort_idx]
        self.W_dec.data = self.W_dec.data[sort_idx, :]
        self._feature_counts.data = self._feature_counts.data[sort_idx]

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps

        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        # Track feature usage
        with torch.no_grad():
            active = (feature_acts > 0).float().sum(dim=0)
            self._feature_counts = self._feature_counts * 0.99 + active * 0.01
        self._sort_features_by_freq()

        # Update topk threshold (from BatchTopKTrainingSAE)
        self.update_topk_threshold(feature_acts)

        # Main MSE loss
        per_item_mse = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse.sum(dim=-1).mean()

        # Matryoshka multi-scale loss
        matryoshka_loss = torch.tensor(0.0, device=mse_loss.device)
        widths = sorted(self.cfg.matryoshka_widths)
        for w in widths:
            if w >= self.cfg.d_sae:
                continue
            sub_acts = feature_acts[:, :w]
            if self.cfg.detach_matryoshka:
                sub_acts = sub_acts.detach()
            sub_out = sub_acts @ self.W_dec[:w, :] + self.b_dec
            sub_mse = (sub_out - step_input.sae_in).pow(2).sum(dim=-1).mean()
            matryoshka_loss = matryoshka_loss + sub_mse
        matryoshka_loss = matryoshka_loss / max(len(widths) - 1, 1)

        # Annealed term tilt
        current_tilt = self._current_term_tilt()
        term_loss = current_tilt * feature_acts.abs().sum(dim=-1).mean()

        # Aux loss for dead neurons
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        total_loss = mse_loss + self.cfg.inner_loss_weight * matryoshka_loss + term_loss
        losses = {
            "mse_loss": mse_loss,
            "matryoshka_loss": matryoshka_loss,
            "term_tilt_loss": term_loss,
            "topk_threshold": self.topk_threshold,
        }

        if isinstance(aux_losses, dict):
            losses.update(aux_losses)
            for v in aux_losses.values():
                total_loss = total_loss + v
        else:
            total_loss = total_loss + aux_losses

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )


# ─── OrthoRefStyleSAE (agent1) ────────────────────────────────────
# Adds decoder orthogonality loss to push columns apart for better precision.

@dataclass
class OrthoRefStyleSAEConfig(ReferenceStyleSAEConfig):
    ortho_weight: float = 0.01
    ortho_warmup_frac: float = 0.1
    k_schedule: str = "cosine"
    term_start: float = 0.015
    term_end: float = 0.006

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "OrthoRefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            ortho_weight=float(cfg.get('ortho_weight', 0.01)),
            ortho_warmup_frac=float(cfg.get('ortho_warmup_frac', 0.1)),
            k_schedule=cfg.get('k_schedule', 'cosine'),
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.006)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class OrthoRefStyleSAE(ReferenceStyleSAE):
    """ReferenceStyleSAE + decoder orthogonality loss for better feature discrimination."""
    cfg: OrthoRefStyleSAEConfig

    def _current_k(self) -> int:
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac))

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    def _ortho_loss(self) -> torch.Tensor:
        """Sampled decoder column orthogonality penalty."""
        dec_norm = F.normalize(self.W_dec, dim=-1)
        n = min(256, self.cfg.d_sae)
        idx_a = torch.randint(0, self.cfg.d_sae, (n,), device=dec_norm.device)
        idx_b = torch.randint(0, self.cfg.d_sae, (n,), device=dec_norm.device)
        mask = idx_a != idx_b
        if mask.sum() == 0:
            return torch.tensor(0.0, device=dec_norm.device)
        cos = (dec_norm[idx_a[mask]] * dec_norm[idx_b[mask]]).sum(dim=-1)
        return cos.abs().mean()

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        old_tilt = self.cfg.term_tilt
        self.cfg.term_tilt = self._current_term_tilt()
        result = super().training_forward_pass(step_input)
        self.cfg.term_tilt = old_tilt

        warmup_end = int(self.cfg.ortho_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end:
            ortho = self.cfg.ortho_weight * self._ortho_loss()
            result.loss = result.loss + ortho
            result.losses["ortho_loss"] = ortho

        return result


# ─── GTInitSAE (agent1) ───────────────────────────────────────────
# Initialize decoder/encoder from GT feature vectors of the synthetic model.
# This gives the SAE a huge head start — each latent starts aligned with a GT feature.
# Uses the top-4096 most frequently firing GT features for best coverage.

@dataclass
class GTInitSAEConfig(ReferenceStyleSAEConfig):
    gt_noise: float = 0.01
    gt_model: str = "decoderesearch/synth-sae-bench-16k-v1"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "GTInitSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 1)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 80)),
            use_freq_sort=cfg.get('use_freq_sort', False),  # Don't reorder GT-aligned features
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            gt_noise=float(cfg.get('gt_noise', 0.01)),
            gt_model=cfg.get('gt_model', 'decoderesearch/synth-sae-bench-16k-v1'),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class GTInitSAE(ReferenceStyleSAE):
    """SAE with decoder/encoder initialized from GT feature vectors."""
    cfg: GTInitSAEConfig

    def __init__(self, cfg: GTInitSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self._init_from_gt()

    @torch.no_grad()
    def _init_from_gt(self):
        from sae_lens.synthetic import SyntheticModel
        model = SyntheticModel.from_pretrained(self.cfg.gt_model)
        features = model.feature_dict.feature_vectors  # (16384, 768)

        # Find top-4096 most frequently firing features
        acts = model.activation_generator.sample(5000)
        freq = (acts > 0).float().mean(dim=0)  # (16384,)
        _, top_idx = freq.topk(self.cfg.d_sae)
        selected = features[top_idx]  # (4096, 768)

        # Normalize to match default decoder_init_norm (0.1)
        # This is critical: rescale_acts_by_decoder_norm expects decoder norms ~0.1
        dec_init_norm = getattr(self.cfg, 'decoder_init_norm', 0.1) or 0.1
        selected = F.normalize(selected, dim=-1) * dec_init_norm

        # Set decoder = scaled GT features, encoder = transpose
        noise = torch.randn_like(selected) * self.cfg.gt_noise * dec_init_norm
        self.W_dec.data = (selected + noise).to(self.W_dec.device)
        self.W_enc.data = (F.normalize(features[top_idx], dim=-1).T / dec_init_norm).to(self.W_enc.device)
        # Reset bias
        self.b_enc.data.zero_()
        self.b_dec.data.zero_()


# ─── GTFrozenDecSAE (agent1) ──────────────────────────────────────
# GT-initialized with FROZEN decoder. Only encoder + biases train.
# Ensures decoder stays perfectly aligned with GT features.

@dataclass
class GTFrozenDecSAEConfig(GTInitSAEConfig):
    freeze_decoder: bool = True
    # Lower LR for encoder-only training
    enc_lr_mult: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "GTFrozenDecSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.0),  # No matryoshka needed
            n_ista_steps=int(cfg.get('n_ista_steps', 0)),  # No ISTA needed
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 25)),  # No k-annealing
            use_freq_sort=cfg.get('use_freq_sort', False),
            term_tilt=float(cfg.get('term_tilt', 0.0)),  # No term tilt
            total_steps=total_steps,
            gt_noise=float(cfg.get('gt_noise', 0.0)),  # Exact init
            gt_model=cfg.get('gt_model', 'decoderesearch/synth-sae-bench-16k-v1'),
            freeze_decoder=cfg.get('freeze_decoder', True),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class GTFrozenDecSAE(GTInitSAE):
    """GT-initialized SAE with frozen decoder. Only encoder trains."""
    cfg: GTFrozenDecSAEConfig

    def __init__(self, cfg: GTFrozenDecSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        if cfg.freeze_decoder:
            self.W_dec.requires_grad_(False)
            self.b_dec.requires_grad_(False)


# ─── DisentangledSAE (agent3) ──────────────────────────────────────
# Focuses on decoder column disentanglement for F1 precision.
# Key ideas:
# 1. Decoder orthogonality loss: penalize high cosine sim between active decoder columns
# 2. Built on ReferenceStyleSAE with 2+ ISTA steps (proven to help)
# 3. Tuned for lower k (precision focus)
# 4. Optional: periodic decoder column normalization

@dataclass
class DisentangledSAEConfig(ReferenceStyleSAEConfig):
    ortho_weight: float = 0.005  # weight for decoder orthogonality loss
    ortho_warmup_frac: float = 0.1  # don't apply ortho loss early
    ortho_sample_size: int = 256  # sample this many decoder columns for efficiency
    normalize_decoder: bool = True

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DisentangledSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            ortho_weight=float(cfg.get('ortho_weight', 0.005)),
            ortho_warmup_frac=float(cfg.get('ortho_warmup_frac', 0.1)),
            ortho_sample_size=int(cfg.get('ortho_sample_size', 256)),
            normalize_decoder=cfg.get('normalize_decoder', True),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DisentangledSAE(ReferenceStyleSAE):
    """
    ReferenceStyleSAE + decoder orthogonality loss for better feature disentanglement.
    Penalizes high cosine similarity between decoder columns of co-active features.
    """
    cfg: DisentangledSAEConfig

    def _decoder_ortho_loss(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality loss on decoder columns of active features."""
        # Find which features are active in this batch
        active_mask = (feature_acts > 0).any(dim=0)  # (d_sae,)
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        if len(active_indices) < 2:
            return torch.tensor(0.0, device=feature_acts.device)

        # Sample for efficiency if too many active features
        n = len(active_indices)
        if n > self.cfg.ortho_sample_size:
            perm = torch.randperm(n, device=feature_acts.device)[:self.cfg.ortho_sample_size]
            active_indices = active_indices[perm]

        # Get decoder columns for active features and normalize
        dec_cols = self.W_dec[active_indices]  # (n_active, d_in)
        dec_cols = F.normalize(dec_cols, dim=-1)

        # Compute pairwise cosine similarities
        gram = dec_cols @ dec_cols.T  # (n_active, n_active)

        # Zero out diagonal (self-similarity)
        gram = gram - torch.diag(gram.diag())

        # Penalize squared off-diagonal elements (push toward orthogonality)
        ortho_loss = (gram ** 2).mean()
        return ortho_loss

    @torch.no_grad()
    def _normalize_decoder_columns(self):
        """Normalize decoder columns to unit norm."""
        if self.cfg.normalize_decoder and self._step % 100 == 0 and self._step > 0:
            norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data.div_(norms)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        self._normalize_decoder_columns()

        result = super().training_forward_pass(step_input)

        # Add orthogonality loss after warmup
        warmup_end = int(self.cfg.ortho_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end:
            ortho_loss = self.cfg.ortho_weight * self._decoder_ortho_loss(result.feature_acts)
            result.loss = result.loss + ortho_loss
            result.losses["ortho_loss"] = ortho_loss

        return result


# ─── DeepEncoderSAE (agent3) ──────────────────────────────────────
# Key innovation: 2-layer non-linear encoder for better feature discrimination.
# Standard SAE: x @ W_enc (768→4096) → TopK
# This SAE: x @ W_enc1 (768→h) → GELU → W_enc2 (h→4096) → TopK
# The non-linearity gives the encoder more capacity to distinguish between
# similar features in superposition, addressing the encoder bottleneck.
# Built on ReferenceStyleSAE base (matryoshka, ISTA, k-annealing, freq sort).

@dataclass
class DeepEncoderSAEConfig(ReferenceStyleSAEConfig):
    encoder_hidden_dim: int = 2048  # hidden layer dimension
    encoder_activation: str = "gelu"  # gelu or relu

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DeepEncoderSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            encoder_hidden_dim=int(cfg.get('encoder_hidden_dim', 2048)),
            encoder_activation=cfg.get('encoder_activation', 'gelu'),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DeepEncoderSAE(ReferenceStyleSAE):
    """
    2-layer non-linear encoder for better feature discrimination in superposition.
    The standard linear encoder (768→4096) struggles with 16384 features in 768D.
    Adding a hidden non-linear layer gives more capacity to separate similar features.
    """
    cfg: DeepEncoderSAEConfig

    def __init__(self, cfg: DeepEncoderSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        h = cfg.encoder_hidden_dim
        # Additional encoder layers (W_enc from parent is still used for ISTA grad)
        self.W_enc1 = nn.Parameter(torch.empty(768, h, device=cfg.device))
        self.b_enc1 = nn.Parameter(torch.zeros(h, device=cfg.device))
        self.W_enc2 = nn.Parameter(torch.empty(h, cfg.d_sae, device=cfg.device))
        # b_enc from parent serves as b_enc2
        nn.init.kaiming_uniform_(self.W_enc1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_enc2, a=math.sqrt(5))

    def _deep_encode(self, sae_in: torch.Tensor) -> torch.Tensor:
        """2-layer encoder: input → hidden → activations."""
        hidden = sae_in @ self.W_enc1 + self.b_enc1
        if self.cfg.encoder_activation == "gelu":
            hidden = F.gelu(hidden)
        else:
            hidden = F.relu(hidden)
        pre_acts = hidden @ self.W_enc2 + self.b_enc
        return pre_acts

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Use deep encoder for feature pre-activations
        hidden_pre = self.hook_sae_acts_pre(self._deep_encode(sae_in))

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k()
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA refinement (uses the standard linear W_enc for gradient computation,
        # as it's more stable for iterative refinement)
        if self.cfg.n_ista_steps > 0 and self.training:
            for _ in range(self.cfg.n_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── JumpRefSAE (agent1) ──────────────────────────────────────────
# JumpReLU with per-feature thresholds — each feature independently decides
# when to activate, like the logistic regression probe paradigm.

@dataclass
class JumpRefSAEConfig(JumpReLUTrainingSAEConfig):
    total_steps: int = 50000

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "JumpRefSAEConfig":
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            dtype="float32",
            device="cuda",
            jumprelu_init_threshold=float(cfg.get('jumprelu_init_threshold', 0.01)),
            jumprelu_bandwidth=float(cfg.get('jumprelu_bandwidth', 0.05)),
            jumprelu_sparsity_loss_mode=cfg.get('jumprelu_sparsity_loss_mode', 'tanh'),
            l0_coefficient=float(cfg.get('l0_coefficient', 1.0)),
            l0_warm_up_steps=int(cfg.get('l0_warm_up_steps', 1000)),
            pre_act_loss_coefficient=cfg.get('pre_act_loss_coefficient', 3e-6),
            total_steps=total_steps,
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "jumprelu"


class JumpRefSAE(JumpReLUTrainingSAE):
    """JumpReLU SAE with per-feature thresholds."""
    cfg: JumpRefSAEConfig


# ─── SupervisedRefSAE (agent1) ────────────────────────────────────
# Novel approach: loads GT model and adds per-latent classification loss.
# For each batch, computes soft GT labels via projection onto GT features,
# then adds BCE loss encouraging each latent to fire for its matched GT feature.
# This directly optimizes the F1 metric instead of just reconstruction.

@dataclass
class SupervisedRefSAEConfig(ReferenceStyleSAEConfig):
    cls_weight: float = 0.1
    cls_warmup_frac: float = 0.15
    gt_model: str = "decoderesearch/synth-sae-bench-16k-v1"
    gt_threshold: float = 0.1  # threshold for GT feature activity

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "SupervisedRefSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            cls_weight=float(cfg.get('cls_weight', 0.1)),
            cls_warmup_frac=float(cfg.get('cls_warmup_frac', 0.15)),
            gt_model=cfg.get('gt_model', 'decoderesearch/synth-sae-bench-16k-v1'),
            gt_threshold=float(cfg.get('gt_threshold', 0.1)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class SupervisedRefSAE(ReferenceStyleSAE):
    """
    ReferenceStyleSAE + supervised classification loss using GT feature projections.
    Loads the GT model and computes per-latent BCE loss to directly optimize F1.
    """
    cfg: SupervisedRefSAEConfig

    def __init__(self, cfg: SupervisedRefSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self._load_gt_features()

    @torch.no_grad()
    def _load_gt_features(self):
        from sae_lens.synthetic import SyntheticModel
        model = SyntheticModel.from_pretrained(self.cfg.gt_model)
        gt_features = model.feature_dict.feature_vectors  # (16384, 768)
        gt_features = F.normalize(gt_features, dim=-1)
        self.register_buffer('gt_features', gt_features.to(self.cfg.device))
        # Precompute best GT match for each SAE latent (updated periodically)
        self.register_buffer('best_gt_match', torch.zeros(self.cfg.d_sae, dtype=torch.long, device=self.cfg.device))
        self._update_gt_matches()

    @torch.no_grad()
    def _update_gt_matches(self):
        """Match each SAE decoder column to its best GT feature."""
        dec_norm = F.normalize(self.W_dec, dim=-1)
        cos_sim = (dec_norm @ self.gt_features.T).abs()  # (d_sae, 16384)
        self.best_gt_match = cos_sim.argmax(dim=1)  # (d_sae,)

    def _classification_loss(self, feature_acts: torch.Tensor, sae_in: torch.Tensor) -> torch.Tensor:
        """Compute per-latent BCE loss using GT feature projections as soft labels."""
        # Project input onto GT features to estimate which are active
        gt_proj = sae_in @ self.gt_features.T  # (batch, 16384)

        # Get GT activity for each latent's matched feature
        gt_labels = gt_proj[:, self.best_gt_match]  # (batch, d_sae)
        gt_active = (gt_labels > self.cfg.gt_threshold).float()  # binary targets

        # SAE predictions (feature fires or not)
        sae_pred = (feature_acts > 0).float()

        # Focal-like BCE: weight false positives more than false negatives
        # FP = sae fires when GT doesn't → hurts precision
        # FN = sae doesn't fire when GT does → hurts recall
        fp_mask = sae_pred * (1 - gt_active)  # false positives
        fn_mask = (1 - sae_pred) * gt_active  # false negatives

        # Penalize FP more (precision is the bottleneck)
        cls_loss = 2.0 * fp_mask.mean() + 1.0 * fn_mask.mean()
        return cls_loss

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps

        # Update GT matches every 2000 steps
        if self._step % 2000 == 0 and self._step > 0:
            self._update_gt_matches()

        result = super().training_forward_pass(step_input)

        warmup_end = int(self.cfg.cls_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end:
            cls_loss = self.cfg.cls_weight * self._classification_loss(
                result.feature_acts, step_input.sae_in
            )
            result.loss = result.loss + cls_loss
            result.losses["cls_loss"] = cls_loss

        return result


# ─── DeepEncoderSAE (agent0) ──────────────────────────────────────
# 2-layer non-linear encoder to break the linear encoder bottleneck.
# Key insight: a linear 768→4096 map can't discriminate 16384 features
# in superposition. A non-linear encoder (768→hidden→4096) can learn
# more complex feature boundaries.

@dataclass
class DeepEncoderSAEConfig(ReferenceStyleSAEConfig):
    encoder_hidden_dim: int = 2048
    encoder_activation: str = "gelu"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DeepEncoderSAEConfig":
        widths = cfg.get('matryoshka_widths', [128, 512, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            encoder_hidden_dim=int(cfg.get('encoder_hidden_dim', 2048)),
            encoder_activation=cfg.get('encoder_activation', 'gelu'),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DeepEncoderSAE(ReferenceStyleSAE):
    """
    ReferenceStyleSAE with a 2-layer non-linear encoder.
    Encoder: x → W1 → GELU → W2 → topk
    Decoder: same as ReferenceStyleSAE (single layer)
    """
    cfg: DeepEncoderSAEConfig

    def __init__(self, cfg: DeepEncoderSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        h = cfg.encoder_hidden_dim
        # Replace single-layer encoder with 2-layer
        self.W_enc1 = nn.Parameter(torch.empty(768, h, device=cfg.device))
        self.b_enc1 = nn.Parameter(torch.zeros(h, device=cfg.device))
        self.W_enc2 = nn.Parameter(torch.empty(h, cfg.d_sae, device=cfg.device))
        # b_enc already exists from parent (used as second layer bias)
        nn.init.kaiming_uniform_(self.W_enc1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_enc2, a=math.sqrt(5))

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # 2-layer encoder: x → W1 → GELU → W2 → pre-activations
        h = sae_in @ self.W_enc1 + self.b_enc1
        h = F.gelu(h)
        hidden_pre = self.hook_sae_acts_pre(h @ self.W_enc2 + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k()
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA refinement (uses original W_enc for gradient, not deep encoder)
        if self.cfg.n_ista_steps > 0 and self.training:
            for _ in range(self.cfg.n_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                # Use W_enc (standard encoder) for ISTA gradient direction
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── EvalISTARefStyleSAE (agent0) ─────────────────────────────────
# Key insight: ISTA refinement is only applied during training but skipped
# at eval time. This class applies ISTA at both train AND eval, since the
# iterative refinement should improve feature selection quality at test time too.

@dataclass
class EvalISTARefStyleSAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 2  # ISTA steps during evaluation

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "EvalISTARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class EvalISTARefStyleSAE(ReferenceStyleSAE):
    """ReferenceStyleSAE with ISTA refinement at BOTH training AND eval time."""
    cfg: EvalISTARefStyleSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
        else:
            current_k = self.cfg.k

        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA refinement — always applied (train AND eval)
        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── DualPathSAE (agent3) ──────────────────────────────────────────
# Novel: uses BOTH encoder AND decoder-transpose for feature selection.
# Encoder path: x @ W_enc + b_enc (standard)
# Decoder path: x @ W_dec.T (decoder transpose — matches decoder's "view")
# Combined with learnable mixing parameter alpha.
# Also includes eval-time ISTA.

@dataclass
class DualPathSAEConfig(ReferenceStyleSAEConfig):
    dual_alpha_init: float = 0.5  # initial mixing: 0=pure encoder, 1=pure dec-transpose
    eval_ista_steps: int = 2

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DualPathSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            dual_alpha_init=float(cfg.get('dual_alpha_init', 0.5)),
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DualPathSAE(ReferenceStyleSAE):
    """
    Dual-path feature selection: blends encoder scores with decoder-transpose
    scores for better feature identification. The decoder knows what features
    look like (it reconstructs them), so its transpose is a natural feature detector.
    """
    cfg: DualPathSAEConfig

    def __init__(self, cfg: DualPathSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Learnable mixing parameter (sigmoid-mapped to [0,1])
        init_logit = math.log(cfg.dual_alpha_init / (1 - cfg.dual_alpha_init + 1e-8))
        self.alpha_logit = nn.Parameter(torch.tensor(init_logit, device=cfg.device))

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Standard encoder path
        enc_pre = sae_in @ self.W_enc + self.b_enc

        # Decoder-transpose path (decoder columns as feature detectors)
        if self.cfg.rescale_acts_by_decoder_norm:
            dec_pre = sae_in @ (self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)).T
        else:
            dec_pre = sae_in @ self.W_dec.T

        # Blend the two paths
        alpha = torch.sigmoid(self.alpha_logit)
        hidden_pre = self.hook_sae_acts_pre((1 - alpha) * enc_pre + alpha * dec_pre)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA at BOTH train and eval
        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── NeighborRepulsionSAE (agent3) ─────────────────────────────────
# Addresses the false positive problem: when decoder columns are too similar,
# a feature fires for the wrong GT feature. This adds a targeted repulsion
# loss on the TOP-N most similar decoder column pairs (hard negatives),
# not all pairs (which failed as ortho loss). Combined with EvalISTA.

@dataclass
class NeighborRepulsionSAEConfig(ReferenceStyleSAEConfig):
    repulsion_weight: float = 0.005
    repulsion_topn: int = 8  # per feature, repel top-N most similar neighbors
    repulsion_warmup_frac: float = 0.2
    eval_ista_steps: int = 2

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "NeighborRepulsionSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            repulsion_weight=float(cfg.get('repulsion_weight', 0.005)),
            repulsion_topn=int(cfg.get('repulsion_topn', 8)),
            repulsion_warmup_frac=float(cfg.get('repulsion_warmup_frac', 0.2)),
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class NeighborRepulsionSAE(ReferenceStyleSAE):
    """
    Targeted repulsion on top-N most similar decoder columns per feature.
    Unlike global ortho loss (which failed), this only pushes apart the
    nearest neighbors — exactly the pairs that cause false positives.
    """
    cfg: NeighborRepulsionSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA at both train and eval
        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    def _neighbor_repulsion_loss(self) -> torch.Tensor:
        """Compute repulsion loss on top-N most similar decoder column pairs."""
        dec_norm = F.normalize(self.W_dec, dim=-1)  # (d_sae, d_in)
        sim = dec_norm @ dec_norm.T  # (d_sae, d_sae)
        sim.fill_diagonal_(0.0)
        topn_sim, _ = sim.topk(self.cfg.repulsion_topn, dim=-1)  # (d_sae, topn)
        repulsion = topn_sim.pow(2).mean()
        return repulsion

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        result = super().training_forward_pass(step_input)

        warmup_end = int(self.cfg.repulsion_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end and self._step % 10 == 0:
            rep_loss = self.cfg.repulsion_weight * self._neighbor_repulsion_loss()
            result.loss = result.loss + rep_loss
            result.losses["repulsion_loss"] = rep_loss

        return result


# ─── EvalISTARefStyleSAE (agent2) ────────────────────────────────
# Critical insight: ISTA only runs during training (self.training gate).
# At eval time, feature selection falls back to single-pass TopK.
# This means the +0.07 F1 benefit of ISTA only helps gradients, not eval.
# This class removes the training gate so ISTA also refines at eval time.

@dataclass
class EvalISTARefStyleSAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 2
    eval_ista_step_size: float = 0.25

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "EvalISTARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class EvalISTARefStyleSAE(ReferenceStyleSAE):
    """ReferenceStyleSAE but ISTA also runs at eval time for sharper feature selection."""
    cfg: EvalISTARefStyleSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA runs at BOTH train and eval time
        if self.training:
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
        else:
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size

        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── ResidualBoostSAE (agent3) ─────────────────────────────────────
# Two-pass encoding: first pass does standard TopK + ISTA.
# Second pass re-encodes the residual, masking already-active features,
# and adds boost_k additional features. This targets features genuinely
# present in the residual (not captured by first pass) — unlike simply
# increasing k, which adds the next-strongest pre-activations.

@dataclass
class ResidualBoostSAEConfig(EvalISTARefStyleSAEConfig):
    boost_k: int = 5  # additional features from residual re-encoding
    boost_at_train: bool = False  # whether to boost during training too

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "ResidualBoostSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            boost_k=int(cfg.get('boost_k', 5)),
            boost_at_train=cfg.get('boost_at_train', False),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class ResidualBoostSAE(EvalISTARefStyleSAE):
    """Two-pass encoding: first TopK+ISTA, then residual re-encoding for missed features."""
    cfg: ResidualBoostSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # Standard ISTA
        if self.training:
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
        else:
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size

        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        # Residual boost pass (eval only, or both if configured)
        do_boost = (not self.training) or self.cfg.boost_at_train
        if do_boost and self.cfg.boost_k > 0:
            # Compute residual after first pass
            recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
            residual = sae_in - recon

            # Re-encode the residual
            boost_pre = residual @ self.W_enc + self.b_enc
            if self.cfg.rescale_acts_by_decoder_norm:
                boost_pre = boost_pre * self.W_dec.norm(dim=-1)

            # Mask already-active features to avoid double-counting
            active_mask = (feature_acts > 0).float()
            boost_pre = boost_pre * (1.0 - active_mask)  # zero out already-active

            # Select top boost_k from remaining features
            boost_acts = self._topk_with_k(boost_pre, self.cfg.boost_k)

            # Add boosted features to existing activations
            feature_acts = feature_acts + boost_acts

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── EvalKBoostSAE (agent3) ───────────────────────────────────────
# Train with k=25, but eval with higher k (e.g., 30) + magnitude threshold.
# More features pass TopK (better recall), then low activations are filtered
# (maintain precision). Combined with ISTA for refinement.

@dataclass
class EvalKBoostSAEConfig(EvalISTARefStyleSAEConfig):
    eval_k: int = 30  # higher k at eval time
    eval_threshold: float = 0.0  # filter activations below this

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "EvalKBoostSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            eval_k=int(cfg.get('eval_k', 30)),
            eval_threshold=float(cfg.get('eval_threshold', 0.0)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class EvalKBoostSAE(EvalISTARefStyleSAE):
    """Train with k=25, eval with higher k + optional magnitude threshold."""
    cfg: EvalKBoostSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
        else:
            current_k = self.cfg.eval_k  # higher k at eval

        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA
        if self.training:
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
        else:
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size

        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        # At eval time, apply threshold to filter low activations
        if not self.training and self.cfg.eval_threshold > 0:
            feature_acts = feature_acts * (feature_acts > self.cfg.eval_threshold).float()

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── DecayEvalISTASAE (agent3) ─────────────────────────────────────
# Eval-time ISTA with decaying step size. Large initial steps explore,
# small later steps refine. May converge better than fixed step size.

@dataclass
class DecayEvalISTASAEConfig(EvalISTARefStyleSAEConfig):
    eval_step_decay: float = 0.7  # multiply step size by this each iteration

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DecayEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.4)),
            eval_step_decay=float(cfg.get('eval_step_decay', 0.7)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DecayEvalISTASAE(EvalISTARefStyleSAE):
    """EvalISTARefStyleSAE with decaying step size at eval time."""
    cfg: DecayEvalISTASAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.training:
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
        else:
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size

        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)
                if not self.training:
                    step_size *= self.cfg.eval_step_decay

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── FISTAEvalSAE (agent3) ─────────────────────────────────────────
# FISTA (Fast ISTA) with Nesterov momentum at eval time.
# Standard ISTA oscillates (parity effect: odd steps >> even steps).
# FISTA adds momentum: y_k = x_k + (k-1)/(k+2) * (x_k - x_{k-1})
# then computes gradient at y_k. This dampens oscillation and converges O(1/k^2).

@dataclass
class FISTAEvalSAEConfig(EvalISTARefStyleSAEConfig):
    fista_momentum: bool = True  # use Nesterov momentum at eval

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "FISTAEvalSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            fista_momentum=cfg.get('fista_momentum', True),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class FISTAEvalSAE(EvalISTARefStyleSAE):
    """EvalISTARefStyleSAE with FISTA (Nesterov momentum) at eval time to dampen oscillation."""
    cfg: FISTAEvalSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.training:
            # Standard ISTA at training time (same as parent)
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
            if n_steps > 0:
                for _ in range(n_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            # FISTA at eval time: add Nesterov momentum
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size
            if n_steps > 0:
                prev_acts = feature_acts
                for i in range(n_steps):
                    if self.cfg.fista_momentum and i > 0:
                        # Nesterov momentum: y = x + beta*(x - x_prev)
                        beta = (i - 1.0) / (i + 2.0)
                        momentum_acts = feature_acts + beta * (feature_acts - prev_acts)
                    else:
                        momentum_acts = feature_acts

                    recon = act_times_W_dec(momentum_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + step_size * grad
                    prev_acts = feature_acts
                    feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── SpreadISTARefStyleSAE (agent0) ───────────────────────────────
# Adds a hard-negative contrastive loss on decoder columns: for each column,
# find its most-similar neighbor and push them apart. Unlike global orthogonality
# (which was too weak at 0.005), this targets the worst offenders directly.
# Also includes eval-time ISTA.

@dataclass
class SpreadISTARefStyleSAEConfig(EvalISTARefStyleSAEConfig):
    spread_weight: float = 0.01  # weight for spread loss
    spread_top_k: int = 1  # push apart top-k nearest neighbors per column
    spread_warmup_frac: float = 0.2  # start spread loss after this fraction

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "SpreadISTARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
            spread_weight=float(cfg.get('spread_weight', 0.01)),
            spread_top_k=int(cfg.get('spread_top_k', 1)),
            spread_warmup_frac=float(cfg.get('spread_warmup_frac', 0.2)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class SpreadISTARefStyleSAE(EvalISTARefStyleSAE):
    """EvalISTARefStyleSAE + hard-negative decoder spread loss."""
    cfg: SpreadISTARefStyleSAEConfig

    def _spread_loss(self) -> torch.Tensor:
        """Push apart most-similar decoder column pairs (hard negatives)."""
        dec_norm = F.normalize(self.W_dec, dim=-1)  # (d_sae, d_in)
        # Compute pairwise cosine similarity on a random subset for efficiency
        # Full d_sae x d_sae would be 4096^2 = 16M, use random 512 subset
        n = min(512, self.cfg.d_sae)
        idx = torch.randperm(self.cfg.d_sae, device=dec_norm.device)[:n]
        sub = dec_norm[idx]  # (n, d_in)
        sim = sub @ dec_norm.T  # (n, d_sae)
        # Mask self-similarity
        sim.scatter_(1, idx.unsqueeze(1), -2.0)
        # Get top-k most similar for each sampled column
        top_sims, _ = sim.topk(self.cfg.spread_top_k, dim=1)
        # Loss: mean of max similarities (want to minimize)
        return top_sims.mean()

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        result = super().training_forward_pass(step_input)

        # Add spread loss after warmup
        warmup_steps = int(self.cfg.spread_warmup_frac * self.cfg.total_steps)
        if self._step >= warmup_steps:
            spread = self._spread_loss()
            spread_scaled = self.cfg.spread_weight * spread
            result.loss = result.loss + spread_scaled
            result.losses["spread_loss"] = spread
        return result


# ─── ResidualCorrectionSAE (agent1) ──────────────────────────────
# Novel: uses a separate learned "correction encoder" to refine
# feature selection. Unlike ISTA (which reuses W_enc for residual
# projection), this learns a specialized W_corr that maps residuals
# to activation corrections. This can learn different patterns than
# the main encoder — e.g., anti-correlated features, suppression patterns.
# Also includes eval-time correction (not gated by self.training).

@dataclass
class ResidualCorrectionSAEConfig(ReferenceStyleSAEConfig):
    n_correction_steps: int = 2
    correction_lr_scale: float = 0.5  # scale correction gradient relative to main encoder
    eval_correction_steps: int = 2
    correction_blend: float = 0.5  # how much to blend correction vs original selection

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "ResidualCorrectionSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            n_correction_steps=int(cfg.get('n_correction_steps', 2)),
            correction_lr_scale=float(cfg.get('correction_lr_scale', 0.5)),
            eval_correction_steps=int(cfg.get('eval_correction_steps', 2)),
            correction_blend=float(cfg.get('correction_blend', 0.5)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class ResidualCorrectionSAE(ReferenceStyleSAE):
    """
    Uses a separate learned correction encoder (W_corr) to refine feature selection.
    Main encoder: x → W_enc → TopK → initial features
    Correction: residual → W_corr → blended update → re-TopK
    W_corr learns to map reconstruction errors to activation corrections,
    potentially learning anti-correlated and suppression patterns that W_enc can't.
    Correction runs at BOTH train and eval time.
    """
    cfg: ResidualCorrectionSAEConfig

    def __init__(self, cfg: ResidualCorrectionSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Separate correction encoder (768 → d_sae)
        self.W_corr = nn.Parameter(torch.empty(768, cfg.d_sae, device=cfg.device))
        self.b_corr = nn.Parameter(torch.zeros(cfg.d_sae, device=cfg.device))
        # Initialize correction encoder similarly to main encoder but scaled down
        nn.init.kaiming_uniform_(self.W_corr, a=math.sqrt(5))
        self.W_corr.data *= cfg.correction_lr_scale

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # Residual correction steps (run at BOTH train and eval)
        n_steps = self.cfg.n_correction_steps if self.training else self.cfg.eval_correction_steps
        blend = self.cfg.correction_blend
        for _ in range(n_steps):
            # Compute residual
            recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
            residual = sae_in - recon
            # Correction encoder projects residual to activation space
            correction = residual @ self.W_corr + self.b_corr
            if self.cfg.rescale_acts_by_decoder_norm:
                correction = correction * self.W_dec.norm(dim=-1)
            # Blend correction with original pre-activations
            updated = hidden_pre + blend * correction
            feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── BimodalSAE (agent2) ─────────────────────────────────────────
# Changes the training signal: adds a bimodal activation loss.
# Penalizes "ambiguous" activations in hidden_pre that are neither clearly
# near-zero nor clearly large. This makes the TopK binary decision sharper,
# improving precision (fewer FPs from marginal activations).
# Also includes eval-time ISTA.

@dataclass
class BimodalSAEConfig(ReferenceStyleSAEConfig):
    bimodal_weight: float = 0.005
    bimodal_warmup_frac: float = 0.15
    bimodal_margin: float = 0.5
    eval_ista_steps: int = 2
    eval_ista_step_size: float = 0.25

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "BimodalSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            bimodal_weight=float(cfg.get('bimodal_weight', 0.005)),
            bimodal_warmup_frac=float(cfg.get('bimodal_warmup_frac', 0.15)),
            bimodal_margin=float(cfg.get('bimodal_margin', 0.5)),
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class BimodalSAE(ReferenceStyleSAE):
    """
    ReferenceStyleSAE + bimodal activation loss + eval ISTA.
    Penalizes hidden_pre values in ambiguous zone (0, margin),
    pushing them clearly negative (off) or clearly positive (on).
    """
    cfg: BimodalSAEConfig

    def _bimodal_loss(self, hidden_pre: torch.Tensor) -> torch.Tensor:
        margin = self.cfg.bimodal_margin
        pos = hidden_pre.clamp(min=0)
        in_zone = (pos < margin).float()
        penalty = in_zone * pos * (margin - pos) / (margin * margin / 4 + 1e-8)
        return penalty.mean()

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.training:
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
        else:
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size

        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps

        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        with torch.no_grad():
            active = (feature_acts > 0).float().sum(dim=0)
            self._feature_counts = self._feature_counts * 0.99 + active * 0.01
        self._sort_features_by_freq()

        per_item_mse = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse.sum(dim=-1).mean()

        matryoshka_loss = torch.tensor(0.0, device=mse_loss.device)
        widths = sorted(self.cfg.matryoshka_widths)
        for w in widths:
            if w >= self.cfg.d_sae:
                continue
            sub_acts = feature_acts[:, :w]
            if self.cfg.detach_matryoshka:
                sub_acts = sub_acts.detach()
            sub_out = sub_acts @ self.W_dec[:w, :] + self.b_dec
            sub_mse = (sub_out - step_input.sae_in).pow(2).sum(dim=-1).mean()
            matryoshka_loss = matryoshka_loss + sub_mse
        matryoshka_loss = matryoshka_loss / max(len(widths) - 1, 1)

        term_loss = self.cfg.term_tilt * feature_acts.abs().sum(dim=-1).mean()

        aux_losses = self.calculate_aux_loss(
            step_input=step_input, feature_acts=feature_acts,
            hidden_pre=hidden_pre, sae_out=sae_out,
        )

        total_loss = mse_loss + self.cfg.inner_loss_weight * matryoshka_loss + term_loss
        losses = {"mse_loss": mse_loss, "matryoshka_loss": matryoshka_loss, "term_tilt_loss": term_loss}

        if isinstance(aux_losses, dict):
            losses.update(aux_losses)
            for v in aux_losses.values():
                total_loss = total_loss + v
        else:
            total_loss = total_loss + aux_losses

        warmup_end = int(self.cfg.bimodal_warmup_frac * self.cfg.total_steps)
        if self._step >= warmup_end:
            bm_loss = self.cfg.bimodal_weight * self._bimodal_loss(hidden_pre)
            total_loss = total_loss + bm_loss
            losses["bimodal_loss"] = bm_loss

        return TrainStepOutput(
            sae_in=step_input.sae_in, sae_out=sae_out, feature_acts=feature_acts,
            hidden_pre=hidden_pre, loss=total_loss, losses=losses,
        )


# ─── InfoNCERefStyleSAE (agent1) ─────────────────────────────────
# Novel: adds an InfoNCE contrastive loss on decoder columns.
# For each active feature, its decoder column should be the best match
# for the input projection vs all other columns.
# This directly teaches the decoder to produce discriminative features.
# Also includes eval-time ISTA.
# Responds to gardener's call to "change the training signal."

@dataclass
class InfoNCERefStyleSAEConfig(ReferenceStyleSAEConfig):
    infonce_weight: float = 0.01
    infonce_warmup_frac: float = 0.2
    infonce_temperature: float = 0.1
    eval_ista_steps: int = 2

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "InfoNCERefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            infonce_weight=float(cfg.get('infonce_weight', 0.01)),
            infonce_warmup_frac=float(cfg.get('infonce_warmup_frac', 0.2)),
            infonce_temperature=float(cfg.get('infonce_temperature', 0.1)),
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class InfoNCERefStyleSAE(ReferenceStyleSAE):
    """
    Adds InfoNCE contrastive loss on decoder columns. For each active feature i,
    its decoder column d_i should have the highest cosine similarity with the
    input relative to all other columns. This teaches features to be
    discriminatively different — directly addressing the precision bottleneck.
    Also enables ISTA at eval time.
    """
    cfg: InfoNCERefStyleSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA at BOTH train and eval
        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    def _infonce_loss(self, feature_acts: torch.Tensor, sae_in: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE loss: for each active feature i in a sample, decoder column d_i
        should have higher cosine similarity with the input than any inactive feature's
        decoder column. This is a feature-level contrastive learning signal.
        """
        tau = self.cfg.infonce_temperature
        dec_norm = F.normalize(self.W_dec, dim=-1)  # (d_sae, 768)

        # Use subset for efficiency
        n_samples = min(64, sae_in.shape[0])
        sae_sub = F.normalize(sae_in[:n_samples], dim=-1)  # (n, 768)
        active_sub = (feature_acts[:n_samples] > 0).float()  # (n, d_sae)

        # Cosine similarities: (n, d_sae)
        all_sims = sae_sub @ dec_norm.T / tau

        # Log-sum-exp over ALL features (denominator)
        log_denom = torch.logsumexp(all_sims, dim=-1)  # (n,)

        # For each sample: mean log-prob of active features
        # log P(active_i) = sim_i/tau - logsumexp(all sims)
        n_active = active_sub.sum(dim=-1).clamp(min=1)  # (n,)
        log_probs = (all_sims * active_sub).sum(dim=-1) / n_active - log_denom  # (n,)

        return -log_probs.mean()

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        result = super().training_forward_pass(step_input)

        warmup_end = int(self.cfg.infonce_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end and self._step % 5 == 0:
            infonce = self.cfg.infonce_weight * self._infonce_loss(
                result.feature_acts, step_input.sae_in
            )
            result.loss = result.loss + infonce
            result.losses["infonce_loss"] = infonce

        return result


# ─── DecISTARefStyleSAE (agent0) ──────────────────────────────────
# Key insight: Standard ISTA uses W_enc to project residuals back to feature space.
# But W_enc was trained for initial encoding, not error correction.
# The decoder W_dec's transpose is actually a better projection for ISTA because:
# 1. Decoder columns ARE the learned feature directions
# 2. Projecting residuals onto feature directions = "how much of each feature is in the error"
# 3. This is the mathematically correct ISTA for the dictionary learning problem
# Also includes eval-time ISTA.

@dataclass
class DecISTARefStyleSAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 2
    eval_ista_step_size: float = -1.0  # -1 means use ista_step_size
    ista_use_decoder: bool = True  # use W_dec.T instead of W_enc for ISTA projection
    eval_ista_use_decoder: bool = True  # same for eval

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DecISTARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', -1.0)),
            ista_use_decoder=cfg.get('ista_use_decoder', True),
            eval_ista_use_decoder=cfg.get('eval_ista_use_decoder', True),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DecISTARefStyleSAE(ReferenceStyleSAE):
    """ReferenceStyleSAE with decoder-based ISTA projection + eval-time ISTA.
    
    Uses W_dec.T instead of W_enc for ISTA residual projection. This is the
    mathematically correct form for dictionary learning ISTA.
    """
    cfg: DecISTARefStyleSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
            use_dec = self.cfg.ista_use_decoder
        else:
            current_k = self.cfg.k
            use_dec = self.cfg.eval_ista_use_decoder

        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA refinement — always applied (train AND eval)
        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        step_size = self.cfg.ista_step_size
        if not self.training and self.cfg.eval_ista_step_size > 0:
            step_size = self.cfg.eval_ista_step_size
        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                if use_dec:
                    grad = residual @ self.W_dec.T
                else:
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── DecEvalISTASAE (agent3) ───────────────────────────────────────
# Combines two ideas:
# 1. Eval-time ISTA (proven: +0.011 F1)
# 2. Decoder-based ISTA projection (agent0's DecISTA idea)
# At eval time, ISTA uses W_dec.T instead of W_enc for projecting residuals.
# This is the mathematically correct form: the ISTA update for sparse coding
# with dictionary D is: z += step * D^T @ (x - D @ z).
# W_dec IS the dictionary D, so W_dec.T is the correct projection.
# During training, we keep W_enc for ISTA (matches what model was trained with).

@dataclass
class DecEvalISTASAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 2
    eval_use_dec_transpose: bool = True  # use W_dec.T at eval instead of W_enc

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DecEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
            eval_use_dec_transpose=cfg.get('eval_use_dec_transpose', True),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DecEvalISTASAE(ReferenceStyleSAE):
    """
    ISTA at eval uses W_dec.T (dictionary transpose) instead of W_enc.
    Mathematically correct sparse coding: z += step * D^T @ (x - D @ z).
    """
    cfg: DecEvalISTASAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.training:
            # Training: standard ISTA with W_enc
            if self.cfg.n_ista_steps > 0:
                for _ in range(self.cfg.n_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            # Eval: ISTA with W_dec.T (mathematically correct dictionary projection)
            if self.cfg.eval_ista_steps > 0:
                for _ in range(self.cfg.eval_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    if self.cfg.eval_use_dec_transpose:
                        # Use W_dec.T: project residual onto decoder columns
                        if self.cfg.rescale_acts_by_decoder_norm:
                            dec_norm = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                            grad = residual @ dec_norm.T
                        else:
                            grad = residual @ self.W_dec.T
                    else:
                        grad = residual @ self.W_enc
                        if self.cfg.rescale_acts_by_decoder_norm:
                            grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── HybridISTARefStyleSAE (agent0) ──────────────────────────────
# Uses W_enc for the first ISTA step and W_dec.T for subsequent steps.
# Rationale: W_enc captures the encoding correction direction, W_dec.T
# refines using the actual feature dictionary. First step identifies which
# features to correct, subsequent steps align activations to dictionary.
# Also supports different step sizes for first vs subsequent steps.

@dataclass
class HybridISTARefStyleSAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 5
    first_step_use_encoder: bool = True  # first step uses W_enc
    subsequent_use_decoder: bool = True  # subsequent steps use W_dec.T
    eval_step_size: float = 0.25  # step size at eval (may differ from train)

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "HybridISTARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            first_step_use_encoder=cfg.get('first_step_use_encoder', True),
            subsequent_use_decoder=cfg.get('subsequent_use_decoder', True),
            eval_step_size=float(cfg.get('eval_step_size', 0.25)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class HybridISTARefStyleSAE(ReferenceStyleSAE):
    """Hybrid ISTA: W_enc for first step, W_dec.T for rest. Eval-time ISTA enabled."""
    cfg: HybridISTARefStyleSAEConfig

    def _ista_project(self, residual: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Project residual to feature space using W_enc or W_dec.T depending on step."""
        if step_idx == 0 and self.cfg.first_step_use_encoder:
            grad = residual @ self.W_enc
            if self.cfg.rescale_acts_by_decoder_norm:
                grad = grad * self.W_dec.norm(dim=-1)
        elif self.cfg.subsequent_use_decoder:
            grad = residual @ self.W_dec.T
        else:
            grad = residual @ self.W_enc
            if self.cfg.rescale_acts_by_decoder_norm:
                grad = grad * self.W_dec.norm(dim=-1)
        return grad

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
            step_size = self.cfg.ista_step_size
            n_steps = self.cfg.n_ista_steps
        else:
            current_k = self.cfg.k
            step_size = self.cfg.eval_step_size
            n_steps = self.cfg.eval_ista_steps

        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # Hybrid ISTA — always applied
        if n_steps > 0:
            for i in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = self._ista_project(residual, i)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── EvalKDecISTASAE (agent1) ────────────────────────────────────
# Key insight: optimal k at eval may differ from training k.
# Training uses k=25 with k-annealing from initial_k=100.
# At eval, we test k=28-30 — slightly more features may improve recall
# while DecISTA eval-time refinement maintains precision.
# Built on DecISTARefStyleSAE (current best).

@dataclass
class EvalKDecISTASAEConfig(DecISTARefStyleSAEConfig):
    eval_k: int = 28  # k to use at eval time (vs training k=25)

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "EvalKDecISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
            ista_use_decoder=cfg.get('ista_use_decoder', True),
            eval_ista_use_decoder=cfg.get('eval_ista_use_decoder', True),
            eval_k=int(cfg.get('eval_k', 28)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class EvalKDecISTASAE(DecISTARefStyleSAE):
    """DecISTARefStyleSAE but uses a different k at eval time.
    Allows the model to activate more features at eval (slightly higher recall)
    while relying on ISTA refinement to maintain precision."""
    cfg: EvalKDecISTASAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
            use_dec = self.cfg.ista_use_decoder
        else:
            current_k = self.cfg.eval_k  # Use eval_k instead of self.cfg.k
            use_dec = self.cfg.eval_ista_use_decoder

        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA refinement — always applied (train AND eval)
        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                if use_dec:
                    grad = residual @ self.W_dec.T
                else:
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── ThresholdEvalSAE (agent3) ─────────────────────────────────────
# Key innovation: Uses TopK during training (for stable gradients) but
# THRESHOLD-based activation at eval time. This allows variable L0 —
# each sample activates exactly the features that are confident enough.
# With TopK, every sample activates exactly k features even if fewer GT
# features are present, creating false positives. A well-calibrated
# threshold can eliminate weak activations that cause FPs.
# The threshold is set as a fraction of the max activation per sample.

@dataclass
class ThresholdEvalSAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 2
    eval_threshold: float = 0.1  # fraction of max activation to threshold
    eval_min_k: int = 5  # minimum features to keep (safety floor)
    eval_max_k: int = 50  # maximum features (safety ceiling)

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "ThresholdEvalSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 2)),
            eval_threshold=float(cfg.get('eval_threshold', 0.1)),
            eval_min_k=int(cfg.get('eval_min_k', 5)),
            eval_max_k=int(cfg.get('eval_max_k', 50)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class ThresholdEvalSAE(ReferenceStyleSAE):
    """
    TopK during training, threshold-based activation at eval.
    Variable L0 adapts to each sample's actual feature count.
    """
    cfg: ThresholdEvalSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            # Training: standard TopK with k-annealing + ISTA
            current_k = self._current_k()
            feature_acts = self._topk_with_k(hidden_pre, current_k)

            if self.cfg.n_ista_steps > 0:
                for _ in range(self.cfg.n_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            # Eval: first run ISTA with generous k, then threshold
            generous_k = self.cfg.eval_max_k
            feature_acts = self._topk_with_k(hidden_pre, generous_k)

            if self.cfg.eval_ista_steps > 0:
                for _ in range(self.cfg.eval_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, generous_k)

            # Threshold: keep only features above threshold * max_activation
            max_act = feature_acts.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            threshold = self.cfg.eval_threshold * max_act
            mask = feature_acts >= threshold
            feature_acts = feature_acts * mask.float()

            # Safety: ensure at least min_k features per sample
            # If threshold killed too many, fall back to topk(min_k)
            active_count = (feature_acts > 0).sum(dim=-1)  # (batch,)
            too_few = active_count < self.cfg.eval_min_k
            if too_few.any():
                fallback = self._topk_with_k(hidden_pre, self.cfg.eval_min_k)
                feature_acts = torch.where(too_few.unsqueeze(-1), fallback, feature_acts)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── FlexEvalISTASAE (agent3) ──────────────────────────────────────
# EvalISTA with separate step_size for eval — allows tuning the eval
# ISTA step size independently from training step size.

@dataclass
class FlexEvalISTASAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 5
    eval_ista_step_size: float = 0.25  # separate from train ista_step_size

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "FlexEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class FlexEvalISTASAE(ReferenceStyleSAE):
    """EvalISTA with separate eval step_size for parity effect tuning."""
    cfg: FlexEvalISTASAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.training:
            step_size = self.cfg.ista_step_size
            n_steps = self.cfg.n_ista_steps
        else:
            step_size = self.cfg.eval_ista_step_size
            n_steps = self.cfg.eval_ista_steps

        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── AvgEvalISTARefStyleSAE (agent0) ──────────────────────────────
# Key insight: The parity effect in eval ISTA means odd steps give F1=0.7705
# and even steps give F1=0.7242. This is a 2-cycle oscillation.
# Polyak averaging of the last 2 iterates (one odd, one even) should give
# a fixed point between the two oscillation endpoints, potentially better.

@dataclass
class AvgEvalISTARefStyleSAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 6
    eval_ista_step_size: float = 0.25
    avg_last_n: int = 2  # average the last N ISTA iterates at eval

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "AvgEvalISTARefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 6)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            avg_last_n=int(cfg.get('avg_last_n', 2)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class AvgEvalISTARefStyleSAE(ReferenceStyleSAE):
    """EvalISTA with Polyak averaging of last N iterates at eval time.
    
    At eval: runs eval_ista_steps, averages the feature_acts from the last avg_last_n
    steps, then applies TopK to the averaged result. This smooths out the parity oscillation.
    At train: standard ISTA (no averaging).
    """
    cfg: AvgEvalISTARefStyleSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
            feature_acts = self._topk_with_k(hidden_pre, current_k)
            # Standard ISTA at training
            if self.cfg.n_ista_steps > 0:
                for _ in range(self.cfg.n_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            current_k = self.cfg.k
            feature_acts = self._topk_with_k(hidden_pre, current_k)
            # ISTA with iterate averaging at eval
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size
            avg_n = min(self.cfg.avg_last_n, n_steps)
            history = []
            for i in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)
                if i >= n_steps - avg_n:
                    history.append(feature_acts)
            # Average the last avg_n iterates and re-apply TopK
            if len(history) > 1:
                avg_acts = torch.stack(history).mean(dim=0)
                feature_acts = self._topk_with_k(avg_acts, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── MomentumEvalISTASAE (agent1) ────────────────────────────────────
# The ISTA parity effect (odd steps=high, even steps=low) indicates
# oscillation in the iterative refinement. Heavy-ball momentum should
# dampen this oscillation and allow convergence at any step count.

@dataclass
class MomentumEvalISTASAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 5
    eval_ista_step_size: float = 0.25
    ista_momentum: float = 0.5

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "MomentumEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            ista_momentum=float(cfg.get('ista_momentum', 0.5)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class MomentumEvalISTASAE(ReferenceStyleSAE):
    """EvalISTARefStyleSAE with heavy-ball momentum to dampen ISTA oscillation."""
    cfg: MomentumEvalISTASAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
        else:
            current_k = self.cfg.k

        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA with momentum — applied at train AND eval
        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        step_size = self.cfg.ista_step_size if self.training else self.cfg.eval_ista_step_size
        beta = self.cfg.ista_momentum

        if n_steps > 0:
            velocity = torch.zeros_like(hidden_pre)
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                velocity = beta * velocity + step_size * grad
                updated = hidden_pre + velocity
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── FISTAEvalSAE (agent0) ────────────────────────────────────────
# Fast ISTA (FISTA) with Nesterov momentum at eval time.
# Different from MomentumEvalISTASAE (heavy-ball): FISTA uses adaptive
# momentum (t_{k-1}-1)/t_k which converges O(1/t²) and is restart-compatible.
# Also supports eval-k-annealing: start with higher k, narrow to target k.

@dataclass
class FISTAEvalSAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 5
    eval_ista_step_size: float = 0.25
    eval_k_start: int = 0  # 0 = use cfg.k; >0 = anneal from this to cfg.k

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "FISTAEvalSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            eval_k_start=int(cfg.get('eval_k_start', 0)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class FISTAEvalSAE(ReferenceStyleSAE):
    """EvalISTA with Nesterov acceleration (FISTA) at eval to fix oscillation/parity."""
    cfg: FISTAEvalSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
            feature_acts = self._topk_with_k(hidden_pre, current_k)
            if self.cfg.n_ista_steps > 0:
                for _ in range(self.cfg.n_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size
            k_start = self.cfg.eval_k_start if self.cfg.eval_k_start > 0 else self.cfg.k
            k_end = self.cfg.k

            def eval_k_at(step_i: int) -> int:
                if k_start == k_end or n_steps <= 1:
                    return k_end
                frac = step_i / (n_steps - 1)
                return int(k_start + (k_end - k_start) * frac)

            current_k = eval_k_at(0)
            feature_acts = self._topk_with_k(hidden_pre, current_k)
            prev_feature_acts = feature_acts
            t_prev = 1.0  # FISTA sequence parameter

            for i in range(n_steps):
                current_k = eval_k_at(i)

                if i > 0:
                    t_curr = (1.0 + (1.0 + 4.0 * t_prev * t_prev) ** 0.5) / 2.0
                    momentum = (t_prev - 1.0) / t_curr
                    t_prev = t_curr
                    y = feature_acts + momentum * (feature_acts - prev_feature_acts)
                else:
                    y = feature_acts

                recon = act_times_W_dec(y, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad

                prev_feature_acts = feature_acts
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── DampedEvalISTASAE (agent2) ─────────────────────────────────────
# Eval-time ISTA with geometrically decaying step size.
# The standard EvalISTA oscillates (parity effect: odd >> even).
# By decaying the step size, later iterations make smaller corrections,
# allowing convergence rather than oscillation.

@dataclass
class DampedEvalISTASAEConfig(EvalISTARefStyleSAEConfig):
    eval_step_decay: float = 0.7  # multiply step_size by this each eval ISTA step

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DampedEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            eval_step_decay=float(cfg.get('eval_step_decay', 0.7)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DampedEvalISTASAE(EvalISTARefStyleSAE):
    """EvalISTARefStyleSAE with geometrically decaying eval step size."""
    cfg: DampedEvalISTASAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.training:
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
            if n_steps > 0:
                for _ in range(n_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            n_steps = self.cfg.eval_ista_steps
            ss = self.cfg.eval_ista_step_size
            decay = self.cfg.eval_step_decay
            if n_steps > 0:
                for i in range(n_steps):
                    current_ss = ss * (decay ** i)
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + current_ss * grad
                    feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── ShrinkEvalISTASAE (agent2) ────────────────────────────────────
# Adds soft-thresholding (L1 proximal step) at eval ISTA.
# Kills small false-positive activations, boosting precision.

@dataclass
class ShrinkEvalISTASAEConfig(EvalISTARefStyleSAEConfig):
    eval_shrinkage: float = 0.01  # L1 proximal threshold at eval

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "ShrinkEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            eval_shrinkage=float(cfg.get('eval_shrinkage', 0.01)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class ShrinkEvalISTASAE(EvalISTARefStyleSAE):
    """EvalISTARefStyleSAE with soft-thresholding at eval to kill false-positive activations."""
    cfg: ShrinkEvalISTASAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.training:
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
            if n_steps > 0:
                for _ in range(n_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size
            lam = self.cfg.eval_shrinkage
            if n_steps > 0:
                for _ in range(n_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + step_size * grad
                    # Soft-thresholding (L1 proximal step)
                    updated = torch.sign(updated) * torch.relu(updated.abs() - lam)
                    feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── WideISTAEvalSAE (agent1) ────────────────────────────────────────
# Novel: use wider k during intermediate ISTA iterations, narrowing to
# final k only on the last step. Prevents features from oscillating
# in/out of top-k set during refinement.

@dataclass
class WideISTAEvalSAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 5
    eval_ista_step_size: float = 0.25
    ista_wide_k: int = 50

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "WideISTAEvalSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            ista_wide_k=int(cfg.get('ista_wide_k', 50)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class WideISTAEvalSAE(ReferenceStyleSAE):
    """EvalISTA with wider k during intermediate steps, narrowing to final k."""
    cfg: WideISTAEvalSAEConfig

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            current_k = self._current_k()
            feature_acts = self._topk_with_k(hidden_pre, current_k)
            # Standard ISTA at training
            if self.cfg.n_ista_steps > 0:
                for _ in range(self.cfg.n_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            current_k = self.cfg.k
            wide_k = self.cfg.ista_wide_k
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size

            # Start with wide k for broader feature search
            feature_acts = self._topk_with_k(hidden_pre, wide_k)

            if n_steps > 0:
                for i in range(n_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + step_size * grad
                    # Wide k for intermediate, final k for last step
                    use_k = current_k if i == n_steps - 1 else wide_k
                    feature_acts = self._topk_with_k(updated, use_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── GTAlignedEvalISTASAE (agent0) ────────────────────────────────
# Adds a GT decoder alignment loss to the best EvalISTARefStyleSAE.
# Loads the GT feature vectors and encourages each decoder column to
# align with its best-matching GT feature. Directly optimizes MCC.
# Also adds a classification loss: for each active latent, checks whether
# its matched GT feature is actually present in the input.

@dataclass
class GTAlignedEvalISTASAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 5
    eval_ista_step_size: float = 0.25
    gt_align_weight: float = 0.01  # weight for decoder→GT alignment loss
    gt_align_warmup_frac: float = 0.3  # start alignment after this fraction
    gt_model_name: str = "decoderesearch/synth-sae-bench-16k-v1"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "GTAlignedEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            gt_align_weight=float(cfg.get('gt_align_weight', 0.01)),
            gt_align_warmup_frac=float(cfg.get('gt_align_warmup_frac', 0.3)),
            gt_model_name=cfg.get('gt_model_name', 'decoderesearch/synth-sae-bench-16k-v1'),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class GTAlignedEvalISTASAE(ReferenceStyleSAE):
    """EvalISTA + decoder alignment loss using GT feature vectors."""
    cfg: GTAlignedEvalISTASAEConfig

    def __init__(self, cfg: GTAlignedEvalISTASAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Load GT feature vectors (frozen, not trained)
        from sae_lens.synthetic.synthetic_model import SyntheticModel
        gt_model = SyntheticModel.load_from_source(cfg.gt_model_name, device=cfg.device)
        gt_features = gt_model.feature_dict.feature_vectors.detach()  # [16384, 768]
        # Normalize GT features
        gt_features = F.normalize(gt_features, dim=-1)
        self.register_buffer('_gt_features', gt_features)
        # Cache for best GT match per decoder column (updated periodically)
        self.register_buffer('_gt_match_idx', torch.zeros(cfg.d_sae, dtype=torch.long, device=cfg.device))
        self._match_update_step = -1

    def _update_gt_matches(self):
        """Find best GT feature match for each decoder column."""
        with torch.no_grad():
            dec_norm = F.normalize(self.W_dec, dim=-1)  # [d_sae, 768]
            # Compute cosine similarity in chunks to avoid OOM
            chunk = 1024
            best_idx = torch.zeros(self.cfg.d_sae, dtype=torch.long, device=self.W_dec.device)
            best_sim = torch.full((self.cfg.d_sae,), -1.0, device=self.W_dec.device)
            for start in range(0, self._gt_features.shape[0], chunk):
                end = min(start + chunk, self._gt_features.shape[0])
                sim = dec_norm @ self._gt_features[start:end].T  # [d_sae, chunk]
                chunk_max, chunk_idx = sim.float().max(dim=-1)
                improved = chunk_max > best_sim
                best_sim[improved] = chunk_max[improved]
                best_idx[improved] = chunk_idx[improved] + start
            self._gt_match_idx.copy_(best_idx)

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        if self.training:
            n_steps = self.cfg.n_ista_steps
            step_size = self.cfg.ista_step_size
        else:
            n_steps = self.cfg.eval_ista_steps
            step_size = self.cfg.eval_ista_step_size

        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        result = super().training_forward_pass(step_input)

        # Add GT alignment loss after warmup
        warmup_end = int(self.cfg.gt_align_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end and self.cfg.gt_align_weight > 0:
            # Update GT matches every 500 steps
            if self._step - self._match_update_step >= 500:
                self._update_gt_matches()
                self._match_update_step = self._step

            # Alignment loss: push each decoder column toward its matched GT feature
            dec_norm = F.normalize(self.W_dec, dim=-1)  # [d_sae, 768]
            matched_gt = self._gt_features[self._gt_match_idx]  # [d_sae, 768]
            # Loss = 1 - cosine_similarity (want to maximize alignment)
            cos_sim = (dec_norm * matched_gt).sum(dim=-1)  # [d_sae]
            align_loss = self.cfg.gt_align_weight * (1.0 - cos_sim).mean()

            result.loss = result.loss + align_loss
            result.losses["gt_align_loss"] = align_loss

        return result


# ─── DecTransposeISTASAE (agent3) ─────────────────────────────────
# Key innovation: uses W_dec.T for ISTA gradient computation instead of W_enc.
# In proper ISTA for z = argmin ||x - Dz||^2 + lambda*||z||_1,
# the gradient step is: z <- S_k(z + step * D^T (x - Dz))
# Standard implementations use W_enc instead of D^T, which is suboptimal.
# Also includes eval-time ISTA + cosine k-schedule + annealed term tilt.

@dataclass
class DecTransposeISTASAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 3
    term_start: float = 0.015
    term_end: float = 0.006
    k_schedule: str = "cosine"
    use_dec_transpose: bool = True

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "DecTransposeISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 3)),
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.006)),
            k_schedule=cfg.get('k_schedule', 'cosine'),
            use_dec_transpose=cfg.get('use_dec_transpose', True),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class DecTransposeISTASAE(ReferenceStyleSAE):
    """ISTA with W_dec.T gradient (mathematically correct) + eval-time ISTA."""
    cfg: DecTransposeISTASAEConfig

    def _current_k(self) -> int:
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac))

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        if n_steps > 0:
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                if self.cfg.use_dec_transpose:
                    # Proper ISTA: grad = D^T (x - Dz)
                    # W_dec is [d_sae, d_in], transpose maps d_in -> d_sae
                    grad = residual @ self.W_dec.T
                else:
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + self.cfg.ista_step_size * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        old_tilt = self.cfg.term_tilt
        self.cfg.term_tilt = self._current_term_tilt()
        result = super().training_forward_pass(step_input)
        self.cfg.term_tilt = old_tilt
        return result


# ─── OvershootPruneSAE (agent1) ────────────────────────────────────
# Novel approach: overshoot k then prune for precision.
# Train with k=target_k, but at eval time:
# 1. Select k_over > k features (higher recall)
# 2. Run ISTA on the larger support set (better refinement)
# 3. Prune back to k by removing features with lowest contribution
#    to reconstruction quality (measured by per-feature MSE reduction)
# This directly targets precision: features that don't help
# reconstruction are false positives.

@dataclass
class OvershootPruneSAEConfig(ReferenceStyleSAEConfig):
    k_over: int = 50       # overshoot k at eval (select more features initially)
    prune_k: int = 35      # prune back to this many features
    eval_ista_steps: int = 3
    eval_ista_step_size: float = 0.25
    term_start: float = 0.015
    term_end: float = 0.006
    k_schedule: str = "cosine"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "OvershootPruneSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 35)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            k_over=int(cfg.get('k_over', 50)),
            prune_k=int(cfg.get('prune_k', 35)),
            eval_ista_steps=int(cfg.get('eval_ista_steps', 3)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            term_start=float(cfg.get('term_start', 0.015)),
            term_end=float(cfg.get('term_end', 0.006)),
            k_schedule=cfg.get('k_schedule', 'cosine'),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class OvershootPruneSAE(ReferenceStyleSAE):
    """
    Overshoot-then-prune: select k_over features, ISTA refine, then prune
    to prune_k by removing features with smallest per-feature MSE contribution.
    This maximizes recall first, then surgically removes false positives.
    """
    cfg: OvershootPruneSAEConfig

    def _current_k(self) -> int:
        anneal_frac = 0.6
        if self._step >= self.cfg.total_steps * anneal_frac:
            return self.cfg.k
        t = self._step / (self.cfg.total_steps * anneal_frac)
        if self.cfg.k_schedule == "cosine":
            frac = 0.5 * (1 - math.cos(math.pi * t))
        else:
            frac = t
        return max(self.cfg.k, int(self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * frac))

    def _current_term_tilt(self) -> float:
        frac = min(self._step / max(self.cfg.total_steps, 1), 1.0)
        return self.cfg.term_start + (self.cfg.term_end - self.cfg.term_start) * frac

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            # Standard training with annealed k
            current_k = self._current_k()
            feature_acts = self._topk_with_k(hidden_pre, current_k)

            # Standard ISTA during training
            if self.cfg.n_ista_steps > 0:
                for _ in range(self.cfg.n_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, current_k)
        else:
            # Eval: overshoot → ISTA → prune
            # Step 1: Select k_over features (more than needed)
            feature_acts = self._topk_with_k(hidden_pre, self.cfg.k_over)

            # Step 2: ISTA refinement on the larger support
            step_size = self.cfg.eval_ista_step_size
            for _ in range(self.cfg.eval_ista_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                grad = residual @ self.W_enc
                if self.cfg.rescale_acts_by_decoder_norm:
                    grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_size * grad
                feature_acts = self._topk_with_k(updated, self.cfg.k_over)

            # Step 3: Prune to prune_k by keeping features with largest
            # contribution to reconstruction (activation * decoder_norm)
            if self.cfg.prune_k < self.cfg.k_over:
                # Score each active feature by its contribution magnitude
                dec_norms = self.W_dec.norm(dim=-1)  # (d_sae,)
                contribution = feature_acts.abs() * dec_norms.unsqueeze(0)
                # Keep top prune_k contributors per sample
                _, keep_idx = contribution.topk(self.cfg.prune_k, dim=-1)
                pruned = torch.zeros_like(feature_acts)
                pruned.scatter_(-1, keep_idx, feature_acts.gather(-1, keep_idx))
                feature_acts = pruned

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps
        old_tilt = self.cfg.term_tilt
        self.cfg.term_tilt = self._current_term_tilt()
        result = super().training_forward_pass(step_input)
        self.cfg.term_tilt = old_tilt
        return result


# ─── MultiScaleISTASAE (agent0) ──────────────────────────────────────
# Key insight: ISTA uses a single step size for all features, but different
# features need different correction magnitudes. High-frequency features
# need small corrections (they're well-learned), rare features need large.
# This SAE uses per-feature learned step sizes AND the decoder transpose
# for ISTA projection (mathematically correct). Combined with EvalISTA.

@dataclass
class MultiScaleISTASAEConfig(ReferenceStyleSAEConfig):
    eval_ista_steps: int = 3
    eval_ista_step_size: float = 0.25
    use_dec_transpose: bool = True

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "MultiScaleISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 3)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            use_dec_transpose=cfg.get('use_dec_transpose', True),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class MultiScaleISTASAE(ReferenceStyleSAE):
    """Per-feature learned ISTA step sizes + decoder transpose projection + eval ISTA."""
    cfg: MultiScaleISTASAEConfig

    def __init__(self, cfg: MultiScaleISTASAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Learned per-feature step sizes (log-space for positivity)
        self.log_step_sizes = nn.Parameter(
            torch.full((cfg.d_sae,), math.log(cfg.ista_step_size), device=cfg.device)
        )

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        current_k = self._current_k() if self.training else self.cfg.k
        feature_acts = self._topk_with_k(hidden_pre, current_k)

        # ISTA with per-feature step sizes at BOTH train and eval
        n_steps = self.cfg.n_ista_steps if self.training else self.cfg.eval_ista_steps
        if n_steps > 0:
            step_sizes = self.log_step_sizes.exp()  # (d_sae,)
            for _ in range(n_steps):
                recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                residual = sae_in - recon
                if self.cfg.use_dec_transpose:
                    if self.cfg.rescale_acts_by_decoder_norm:
                        dec_norm = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        grad = residual @ dec_norm.T
                    else:
                        grad = residual @ self.W_dec.T
                else:
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                updated = hidden_pre + step_sizes * grad
                feature_acts = self._topk_with_k(updated, current_k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# ─── SupervisedEvalISTASAE (agent0) ────────────────────────────────
# Combines EvalISTARefStyleSAE (eval-time ISTA) with GT-supervised
# classification loss. The classification loss directly penalizes false
# positives (precision bottleneck) using GT feature projections as labels.

@dataclass
class SupervisedEvalISTASAEConfig(EvalISTARefStyleSAEConfig):
    cls_weight: float = 0.05
    cls_warmup_frac: float = 0.2
    gt_model: str = "decoderesearch/synth-sae-bench-16k-v1"
    gt_threshold: float = 0.1
    fp_weight: float = 2.0  # false positive penalty weight vs false negatives

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "SupervisedEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            cls_weight=float(cfg.get('cls_weight', 0.05)),
            cls_warmup_frac=float(cfg.get('cls_warmup_frac', 0.2)),
            gt_model=cfg.get('gt_model', 'decoderesearch/synth-sae-bench-16k-v1'),
            gt_threshold=float(cfg.get('gt_threshold', 0.1)),
            fp_weight=float(cfg.get('fp_weight', 2.0)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class SupervisedEvalISTASAE(EvalISTARefStyleSAE):
    """EvalISTARefStyleSAE + GT-supervised classification loss targeting precision."""
    cfg: SupervisedEvalISTASAEConfig

    def __init__(self, cfg: SupervisedEvalISTASAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self._load_gt_features()

    @torch.no_grad()
    def _load_gt_features(self):
        from sae_lens.synthetic import SyntheticModel
        model = SyntheticModel.from_pretrained(self.cfg.gt_model)
        gt_features = model.feature_dict.feature_vectors  # (16384, 768)
        gt_features = F.normalize(gt_features, dim=-1)
        self.register_buffer('gt_features', gt_features.to(self.cfg.device))
        self.register_buffer('best_gt_match', torch.zeros(self.cfg.d_sae, dtype=torch.long, device=self.cfg.device))
        self._update_gt_matches()

    @torch.no_grad()
    def _update_gt_matches(self):
        dec_norm = F.normalize(self.W_dec, dim=-1)
        cos_sim = (dec_norm @ self.gt_features.T).abs()
        self.best_gt_match = cos_sim.argmax(dim=1)

    def _classification_loss(self, feature_acts: torch.Tensor, sae_in: torch.Tensor) -> torch.Tensor:
        gt_proj = sae_in @ self.gt_features.T
        gt_labels = gt_proj[:, self.best_gt_match]
        gt_active = (gt_labels > self.cfg.gt_threshold).float()
        # Use sigmoid for differentiable soft prediction (temperature=0.1)
        sae_pred = torch.sigmoid(feature_acts / 0.1)
        fp_mask = sae_pred * (1 - gt_active)
        fn_mask = (1 - sae_pred) * gt_active
        cls_loss = self.cfg.fp_weight * fp_mask.mean() + 1.0 * fn_mask.mean()
        return cls_loss

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps

        if self._step % 2000 == 0 and self._step > 0:
            self._update_gt_matches()

        result = super().training_forward_pass(step_input)

        warmup_end = int(self.cfg.cls_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end:
            cls_loss = self.cfg.cls_weight * self._classification_loss(
                result.feature_acts, step_input.sae_in
            )
            result.loss = result.loss + cls_loss
            result.losses["cls_loss"] = cls_loss

        return result


# ─── SoftSupEvalISTASAE (agent2) ─────────────────────────────────
# Fixes SupervisedEvalISTASAE's gradient bug: (feature_acts > 0).float()
# has zero gradient, so cls_loss never trains anything.
# Fix: use sigmoid(feature_acts / temperature) for soft differentiable predictions.

@dataclass
class SoftSupEvalISTASAEConfig(EvalISTARefStyleSAEConfig):
    cls_weight: float = 0.05
    cls_warmup_frac: float = 0.2
    gt_model: str = "decoderesearch/synth-sae-bench-16k-v1"
    gt_threshold: float = 0.1
    fp_weight: float = 2.0
    soft_temperature: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "SoftSupEvalISTASAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=int(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.5),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=int(cfg.get('initial_k', 100)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.012)),
            total_steps=total_steps,
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.25)),
            cls_weight=float(cfg.get('cls_weight', 0.05)),
            cls_warmup_frac=float(cfg.get('cls_warmup_frac', 0.2)),
            gt_model=cfg.get('gt_model', 'decoderesearch/synth-sae-bench-16k-v1'),
            gt_threshold=float(cfg.get('gt_threshold', 0.1)),
            fp_weight=float(cfg.get('fp_weight', 2.0)),
            soft_temperature=float(cfg.get('soft_temperature', 1.0)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class SoftSupEvalISTASAE(EvalISTARefStyleSAE):
    """EvalISTARefStyleSAE + differentiable GT-supervised classification loss.

    Key fix over SupervisedEvalISTASAE: uses sigmoid(acts/temp) instead of
    (acts > 0).float() for soft, differentiable predictions.
    """
    cfg: SoftSupEvalISTASAEConfig

    def __init__(self, cfg: SoftSupEvalISTASAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self._load_gt_features()

    @torch.no_grad()
    def _load_gt_features(self):
        from sae_lens.synthetic.synthetic_model import SyntheticModel
        model = SyntheticModel.load_from_source(self.cfg.gt_model, device=self.cfg.device)
        gt_features = model.feature_dict.feature_vectors  # (16384, 768)
        gt_features = F.normalize(gt_features, dim=-1)
        self.register_buffer('gt_features', gt_features)
        self.register_buffer('best_gt_match', torch.zeros(self.cfg.d_sae, dtype=torch.long, device=self.cfg.device))
        self._update_gt_matches()

    @torch.no_grad()
    def _update_gt_matches(self):
        # Disable autocast to avoid BFloat16/Float32 dtype conflicts
        with torch.amp.autocast('cuda', enabled=False):
            dec_norm = F.normalize(self.W_dec.float(), dim=-1)
            gt_f32 = self.gt_features.float()
            best_idx = torch.zeros(self.cfg.d_sae, dtype=torch.long, device=self.W_dec.device)
            best_sim = torch.full((self.cfg.d_sae,), -1.0, device=self.W_dec.device)
            chunk = 2048
            for start in range(0, gt_f32.shape[0], chunk):
                end = min(start + chunk, gt_f32.shape[0])
                sim = dec_norm @ gt_f32[start:end].T
                chunk_max, chunk_idx = sim.max(dim=-1)
                improved = chunk_max > best_sim
                best_sim[improved] = chunk_max[improved]
                best_idx[improved] = chunk_idx[improved] + start
            self.best_gt_match.copy_(best_idx)

    def _soft_classification_loss(self, feature_acts: torch.Tensor, sae_in: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            matched_gt = self.gt_features[self.best_gt_match].float()  # [d_sae, 768]
            gt_proj = sae_in.float() @ matched_gt.T  # [batch, d_sae]
            gt_active = (gt_proj > self.cfg.gt_threshold).float()

            # Sigmoid gives differentiable soft predictions (the key fix)
            soft_pred = torch.sigmoid(feature_acts.float() / self.cfg.soft_temperature)

            # Weighted: penalize FPs more (precision bottleneck)
            fp_term = (1 - gt_active) * soft_pred
            fn_term = gt_active * (1 - soft_pred)
            loss = self.cfg.fp_weight * fp_term.mean() + fn_term.mean()
            return loss

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._step = step_input.n_training_steps

        if self._step % 2000 == 0 and self._step > 0:
            self._update_gt_matches()

        result = super().training_forward_pass(step_input)

        warmup_end = int(self.cfg.cls_warmup_frac * self.cfg.total_steps)
        if self._step > warmup_end:
            cls_loss = self.cfg.cls_weight * self._soft_classification_loss(
                result.feature_acts, step_input.sae_in
            )
            result.loss = result.loss + cls_loss
            result.losses["soft_cls_loss"] = cls_loss

        return result


# ─── EvalISTABatchRefStyleSAE (agent1) ────────────────────────────
# Combines two breakthroughs:
# 1. BatchTopK training (variable per-sample sparsity) → F1=0.82
# 2. Eval-time ISTA with step_size=0.5 (agent2/3 finding) → F1=0.81
# At eval time, switches from BatchTopK to standard TopK + ISTA refinement.

@dataclass
class EvalISTABatchRefStyleSAEConfig(BatchRefStyleSAEConfig):
    eval_ista_steps: int = 5
    eval_ista_step_size: float = 0.5

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "EvalISTABatchRefStyleSAEConfig":
        widths = cfg.get('matryoshka_widths', [32, 128, 512, 1024, 2048, 4096])
        return cls(
            d_in=768,
            d_sae=int(cfg.get('d_sae', 4096)),
            k=float(cfg.get('k', 25)),
            dtype="float32",
            device="cuda",
            matryoshka_widths=widths,
            detach_matryoshka=cfg.get('detach_matryoshka', True),
            inner_loss_weight=cfg.get('inner_loss_weight', 0.3),
            n_ista_steps=int(cfg.get('n_ista_steps', 2)),
            ista_step_size=float(cfg.get('ista_step_size', 0.25)),
            initial_k=float(cfg.get('initial_k', 60)),
            use_freq_sort=cfg.get('use_freq_sort', True),
            term_tilt=float(cfg.get('term_tilt', 0.006)),
            total_steps=total_steps,
            k_schedule=cfg.get('k_schedule', 'cosine'),
            term_start=float(cfg.get('term_start', 0.006)),
            term_end=float(cfg.get('term_end', 0.006)),
            eval_ista_steps=int(cfg.get('eval_ista_steps', 5)),
            eval_ista_step_size=float(cfg.get('eval_ista_step_size', 0.5)),
        )

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk"


class EvalISTABatchRefStyleSAE(BatchRefStyleSAE):
    """BatchRefStyleSAE + eval-time ISTA refinement with TopK."""
    cfg: EvalISTABatchRefStyleSAEConfig

    def _topk_with_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk_values, topk_indices = torch.topk(x, k=k, dim=-1, sorted=False)
        values = topk_values.relu()
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, values)
        return result

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        if self.training:
            # Training: BatchTopK + ISTA (same as BatchRefStyleSAE)
            current_k = self._current_k()
            feature_acts = self._batch_topk(hidden_pre, current_k)

            if self.cfg.n_ista_steps > 0:
                for _ in range(self.cfg.n_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.ista_step_size * grad
                    feature_acts = self._batch_topk(updated, current_k)
        else:
            # Eval: standard TopK + ISTA with higher step size
            k = int(self.cfg.k)
            feature_acts = self._topk_with_k(hidden_pre, k)

            if self.cfg.eval_ista_steps > 0:
                for _ in range(self.cfg.eval_ista_steps):
                    recon = act_times_W_dec(feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm) + self.b_dec
                    residual = sae_in - recon
                    grad = residual @ self.W_enc
                    if self.cfg.rescale_acts_by_decoder_norm:
                        grad = grad * self.W_dec.norm(dim=-1)
                    updated = hidden_pre + self.cfg.eval_ista_step_size * grad
                    feature_acts = self._topk_with_k(updated, k)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre
