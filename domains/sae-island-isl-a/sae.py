# SAE Architecture — agents edit this file
#
# Custom SAE classes for SynthSAEBench-16k optimization.

from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import override

from sae_lens.saes.matching_pursuit_sae import (
    MatchingPursuitTrainingSAE,
    MatchingPursuitTrainingSAEConfig,
)

D_IN = 768


# --- Matching Pursuit SAE ---
# Iterative greedy feature selection from residuals with tied weights.
# Should recover far more features in extreme superposition than one-shot BatchTopK.

@dataclass
class MPSAEConfig(MatchingPursuitTrainingSAEConfig):
    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int) -> "MPSAEConfig":
        return cls(
            d_in=D_IN,
            d_sae=int(cfg['d_sae']),
            residual_threshold=float(cfg.get('residual_threshold', 0.05)),
            max_iterations=cfg.get('max_iterations', None),
            stop_on_duplicate_support=cfg.get('stop_on_duplicate_support', True),
            dtype="float32",
            device="cuda",
        )


class MPSAE(MatchingPursuitTrainingSAE):
    """Matching Pursuit SAE wrapper for engine.py compatibility."""
    cfg: MPSAEConfig

    def __init__(self, cfg: MPSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)


# --- Diverse BatchTopK SAE (Agent 2) ---
# Adds decoder diversity loss to push latent directions apart,
# maximizing unique GT feature coverage (directly targets F1 metric).

import torch.nn.functional as F
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput


@dataclass
class DiverseTopKSAEConfig(BatchTopKTrainingSAEConfig):
    """BatchTopK with decoder diversity loss."""

    diversity_coeff: float = 0.02
    diversity_sample_size: int = 512

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "DiverseTopKSAEConfig":
        return cls(
            d_in=D_IN,
            d_sae=int(cfg.get("d_sae", 4096)),
            k=float(cfg.get("k", 25)),
            dtype=cfg.get("dtype", "float32"),
            device=cfg.get("device", "cuda"),
            diversity_coeff=float(cfg.get("diversity_coeff", 0.02)),
            diversity_sample_size=int(cfg.get("diversity_sample_size", 512)),
        )


class DiverseTopKSAE(BatchTopKTrainingSAE):
    """BatchTopK SAE with decoder column diversity loss.

    Penalizes high cosine similarity between decoder columns to encourage
    each latent to represent a unique ground-truth feature direction.
    """

    cfg: DiverseTopKSAEConfig

    def __init__(self, cfg: DiverseTopKSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)

        if self.cfg.diversity_coeff > 0:
            n = self.cfg.diversity_sample_size
            d_sae = self.W_dec.shape[0]

            idx_a = torch.randint(0, d_sae, (n,), device=self.W_dec.device)
            idx_b = torch.randint(0, d_sae, (n,), device=self.W_dec.device)
            mask = idx_a == idx_b
            idx_b[mask] = (idx_b[mask] + 1) % d_sae

            w_a = F.normalize(self.W_dec[idx_a], dim=-1)
            w_b = F.normalize(self.W_dec[idx_b], dim=-1)
            cos_sim = (w_a * w_b).sum(dim=-1)

            diversity_loss = cos_sim.pow(2).mean()
            output.loss = output.loss + self.cfg.diversity_coeff * diversity_loss
            output.losses["diversity_loss"] = diversity_loss

        return output


# --- ISTA BatchTopK SAE (Agent 1) ---
# Iterative Shrinkage-Thresholding encoder for better feature recovery
# in extreme superposition. Multiple encode-decode-correct cycles.

from sae_lens.saes.topk_sae import act_times_W_dec
from dataclasses import field
from sae_lens.saes.matryoshka_batchtopk_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)


# --- Matryoshka BatchTopK SAE (Agent 3) ---
# Per SynthSAEBench paper, Matryoshka SAEs achieve best F1 (~0.88).
# Nested reconstruction losses at multiple widths prevent feature absorption
# and encourage learning distinct features at each level.

@dataclass
class MatryoshkaSAEConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """Matryoshka BatchTopK for SynthSAEBench-16k F1 optimization."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "matryoshka_batchtopk"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "MatryoshkaSAEConfig":
        # Default widths: geometric progression ending at d_sae
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            dtype="float32",
            device="cuda",
        )


class MatryoshkaSAE(MatryoshkaBatchTopKTrainingSAE):
    """Matryoshka BatchTopK SAE wrapper for engine.py compatibility."""
    cfg: MatryoshkaSAEConfig

    def __init__(self, cfg: MatryoshkaSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)


@dataclass
class ISTABatchTopKConfig(BatchTopKTrainingSAEConfig):
    """BatchTopK with ISTA-style iterative refinement in the encoder."""

    n_ista_steps: int = 3
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "ista_batchtopk"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "ISTABatchTopKConfig":
        return cls(
            d_in=D_IN,
            d_sae=int(cfg['d_sae']),
            k=float(cfg['k']),
            n_ista_steps=int(cfg.get('n_ista_steps', 3)),
            ista_step_size=float(cfg.get('ista_step_size', 0.3)),
            dtype="float32",
            device="cuda",
        )


class ISTABatchTopK(BatchTopKTrainingSAE):
    """BatchTopK SAE with ISTA-style iterative encoder refinement.

    Instead of a single encoder pass, runs multiple iterations:
    1. Initial encoding: z = topk(x @ W_enc + b_enc)
    2. For each step: compute residual, project back, update, re-topk

    This helps recover features missed by single-pass encoding due to
    interference in extreme superposition (16k features in 768 dims).
    """

    cfg: ISTABatchTopKConfig  # type: ignore[assignment]

    def __init__(self, cfg: ISTABatchTopKConfig, use_error_term: bool = False):
        super().__init__(cfg)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Initial encoding (same as standard BatchTopK)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # ISTA refinement steps
        for _ in range(self.cfg.n_ista_steps):
            # Reconstruct from current activations
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            # Compute residual and project to latent space
            residual = sae_in - recon
            correction = residual @ self.W_enc
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            # Update and re-apply activation
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- ISTA + Matryoshka Hybrid SAE (Agent 1, EXP2) ---
# Combines ISTA iterative encoder with Matryoshka nested losses.
# ISTA improves feature recovery per sample; Matryoshka prevents feature absorption.

@dataclass
class ISTAMatryoshkaConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """ISTA encoder + Matryoshka nested reconstruction losses."""

    n_ista_steps: int = 3
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "ista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "ISTAMatryoshkaConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 3)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class ISTAMatryoshka(MatryoshkaBatchTopKTrainingSAE):
    """Matryoshka SAE with ISTA-style iterative encoder.

    Combines two complementary improvements:
    1. ISTA encoder: iteratively refines feature activations using
       reconstruction residual feedback (better feature recovery)
    2. Matryoshka loss: nested reconstruction at multiple widths
       prevents feature absorption (better feature diversity)
    """

    cfg: ISTAMatryoshkaConfig  # type: ignore[assignment]

    def __init__(self, cfg: ISTAMatryoshkaConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # ISTA refinement steps
        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_enc
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Residual SAE (Agent 2, EXP2) ---
# Two-encoder architecture: primary encoder for initial pass,
# separate learned residual encoder that specializes in catching
# features missed by the first pass. Unlike ISTA (reuses W_enc),
# the residual encoder can learn patterns specific to residual signals.

import torch.nn as nn
from sae_lens.saes.batchtopk_sae import BatchTopK as BatchTopKActivation


@dataclass
class ResidualSAEConfig(BatchTopKTrainingSAEConfig):
    """BatchTopK with a second encoder for residual-based feature recovery."""

    k_pass1: int = 15  # features in first pass
    k_pass2: int = 10  # features in second pass (from residual)

    @override
    @classmethod
    def architecture(cls) -> str:
        return "batchtopk"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "ResidualSAEConfig":
        k = int(cfg.get("k", 25))
        k_pass1 = int(cfg.get("k_pass1", k * 3 // 5))  # 60% of k
        k_pass2 = int(cfg.get("k_pass2", k - k_pass1))  # remaining
        return cls(
            d_in=D_IN,
            d_sae=int(cfg.get("d_sae", 4096)),
            k=float(k),  # total k for parent class
            k_pass1=k_pass1,
            k_pass2=k_pass2,
            dtype="float32",
            device="cuda",
        )


class ResidualSAE(BatchTopKTrainingSAE):
    """Two-encoder BatchTopK SAE with residual pursuit.

    Architecture:
    - W_enc (primary): standard encoder, selects k_pass1 features
    - W_enc2 (residual): separately learned encoder, applied to
      reconstruction residual, selects k_pass2 features from
      latents NOT selected in pass 1
    - W_dec (shared): single decoder for both passes

    Why this works:
    In extreme superposition (16k features in 768d), a single encoder pass
    misses features "shadowed" by stronger ones. The residual encoder
    learns to detect these shadowed features specifically, without
    interference from already-detected features.

    Unlike ISTA: ISTA reuses the same encoder weights for corrections.
    Our residual encoder has its own learned weights optimized specifically
    for the residual signal distribution, which differs from the input distribution.
    """

    cfg: ResidualSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: ResidualSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Second encoder for residual pass
        self.W_enc2 = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )
        self.b_enc2 = nn.Parameter(
            torch.zeros(cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        # Separate TopK for each pass (per-sample, not batch-wide)
        from sae_lens.saes.topk_sae import TopK
        self.topk_pass1 = TopK(cfg.k_pass1)
        self.topk_pass2 = TopK(cfg.k_pass2)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Pass 1: primary encoder
        hidden_pre1 = sae_in @ self.W_enc + self.b_enc
        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre1 = hidden_pre1 * W_dec_norms

        acts1 = self.topk_pass1(hidden_pre1)

        # Compute residual
        recon1 = act_times_W_dec(acts1, self.W_dec, self.cfg.rescale_acts_by_decoder_norm)
        residual = sae_in - recon1

        # Pass 2: residual encoder
        hidden_pre2 = residual @ self.W_enc2 + self.b_enc2
        if W_dec_norms is not None:
            hidden_pre2 = hidden_pre2 * W_dec_norms

        # Mask out latents already active in pass 1 so pass 2 uses different ones
        active_mask = (acts1 > 0)
        hidden_pre2 = hidden_pre2.masked_fill(active_mask, float('-inf'))

        acts2 = self.topk_pass2(hidden_pre2)

        # Combine activations
        feature_acts = acts1 + acts2
        hidden_pre = torch.max(hidden_pre1, hidden_pre2)  # for aux loss compatibility

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- LISTA + Matryoshka SAE (Agent 1, EXP2) ---
# Learned ISTA: uses a separate learned correction matrix W_corr for
# residual projection instead of reusing W_enc. The residual signal
# distribution differs from the input distribution, so a dedicated
# correction encoder learns better residual-to-latent mappings.
# From compressed sensing LISTA (Learned ISTA) literature.

@dataclass
class LISTAMatryoshkaConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """Learned ISTA encoder + Matryoshka nested reconstruction losses."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "LISTAMatryoshkaConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class LISTAMatryoshka(MatryoshkaBatchTopKTrainingSAE):
    """Matryoshka SAE with Learned ISTA encoder.

    Key difference from ISTAMatryoshka: uses a separate learned W_corr
    matrix for the ISTA correction step instead of reusing W_enc.

    Why: W_enc is optimized to encode the raw input x. But the ISTA
    correction step operates on reconstruction residuals, which have a
    different distribution (smaller magnitude, concentrated on missed
    features). A dedicated W_corr learns to detect residual features
    specifically, without the constraint of also being a good initial encoder.

    This is the LISTA (Learned ISTA) principle from Gregor & LeCun (2010).
    """

    cfg: LISTAMatryoshkaConfig  # type: ignore[assignment]

    def __init__(self, cfg: LISTAMatryoshkaConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Separate correction encoder for residual projection
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Initial encoding with W_enc (same as standard)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # LISTA refinement steps using learned W_corr
        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            # Use W_corr instead of W_enc for residual projection
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Enhanced LISTA + Matryoshka SAE (Agent 0, EXP11) ---
# Combines our LISTA (5 steps, W_corr) with innovations from the 0.97 reference:
# 1. Detached Matryoshka inner losses (prevent gradient conflicts between widths)
# 2. TERM loss (softmax-weighted, up-weights hard samples)
# 3. Correct Matryoshka widths [128, 512, 2048, 4096]

import math

@dataclass
class EnhancedLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA + Detached Matryoshka + TERM loss."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3
    term_tilt: float = 0.002  # very mild TERM (reference uses 0.002)
    detach_matryoshka: bool = True  # detach accumulated reconstruction in inner losses

    @override
    @classmethod
    def architecture(cls) -> str:
        return "enhanced_lista"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "EnhancedLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [128, 512, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            term_tilt=float(cfg.get("term_tilt", 0.002)),
            detach_matryoshka=cfg.get("detach_matryoshka", True),
            dtype="float32",
            device="cuda",
        )


class EnhancedLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA + Detached Matryoshka + TERM loss.

    Key innovations over plain LISTAMatryoshka:
    1. Detached inner Matryoshka losses: gradient from inner losses only flows
       through the current width's features, not previously decoded ones.
       Prevents gradient conflicts between widths.
    2. TERM loss: softmax-weighted mean that up-weights hard samples.
       Forces SAE to handle extreme superposition cases better.
    """

    cfg: EnhancedLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: EnhancedLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    def _term_mean(self, per_sample_loss: torch.Tensor) -> torch.Tensor:
        """TERM-weighted mean using softmax reweighting (reference implementation)."""
        t = self.cfg.term_tilt
        if t <= 0:
            return per_sample_loss.mean()
        # Softmax reweighting with normalized tilt
        weights = torch.softmax(
            per_sample_loss.detach() * t / per_sample_loss.detach().mean(), dim=0
        )
        return (weights * per_sample_loss).sum()

    def _detached_iterable_decode(
        self, feature_acts: torch.Tensor
    ):
        """Decode at each Matryoshka width with detached accumulation.

        Unlike standard Matryoshka, detaches the accumulated reconstruction
        before adding the next width's contribution. This ensures inner loss
        gradients only flow through features in the current width range.
        """
        if self.cfg.rescale_acts_by_decoder_norm:
            inv_W_dec_norm = 1 / self.W_dec.norm(dim=-1)
            feature_acts = feature_acts * inv_W_dec_norm

        decoded = self.b_dec
        prev_width = 0
        # Skip the last (full) width — that's the main MSE loss
        widths = self.cfg.matryoshka_widths[:-1]

        for i, width in enumerate(widths):
            inner_acts = feature_acts[:, prev_width:width]
            current_delta = inner_acts @ self.W_dec[prev_width:width]
            if self.cfg.detach_matryoshka and i > 0:
                decoded = decoded.detach()
            decoded = decoded + current_delta
            prev_width = width
            yield width, self.run_time_activation_norm_fn_out(decoded)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)
        sae_in = step_input.sae_in

        # TERM-weighted main MSE loss
        per_sample_mse = self.mse_loss_fn(sae_out, sae_in).sum(dim=-1)
        mse_loss = self._term_mean(per_sample_mse)

        # Aux loss (dead neuron revival)
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        losses = {"mse_loss": mse_loss}
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)

        # Detached + TERM-weighted Matryoshka inner losses
        for width, inner_recon in self._detached_iterable_decode(feature_acts):
            inner_per_sample = self.mse_loss_fn(inner_recon, sae_in).sum(dim=-1)
            inner_loss = self._term_mean(inner_per_sample)
            losses[f"inner_mse_loss_{width}"] = inner_loss

        # Sum all losses
        total_loss = torch.stack(list(losses.values())).sum()

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )


# --- FreqSortEnhancedLISTA: EnhancedLISTA + periodic frequency sorting ---
# Periodically sorts all weight matrices by activation frequency so that
# the most frequently-activated features occupy the lowest indices.
# This makes Matryoshka inner losses focus on the most important features.

@dataclass
class FreqSortEnhancedLISTAConfig(EnhancedLISTAConfig):
    """EnhancedLISTA + periodic frequency sorting."""

    sort_every: int = 1000
    sort_warmup: int = 2000

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "FreqSortEnhancedLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [128, 512, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            term_tilt=float(cfg.get("term_tilt", 0.002)),
            detach_matryoshka=cfg.get("detach_matryoshka", True),
            sort_every=int(cfg.get("sort_every", 1000)),
            sort_warmup=int(cfg.get("sort_warmup", 2000)),
            dtype="float32",
            device="cuda",
        )


class FreqSortEnhancedLISTA(EnhancedLISTA):
    """EnhancedLISTA with periodic frequency-based feature sorting."""

    cfg: FreqSortEnhancedLISTAConfig

    def __init__(self, cfg: FreqSortEnhancedLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.register_buffer(
            "feature_freq",
            torch.zeros(cfg.d_sae, dtype=torch.float32, device=cfg.device),
        )
        self._last_sort_step = 0

    @torch.no_grad()
    def _update_freq(self, feature_acts: torch.Tensor) -> None:
        batch_freq = (feature_acts > 0).float().mean(dim=0)
        self.feature_freq.mul_(0.99).add_(batch_freq, alpha=0.01)

    @torch.no_grad()
    def _sort_by_freq(self) -> None:
        sorted_indices = self.feature_freq.argsort(descending=True)
        self.W_enc.data = self.W_enc.data[:, sorted_indices]
        self.W_dec.data = self.W_dec.data[sorted_indices]
        self.W_corr.data = self.W_corr.data[:, sorted_indices]
        self.b_enc.data = self.b_enc.data[sorted_indices]
        self.feature_freq.data = self.feature_freq.data[sorted_indices]
        if hasattr(self, 'log_threshold') and self.log_threshold is not None:
            self.log_threshold.data = self.log_threshold.data[sorted_indices]

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        step = step_input.n_training_steps
        if (
            step >= self.cfg.sort_warmup
            and step - self._last_sort_step >= self.cfg.sort_every
        ):
            self._sort_by_freq()
            self._last_sort_step = step

        output = super().training_forward_pass(step_input)
        self._update_freq(output.feature_acts)
        return output


# --- FullEnhancedLISTA: FreqSort + Decreasing K + EnhancedLISTA ---
# Combines all known improvements from the reference implementation.

@dataclass
class FullEnhancedLISTAConfig(FreqSortEnhancedLISTAConfig):
    """EnhancedLISTA + frequency sorting + decreasing K schedule."""

    k_start: float = 40.0  # starting K (higher for exploration)
    k_end: float = 25.0    # final K
    k_warmup_frac: float = 0.1  # fraction of training at k_start before decay
    total_training_steps: int = 48828  # for K schedule

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "FullEnhancedLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [128, 512, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        ts = total_steps if total_steps > 0 else int(cfg.get("training_samples", 50000000)) // int(cfg.get("batch_size", 1024))
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            term_tilt=float(cfg.get("term_tilt", 0.002)),
            detach_matryoshka=cfg.get("detach_matryoshka", True),
            sort_every=int(cfg.get("sort_every", 1000)),
            sort_warmup=int(cfg.get("sort_warmup", 2000)),
            k_start=float(cfg.get("k_start", 40.0)),
            k_end=float(cfg.get("k_end", cfg.get("k", 25.0))),
            k_warmup_frac=float(cfg.get("k_warmup_frac", 0.1)),
            total_training_steps=ts,
            dtype="float32",
            device="cuda",
        )


class FullEnhancedLISTA(FreqSortEnhancedLISTA):
    """EnhancedLISTA + frequency sorting + decreasing K schedule.

    K schedule: k_start for first k_warmup_frac of training, then
    linearly decreases to k_end over remaining training.
    """

    cfg: FullEnhancedLISTAConfig

    def _get_current_k(self, step: int, total_steps: int) -> float:
        """Compute current k value based on training progress."""
        warmup_steps = int(total_steps * self.cfg.k_warmup_frac)
        if step < warmup_steps:
            return self.cfg.k_start
        # Linear decay from k_start to k_end
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return self.cfg.k_start + (self.cfg.k_end - self.cfg.k_start) * progress

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        # Update K based on training schedule
        current_k = self._get_current_k(
            step_input.n_training_steps, self.cfg.total_training_steps
        )
        self.activation_fn.k = current_k

        # Run parent forward pass (includes freq sorting + EnhancedLISTA)
        return super().training_forward_pass(step_input)


# --- ReferenceStyleSAE: 1-step ISTA + freq-sort Matryoshka + TERM + decreasing K ---
# Faithful reimplementation of the 0.97 reference approach.
# Key differences from our EnhancedLISTA:
# 1. Uses W_enc for correction (not separate W_corr) — simpler, no extra params
# 2. 1 ISTA step (not 5) — faster per batch
# 3. Frequency sorting via INDEX MAPPING (not weight permutation) — no Adam issues
# 4. Decreasing K schedule (60→25 linearly) — wider exploration early
# 5. Standard topk aux loss (not matryoshka aux loss)

class FiringFreqTracker(nn.Module):
    """Track per-latent firing frequencies via EMA."""
    def __init__(self, d_sae: int, ema_decay: float = 0.99, device: str = "cuda"):
        super().__init__()
        self.ema_decay = ema_decay
        self.register_buffer("frequencies", torch.zeros(d_sae, device=device, dtype=torch.float64))
        self.register_buffer("_initialized", torch.zeros(1, device=device, dtype=torch.bool))

    @torch.no_grad()
    def update(self, feature_acts: torch.Tensor) -> None:
        flat = feature_acts.reshape(-1, feature_acts.shape[-1])
        batch_freq = (flat != 0).sum(dim=0).to(self.frequencies.dtype) / flat.shape[0]
        if not self._initialized.item():
            self.frequencies.copy_(batch_freq)
            self._initialized.fill_(True)
        else:
            self.frequencies.mul_(self.ema_decay).add_(batch_freq, alpha=1 - self.ema_decay)


@dataclass
class ReferenceStyleSAEConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """1-step ISTA + freq-sorted detached Matryoshka + TERM + decreasing K."""

    n_ista_steps: int = 1
    ista_step_size: float = 0.3
    term_tilt: float = 0.002
    detach_matryoshka: bool = True
    use_freq_sort: bool = True
    initial_k: float = 60.0
    k_transition_steps: int = 0  # 0 = auto (full training)
    total_training_steps: int = 0
    k_schedule: str = "linear"  # "linear" or "cosine"
    inner_loss_weight: float = 1.0  # weight for inner Matryoshka losses

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "ReferenceStyleSAEConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [128, 512, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        ts = total_steps if total_steps > 0 else int(cfg.get("training_samples", 50000000)) // int(cfg.get("batch_size", 1024))
        k_trans = int(cfg.get("k_transition_steps", 0))
        if k_trans == 0:
            k_frac = float(cfg.get("k_transition_frac", 1.0))
            k_trans = int(ts * k_frac)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=False,
            n_ista_steps=int(cfg.get("n_ista_steps", 1)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            term_tilt=float(cfg.get("term_tilt", 0.002)),
            detach_matryoshka=cfg.get("detach_matryoshka", True),
            use_freq_sort=cfg.get("use_freq_sort", True),
            initial_k=float(cfg.get("initial_k", 60.0)),
            k_transition_steps=k_trans,
            total_training_steps=ts,
            k_schedule=cfg.get("k_schedule", "linear"),
            inner_loss_weight=float(cfg.get("inner_loss_weight", 1.0)),
            dtype="float32",
            device="cuda",
        )


class ReferenceStyleSAE(MatryoshkaBatchTopKTrainingSAE):
    """Faithful reimplementation of the 0.97 reference approach."""

    cfg: ReferenceStyleSAEConfig

    def __init__(self, cfg: ReferenceStyleSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        if cfg.use_freq_sort:
            self.freq_tracker = FiringFreqTracker(cfg.d_sae, device=cfg.device)
        else:
            self.freq_tracker = None

    def _get_current_k(self, step: int) -> float:
        if self.cfg.k_transition_steps <= 0:
            return self.cfg.k
        if step >= self.cfg.k_transition_steps:
            return self.cfg.k
        progress = step / self.cfg.k_transition_steps
        if self.cfg.k_schedule == "cosine":
            import math
            progress = (1 - math.cos(math.pi * progress)) / 2
        return self.cfg.initial_k + (self.cfg.k - self.cfg.initial_k) * progress

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        for _ in range(self.cfg.n_ista_steps):
            if self.cfg.rescale_acts_by_decoder_norm:
                recon = (feature_acts / W_dec_norms) @ self.W_dec
            else:
                recon = feature_acts @ self.W_dec
            residual = sae_in - recon
            delta = residual @ self.W_enc
            if W_dec_norms is not None:
                delta = delta * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * delta
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    def _freq_sorted_iterable_decode(self, feature_acts: torch.Tensor):
        """Decode with frequency-sorted index mapping for Matryoshka."""
        if self.cfg.rescale_acts_by_decoder_norm:
            inv_norm = 1 / self.W_dec.norm(dim=-1)
            acts = feature_acts * inv_norm
        else:
            acts = feature_acts

        sorted_indices = None
        if self.freq_tracker is not None:
            sorted_indices = self.freq_tracker.frequencies.argsort(descending=True)

        decoded = self.b_dec
        prev_portion = 0
        widths = self.cfg.matryoshka_widths[:-1]

        for i, width in enumerate(widths):
            if sorted_indices is not None:
                idx = sorted_indices[prev_portion:width]
                current_delta = acts[:, idx] @ self.W_dec[idx]
            else:
                current_delta = acts[:, prev_portion:width] @ self.W_dec[prev_portion:width]
            if self.cfg.detach_matryoshka and i > 0:
                decoded = decoded.detach()
            decoded = decoded + current_delta
            prev_portion = width
            yield width, self.run_time_activation_norm_fn_out(decoded)

    def _term_mean(self, per_sample_loss: torch.Tensor) -> torch.Tensor:
        t = self.cfg.term_tilt
        if t <= 0:
            return per_sample_loss.mean()
        weights = torch.softmax(
            per_sample_loss.detach() * t / per_sample_loss.detach().mean(), dim=0
        )
        return (weights * per_sample_loss).sum()

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        # Update K schedule
        current_k = self._get_current_k(step_input.n_training_steps)
        self.activation_fn.k = current_k

        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)
        sae_in = step_input.sae_in

        # Update frequency tracker
        if self.freq_tracker is not None:
            self.freq_tracker.update(feature_acts)

        # TERM-weighted outer MSE loss
        per_sample_mse = self.mse_loss_fn(sae_out, sae_in).sum(dim=-1)
        mse_loss = self._term_mean(per_sample_mse)

        # Standard dead latent aux loss
        aux_loss = self.calculate_topk_aux_loss(
            sae_in, sae_out, hidden_pre, step_input.dead_neuron_mask
        )

        losses = {
            "mse_loss": mse_loss,
            "auxiliary_reconstruction_loss": aux_loss,
        }

        # Freq-sorted detached inner Matryoshka losses
        for width, inner_recon in self._freq_sorted_iterable_decode(feature_acts):
            inner_per_sample = self.mse_loss_fn(inner_recon, sae_in).sum(dim=-1)
            inner_loss = self._term_mean(inner_per_sample) * self.cfg.inner_loss_weight
            losses[f"inner_mse_loss_{width}"] = inner_loss

        total_loss = torch.stack(list(losses.values())).sum()

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )


# --- HybridLISTARef: ReferenceStyle + LISTA W_corr (best of both) ---
# Combines: index mapping freq sort + decreasing K (from Reference, great recall)
# with 5-step LISTA W_corr correction (from EnhancedLISTA, great precision)

@dataclass
class HybridLISTARefConfig(ReferenceStyleSAEConfig):
    """ReferenceStyle + LISTA W_corr correction steps."""
    n_lista_steps: int = 5  # separate from n_ista_steps to avoid confusion

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "HybridLISTARefConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [128, 512, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        ts = total_steps if total_steps > 0 else int(cfg.get("training_samples", 50000000)) // int(cfg.get("batch_size", 1024))
        k_trans = int(cfg.get("k_transition_steps", 0))
        if k_trans == 0:
            k_trans = ts
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=False,
            n_ista_steps=0,  # not used; we use n_lista_steps instead
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            term_tilt=float(cfg.get("term_tilt", 0.002)),
            detach_matryoshka=cfg.get("detach_matryoshka", True),
            use_freq_sort=cfg.get("use_freq_sort", True),
            initial_k=float(cfg.get("initial_k", 60.0)),
            k_transition_steps=k_trans,
            total_training_steps=ts,
            n_lista_steps=int(cfg.get("n_lista_steps", 5)),
            dtype="float32",
            device="cuda",
        )


class HybridLISTARef(ReferenceStyleSAE):
    """ReferenceStyle base + LISTA W_corr correction.

    Combines the reference's index mapping freq sort + decreasing K (great recall)
    with LISTA's learned W_corr correction matrix (great precision).
    """

    cfg: HybridLISTARefConfig

    def __init__(self, cfg: HybridLISTARefConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Add separate W_corr for LISTA correction steps
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, device=cfg.device, dtype=torch.float32) * 0.01
        )

    @override
    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # LISTA correction with W_corr (not W_enc)
        for _ in range(self.cfg.n_lista_steps):
            if self.cfg.rescale_acts_by_decoder_norm:
                recon = (feature_acts / W_dec_norms) @ self.W_dec
            else:
                recon = feature_acts @ self.W_dec
            residual = sae_in - recon
            delta = residual @ self.W_corr  # W_corr, not W_enc
            if W_dec_norms is not None:
                delta = delta * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * delta
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Least-Squares Magnitude LISTA + Matryoshka SAE (Agent 0, EXP10) ---
# Uses LISTA for support selection (which features fire), then replaces
# magnitudes with least-squares optimal values. Diagnostic shows LISTA has
# shrinkage=0.898 (10% magnitude underestimation). LS magnitudes are
# theoretically optimal for the selected support, fixing shrinkage and
# improving reconstruction → better decoder training → higher precision.

@dataclass
class LSMagnitudeLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA with least-squares magnitude refinement + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "ls_magnitude_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "LSMagnitudeLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class LSMagnitudeLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA+Matryoshka with least-squares magnitude refinement.

    LISTA determines which features fire (support set), then magnitudes
    are replaced with the least-squares optimal values for that support.
    This fixes the systematic shrinkage (0.898) in LISTA activations.
    """

    cfg: LSMagnitudeLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: LSMagnitudeLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # LISTA refinement for support selection
        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        # Least-squares magnitude refinement on the selected support
        k = int(self.cfg.k)
        # Get indices of active features (top-k from BatchTopK)
        topk_indices = feature_acts.abs().topk(k, dim=-1).indices  # (batch, k)

        # Gather decoder columns for active features: (batch, k, d_in)
        W_dec_S = self.W_dec[topk_indices]  # fancy indexing

        # Compute Gram matrix: (batch, k, k)
        gram = W_dec_S @ W_dec_S.transpose(-1, -2)
        # Add small regularization for numerical stability
        gram = gram + 1e-6 * torch.eye(k, device=gram.device, dtype=gram.dtype)

        # RHS: project sae_in onto active decoder columns: (batch, k)
        rhs = (W_dec_S * sae_in.unsqueeze(1)).sum(-1)

        # Solve for optimal magnitudes: (batch, k)
        optimal_mags = torch.linalg.solve(gram, rhs)

        # Ensure non-negative magnitudes (features should be positive)
        optimal_mags = F.relu(optimal_mags)

        # Scatter back to full activation tensor
        feature_acts = torch.zeros_like(feature_acts)
        feature_acts.scatter_(1, topk_indices, optimal_mags)

        # If rescale_acts_by_decoder_norm, convert back to rescaled space
        if W_dec_norms is not None:
            feature_acts = feature_acts * W_dec_norms

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Decoder-Transpose LISTA + Matryoshka SAE (Agent 0, EXP9) ---
# Uses W_dec.T for correction instead of separate W_corr.
# Zero extra parameters. In compressed sensing, the dictionary transpose
# is the theoretically optimal encoder. This creates a tight coupling:
# W_dec is optimized for BOTH reconstruction AND correction simultaneously,
# so improvements to decoder feature alignment directly improve correction.

@dataclass
class DecTransposeLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA using W_dec.T for correction + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "dec_transpose_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "DecTransposeLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class DecTransposeLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA+Matryoshka using W_dec.T for correction (no extra parameters).

    Instead of a separate W_corr matrix, uses the transpose of the decoder.
    In compressed sensing theory, the dictionary transpose is the optimal
    analysis operator. This creates a gradient coupling where W_dec gets
    optimized for BOTH reconstruction quality AND residual detection,
    which directly targets the F1 metric (decoder-GT cosine similarity).
    """

    cfg: DecTransposeLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: DecTransposeLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            # Use decoder transpose for correction (theoretically optimal)
            correction = residual @ self.W_dec.t()
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Deep Supervised LISTA + Matryoshka SAE (Agent 0, EXP8) ---
# Adds auxiliary MSE losses at each intermediate LISTA correction step.
# Standard LISTA relies on gradient backpropagating through 5 sequential
# steps + TopK masking, which can be noisy. Deep supervision provides
# "shortcut" gradient paths (analogous to skip connections in ResNets),
# giving W_corr direct training signal at every step.
# Same architecture as LISTAMatryoshka — zero extra parameters.

@dataclass
class DeepSupervisedLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA with deep supervision at each ISTA step + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3
    deep_supervision_weight: float = 0.1

    @override
    @classmethod
    def architecture(cls) -> str:
        return "deep_supervised_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "DeepSupervisedLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            deep_supervision_weight=float(cfg.get("deep_supervision_weight", 0.1)),
            dtype="float32",
            device="cuda",
        )


class DeepSupervisedLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA+Matryoshka with deep supervision at each ISTA step.

    Same architecture as LISTAMatryoshka (shared W_corr, 5 steps),
    but adds auxiliary MSE losses at each intermediate LISTA step.
    This provides direct gradient signal to W_corr at every step,
    preventing gradient degradation through sequential TopK masking.
    """

    cfg: DeepSupervisedLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: DeepSupervisedLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )
        self._intermediate_acts: list[torch.Tensor] = []

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)
        self._intermediate_acts = []

        for step in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)
            # Store intermediate acts for deep supervision (all but last)
            if step < self.cfg.n_ista_steps - 1:
                self._intermediate_acts.append(feature_acts)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        # Get base output (includes outer MSE + Matryoshka inner losses on final acts)
        output = super().training_forward_pass(step_input)

        # Add deep supervision losses at each intermediate LISTA step
        base_weight = self.cfg.deep_supervision_weight
        for step_idx, inter_acts in enumerate(self._intermediate_acts):
            recon = act_times_W_dec(
                inter_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            recon = recon + self.b_dec
            step_mse = (
                self.mse_loss_fn(recon, step_input.sae_in)
                .sum(dim=-1)
                .mean()
            )
            # Increasing weight for later steps (they should be better)
            weight = base_weight * (step_idx + 1) / self.cfg.n_ista_steps
            output.losses[f"step_{step_idx}_mse"] = weight * step_mse
            output.loss = output.loss + weight * step_mse

        self._intermediate_acts = []  # free memory
        return output


# --- Per-Step LISTA + Matryoshka SAE (Agent 2, EXP3) ---
# Full unrolled LISTA: each ISTA step gets its own learned correction
# matrix W_corr_i AND step size. In the original LISTA paper (Gregor &
# LeCun 2010), unrolling with per-step parameters outperforms shared params.
# This should be strictly better than shared-W_corr LISTA.

@dataclass
class PerStepLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """Full unrolled LISTA with per-step W_corr and step_size + Matryoshka."""

    n_ista_steps: int = 5

    @override
    @classmethod
    def architecture(cls) -> str:
        return "perstep_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "PerStepLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            dtype="float32",
            device="cuda",
        )


class PerStepLISTA(MatryoshkaBatchTopKTrainingSAE):
    """Full unrolled LISTA + Matryoshka.

    Each ISTA refinement step has its own:
    - W_corr_i: learned correction matrix for residual → latent projection
    - alpha_i: learned step size (initialized to 0.3)

    This allows each step to specialize: early steps make large corrections
    to catch obvious missed features, later steps make fine adjustments.
    Standard LISTA shares one W_corr across steps, limiting adaptivity.
    """

    cfg: PerStepLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: PerStepLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        n = cfg.n_ista_steps
        # Per-step correction matrices
        self.W_corrs = nn.ParameterList([
            nn.Parameter(
                torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
                * (1 / cfg.d_in ** 0.5)
            )
            for _ in range(n)
        ])
        # Per-step learned step sizes (log-space for stability)
        self.log_alphas = nn.Parameter(
            torch.full((n,), fill_value=-1.2,  # log(0.3) ≈ -1.2
                        dtype=self.dtype, device=self.device)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Initial encoding
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # Per-step LISTA refinement
        alphas = self.log_alphas.exp()
        for i in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corrs[i]
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + alphas[i] * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Low-Rank LISTA + Matryoshka SAE (Agent 2, EXP5) ---
# W_corr = W_enc + A @ B (low-rank correction). Instead of a full separate
# W_corr (3.15M params), the correction is parameterized as W_enc plus a
# low-rank residual A@B. This keeps W_corr structurally close to W_enc
# (preventing overfitting) while allowing targeted specialization for
# residual feature detection. Inspired by LoRA's success in fine-tuning.

@dataclass
class LowRankLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """Low-Rank LISTA encoder + Matryoshka nested reconstruction losses."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3
    corr_rank: int = 64

    @override
    @classmethod
    def architecture(cls) -> str:
        return "lowrank_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "LowRankLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            corr_rank=int(cfg.get("corr_rank", 64)),
            dtype="float32",
            device="cuda",
        )


class LowRankLISTA(MatryoshkaBatchTopKTrainingSAE):
    """Matryoshka SAE with Low-Rank LISTA encoder.

    Instead of a full separate W_corr (768×4096), parameterizes the correction
    matrix as W_corr = W_enc + A @ B, where A (768×r) and B (r×4096) form a
    low-rank deviation from W_enc.

    Why: LISTA (full W_corr) only improves 0.4% over ISTA (reusing W_enc),
    and PerStepLISTA (5× more params) was WORSE due to overfitting.
    This suggests W_corr doesn't need to differ much from W_enc.
    Low-rank parameterization constrains the deviation while allowing
    targeted specialization for residual feature detection.
    """

    cfg: LowRankLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: LowRankLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        r = cfg.corr_rank
        # Low-rank factors for W_corr = W_enc + A @ B
        self.corr_A = nn.Parameter(
            torch.randn(cfg.d_in, r, dtype=self.dtype, device=self.device)
            * (1 / (cfg.d_in * r) ** 0.25)
        )
        self.corr_B = nn.Parameter(
            torch.zeros(r, cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Initial encoding
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # Compute effective W_corr = W_enc + A @ B
        W_corr = self.W_enc + self.corr_A @ self.corr_B

        # LISTA refinement steps
        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Curriculum LISTA + Matryoshka SAE (Agent 2, EXP7) ---
# Training curriculum: delay Matryoshka inner losses for the first
# warmup_frac of training. This lets features develop freely before
# anti-absorption pressure kicks in. The hypothesis is that early
# Matryoshka losses constrain feature development, preventing the
# model from first finding the right feature directions.

@dataclass
class CurriculumLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA + Matryoshka with curriculum schedule for inner losses."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3
    warmup_frac: float = 0.3  # fraction of training before inner losses start
    total_steps: int = 48828  # filled in by from_dict

    @override
    @classmethod
    def architecture(cls) -> str:
        return "curriculum_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "CurriculumLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            warmup_frac=float(cfg.get("warmup_frac", 0.3)),
            total_steps=total_steps,
            dtype="float32",
            device="cuda",
        )


class CurriculumLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA + Matryoshka with curriculum: inner losses ramp up during training.

    For the first warmup_frac of training, inner Matryoshka losses are zero
    (training like vanilla LISTA without anti-absorption). Then inner losses
    linearly ramp up to full weight over the next warmup_frac.

    This gives features time to find the right directions before the
    Matryoshka constraint enforces that subsets reconstruct well independently.
    """

    cfg: CurriculumLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: CurriculumLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )
        self._step_counter = 0

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        # Get base output (includes outer MSE loss + aux loss from BatchTopK parent)
        # We call the grandparent's training_forward_pass to avoid double-adding inner losses
        base_output = BatchTopKTrainingSAE.training_forward_pass(self, step_input)

        # Compute curriculum weight for inner Matryoshka losses
        step = step_input.n_training_steps
        warmup_start = int(self.cfg.total_steps * self.cfg.warmup_frac)
        warmup_end = int(self.cfg.total_steps * self.cfg.warmup_frac * 2)

        if step < warmup_start:
            inner_weight = 0.0
        elif step < warmup_end:
            inner_weight = (step - warmup_start) / max(warmup_end - warmup_start, 1)
        else:
            inner_weight = 1.0

        # Add weighted inner Matryoshka losses
        for width, inner_reconstruction in self._iterable_decode(
            base_output.feature_acts, include_outer_loss=False
        ):
            inner_mse_loss = (
                self.mse_loss_fn(inner_reconstruction, step_input.sae_in)
                .sum(dim=-1)
                .mean()
            )
            weighted_loss = inner_weight * inner_mse_loss
            base_output.losses[f"inner_mse_loss_{width}"] = weighted_loss
            base_output.loss = base_output.loss + weighted_loss

        return base_output


# --- Decay-Step LISTA + Matryoshka SAE (Agent 2, EXP9) ---
# Fixed decreasing step sizes across ISTA iterations.
# Unlike PerStepLISTA (which learned per-step sizes and overfit),
# this uses a fixed geometric decay schedule: no extra parameters.
# Large steps early for coarse correction, small steps for fine-tuning.

@dataclass
class DecayStepLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA with decaying step sizes + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.5  # starting step size (decays across steps)
    step_decay: float = 0.6  # geometric decay factor per step

    @override
    @classmethod
    def architecture(cls) -> str:
        return "decaystep_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "DecayStepLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.5)),
            step_decay=float(cfg.get("step_decay", 0.6)),
            dtype="float32",
            device="cuda",
        )


class DecayStepLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA + Matryoshka with geometrically decaying step sizes.

    Step sizes: [s, s*d, s*d^2, s*d^3, s*d^4] where s=ista_step_size, d=step_decay.
    Default: [0.5, 0.3, 0.18, 0.108, 0.065] — large initial corrections, fine later.
    No extra learnable parameters (unlike PerStepLISTA).
    """

    cfg: DecayStepLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: DecayStepLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )
        # Precompute fixed step schedule
        self.step_schedule = [
            cfg.ista_step_size * (cfg.step_decay ** i)
            for i in range(cfg.n_ista_steps)
        ]

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        for i in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.step_schedule[i] * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Extragradient LISTA + Matryoshka SAE (Agent 2, EXP8) ---
# Based on ELISTA (Liu et al., AAAI 2021): uses extragradient method
# for each LISTA correction step. Instead of correcting based on the
# current residual, takes a trial step, computes the residual at the
# trial point, then uses THAT correction for the actual update.
# This "look-ahead" correction uses information from the future state,
# leading to better convergence in optimization theory.

@dataclass
class ExtragradientLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """Extragradient LISTA + Matryoshka."""

    n_ista_steps: int = 3  # fewer steps since each step is 2x compute
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "extragradient_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "ExtragradientLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 3)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class ExtragradientLISTA(MatryoshkaBatchTopKTrainingSAE):
    """Matryoshka SAE with Extragradient LISTA encoder.

    Each correction step uses the extragradient method:
    1. Trial step: trial_pre = hidden_pre + step * (residual @ W_corr)
    2. Trial activation: trial_acts = TopK(trial_pre)
    3. Trial residual: trial_res = sae_in - trial_acts @ W_dec
    4. Actual correction: hidden_pre = hidden_pre + step * (trial_res @ W_corr)

    This computes the correction at the "look-ahead" point rather than
    the current point, giving better convergence. Each step costs 2x
    compute but may converge in fewer steps.
    """

    cfg: ExtragradientLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: ExtragradientLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)
        step = self.cfg.ista_step_size

        for _ in range(self.cfg.n_ista_steps):
            # Current residual and correction
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms

            # Trial step
            trial_pre = hidden_pre + step * correction
            trial_acts = self.activation_fn(trial_pre)

            # Trial residual and correction (look-ahead)
            trial_recon = act_times_W_dec(
                trial_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            trial_residual = sae_in - trial_recon
            trial_correction = trial_residual @ self.W_corr
            if W_dec_norms is not None:
                trial_correction = trial_correction * W_dec_norms

            # Actual update using look-ahead correction
            hidden_pre = hidden_pre + step * trial_correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Deep Encoder Matryoshka SAE (Agent 0) ---
# A 2-layer encoder (MLP) gives much more capacity to untangle
# 16k superposed features than a single linear layer in 768d.
# Combined with Matryoshka nested losses for anti-absorption.


@dataclass
class DeepEncoderMatryoshkaConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """Matryoshka SAE with a deeper (2-layer) encoder."""

    encoder_hidden_dim: int = 1536  # 2x d_in for more capacity

    @override
    @classmethod
    def architecture(cls) -> str:
        return "deep_encoder_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "DeepEncoderMatryoshkaConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            encoder_hidden_dim=int(cfg.get("encoder_hidden_dim", 1536)),
            dtype="float32",
            device="cuda",
        )


class DeepEncoderMatryoshka(MatryoshkaBatchTopKTrainingSAE):
    """Matryoshka SAE with a 2-layer MLP encoder.

    Standard SAEs use a single linear encoder: z = x @ W_enc + b_enc
    This limits feature detection capacity in extreme superposition.

    This SAE uses a 2-layer encoder:
      hidden = GELU(x @ W_enc1 + b_enc1)   # [batch, d_hidden]
      z = hidden @ W_enc2 + b_enc2          # [batch, d_sae]

    The deeper encoder can learn nonlinear feature detectors that better
    disentangle 16k features from 768 dimensions. The decoder stays linear.
    """

    cfg: DeepEncoderMatryoshkaConfig  # type: ignore[assignment]

    def __init__(self, cfg: DeepEncoderMatryoshkaConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        d_hidden = cfg.encoder_hidden_dim
        # Replace the linear encoder with a 2-layer MLP
        self.W_enc1 = nn.Parameter(
            torch.randn(cfg.d_in, d_hidden, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )
        self.b_enc1 = nn.Parameter(
            torch.zeros(d_hidden, dtype=self.dtype, device=self.device)
        )
        self.W_enc2_deep = nn.Parameter(
            torch.randn(d_hidden, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / d_hidden ** 0.5)
        )
        # b_enc from parent is reused as the final bias

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # 2-layer encoder
        hidden = torch.nn.functional.gelu(sae_in @ self.W_enc1 + self.b_enc1)
        hidden_pre = self.hook_sae_acts_pre(hidden @ self.W_enc2_deep + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        feature_acts = self.activation_fn(hidden_pre)
        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- FISTA + LISTA + Matryoshka SAE (Agent 1, EXP5) ---
# Momentum-accelerated LISTA: applies Nesterov momentum (FISTA principle)
# to the ISTA correction steps. Momentum accelerates convergence by
# extrapolating in the direction of recent updates, recovering more
# features in the same number of iterations.

@dataclass
class FISTAMatryoshkaConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """FISTA (momentum-accelerated LISTA) + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "fista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "FISTAMatryoshkaConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class FISTAMatryoshka(MatryoshkaBatchTopKTrainingSAE):
    """Matryoshka SAE with momentum-accelerated Learned ISTA encoder.

    Adds Nesterov momentum to LISTA correction steps (FISTA principle).
    In optimization, momentum dramatically accelerates convergence by
    extrapolating in the direction of recent updates. Applied to SAE
    encoding, this means faster feature recovery — each step builds on
    the momentum of previous corrections rather than starting fresh.

    The momentum schedule follows the classical FISTA sequence:
    t_{i+1} = (1 + sqrt(1 + 4*t_i^2)) / 2
    beta_i = (t_i - 1) / t_{i+1}
    """

    cfg: FISTAMatryoshkaConfig  # type: ignore[assignment]

    def __init__(self, cfg: FISTAMatryoshkaConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Initial encoding
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # FISTA: momentum-accelerated LISTA refinement
        t = 1.0
        hidden_pre_prev = hidden_pre.clone()

        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms

            # Standard LISTA update
            hidden_pre_new = hidden_pre + self.cfg.ista_step_size * correction

            # Nesterov momentum
            t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
            beta = (t - 1.0) / t_new

            # Extrapolate with momentum
            hidden_pre = hidden_pre_new + beta * (hidden_pre_new - hidden_pre_prev)

            hidden_pre_prev = hidden_pre_new
            t = t_new

            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- LISTA + Matryoshka with W_enc-initialized W_corr (Agent 3, EXP6) ---
# Standard LISTA initializes W_corr randomly. But ISTA (reusing W_enc) already
# achieves 0.9175, showing W_enc is a good correction matrix. Random init
# forces W_corr to rediscover this from scratch. By initializing W_corr = W_enc.clone(),
# LISTA starts from the ISTA solution and learns only the deviation needed.
# Also adds a learned scalar step_size for fine-tuning the correction magnitude.

@dataclass
class WarmLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA with W_enc-initialized W_corr + learned step_size + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "warm_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "WarmLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class WarmLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA with warm-initialized W_corr from W_enc.

    Key insight: ISTA (reusing W_enc as correction matrix) achieves 0.9175,
    while LISTA (random W_corr) achieves 0.9215. The marginal improvement
    suggests W_corr only needs to differ slightly from W_enc.

    By initializing W_corr = W_enc.clone(), we:
    1. Start from a good correction matrix (the ISTA solution)
    2. Allow gradient descent to find the optimal small deviation
    3. Avoid wasting training capacity rediscovering W_enc's structure

    Also learns a scalar step_size (initialized to 0.3) for optimal
    correction magnitude without per-step parameterization overhead.
    """

    cfg: WarmLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: WarmLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Initialize W_corr from W_enc (warm start)
        self.W_corr = nn.Parameter(self.W_enc.data.clone())
        # Learned step size (log-space for positivity)
        import math
        self.log_step_size = nn.Parameter(
            torch.tensor(math.log(cfg.ista_step_size),
                         dtype=self.dtype, device=self.device)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        step_size = self.log_step_size.exp()

        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Wide-Intermediate LISTA + Matryoshka SAE (Agent 1, EXP8) ---
# Uses a wider k for intermediate LISTA correction steps (better residual
# computation by capturing more features) and the standard k only for the
# final output (maintaining precision). This separates the "exploration" k
# (intermediate) from the "output" k (final).

from sae_lens.saes.topk_sae import TopK


@dataclass
class WideIntermediateLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA with wider k for intermediate steps + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3
    k_intermediate: int = 50

    @override
    @classmethod
    def architecture(cls) -> str:
        return "wide_intermediate_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "WideIntermediateLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            k_intermediate=int(cfg.get("k_intermediate", 50)),
            dtype="float32",
            device="cuda",
        )


class WideIntermediateLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA + Matryoshka with wider k for intermediate correction steps.

    Standard LISTA uses k=25 at every step. But intermediate steps only need
    feature activations to compute residuals. Using wider k (e.g., 50) for
    intermediate steps captures more features in reconstruction, producing
    more accurate residuals that reveal truly missed features.

    Only the final step uses standard BatchTopK k=25, maintaining precision.
    Intermediate steps use per-sample TopK with k_intermediate.
    """

    cfg: WideIntermediateLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: WideIntermediateLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )
        self.topk_intermediate = TopK(cfg.k_intermediate)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.topk_intermediate(hidden_pre)

        for i in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction

            if i < self.cfg.n_ista_steps - 1:
                feature_acts = self.topk_intermediate(hidden_pre)
            else:
                feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- SoftLISTA + Matryoshka SAE (Agent 1, EXP9) ---
# Uses differentiable soft thresholding for intermediate LISTA steps instead of
# hard TopK. Only the final step uses BatchTopK. This is more faithful to the
# original LISTA formulation (Gregor & LeCun 2010) where the proximal operator
# is soft thresholding, not hard top-k. Soft thresholding provides smooth
# gradients through intermediate steps, potentially improving W_corr learning.

@dataclass
class SoftLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA with soft thresholding intermediate steps + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3
    initial_threshold: float = 0.1

    @override
    @classmethod
    def architecture(cls) -> str:
        return "soft_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "SoftLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            initial_threshold=float(cfg.get("initial_threshold", 0.1)),
            dtype="float32",
            device="cuda",
        )


class SoftLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA + Matryoshka with soft thresholding for intermediate steps.

    Standard LISTA uses BatchTopK at every step, but BatchTopK is non-differentiable
    (zero gradient for inactive features). This means intermediate correction steps
    get no gradient signal for features that were just below the threshold.

    SoftLISTA uses soft thresholding (shrinkage operator) for intermediate steps:
        S_λ(x) = sign(x) * max(|x| - λ, 0)
    This is the proximal operator for L1 regularization — the theoretically
    correct activation for ISTA. It's differentiable everywhere except x=±λ,
    giving smooth gradients through the unrolled iteration.

    Only the final step uses BatchTopK for hard sparsity (k=25).
    The threshold λ is learned per-feature.
    """

    cfg: SoftLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: SoftLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )
        # Learned per-feature soft threshold (initialized positive)
        self.threshold = nn.Parameter(
            torch.full((cfg.d_sae,), cfg.initial_threshold,
                       dtype=self.dtype, device=self.device)
        )

    def soft_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """Soft thresholding (shrinkage) operator: S_λ(x) = sign(x) * max(|x| - λ, 0)."""
        lam = self.threshold.abs()  # ensure non-negative
        return torch.sign(x) * F.relu(x.abs() - lam)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        # Intermediate steps: soft thresholding (differentiable)
        feature_acts = self.soft_threshold(hidden_pre)

        for i in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction

            if i < self.cfg.n_ista_steps - 1:
                # Intermediate: soft thresholding for smooth gradients
                feature_acts = self.soft_threshold(hidden_pre)
            else:
                # Final step: hard BatchTopK for exact sparsity
                feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Magnitude Refine LISTA + Matryoshka SAE (Agent 2, EXP11) ---
# Standard LISTA selects WHICH features fire (support set) via iterative
# TopK refinement. But TopK couples two decisions: selection AND magnitude.
# After LISTA determines the support set, this adds a coordinate descent
# step that refines only the magnitudes of selected features WITHOUT
# re-applying TopK. This decouples support selection from magnitude
# estimation, avoiding the TopK discontinuity that killed FISTA/extragradient.

@dataclass
class MagnitudeRefineLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA + post-TopK magnitude refinement + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3
    refine_gamma: float = 0.3
    n_refine_steps: int = 1

    @override
    @classmethod
    def architecture(cls) -> str:
        return "magnitude_refine_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "MagnitudeRefineLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            refine_gamma=float(cfg.get("refine_gamma", 0.3)),
            n_refine_steps=int(cfg.get("n_refine_steps", 1)),
            dtype="float32",
            device="cuda",
        )


class MagnitudeRefineLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA + coordinate descent magnitude refinement + Matryoshka.

    After LISTA determines which features are active (support set),
    performs coordinate descent steps on the magnitudes of active features
    only. This refines reconstruction quality without changing which
    features fire, avoiding the TopK discontinuity issue.
    """

    cfg: MagnitudeRefineLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: MagnitudeRefineLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # Standard LISTA loop for support selection
        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        # Coordinate descent magnitude refinement (support set is FIXED)
        support_mask = (feature_acts != 0).float()
        for _ in range(self.cfg.n_refine_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            # Project residual onto decoder directions for magnitude update
            mag_correction = residual @ self.W_dec.t()
            if W_dec_norms is not None:
                mag_correction = mag_correction / (W_dec_norms + 1e-8)
            # Only update active features (preserve sparsity)
            feature_acts = feature_acts + self.cfg.refine_gamma * support_mask * mag_correction

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Tied Encoder LISTA + Matryoshka SAE (Agent 2, EXP12) ---
# Uses W_dec.T for INITIAL encoding instead of separate W_enc.
# Different from DecTransposeLISTA (Agent 0) which uses W_dec.T for CORRECTION.
# Rationale: decoder columns ARE the feature directions. Using W_dec.T for
# initial encoding means we project onto exact feature directions, then
# W_corr handles residual correction. This ties encoding quality to
# decoder quality — the same directions used for GT matching (cosine sim)
# are used for feature detection.

@dataclass
class TiedEncoderLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA with W_dec.T initial encoding + separate W_corr + Matryoshka."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "tied_encoder_lista_matryoshka"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "TiedEncoderLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class TiedEncoderLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA with W_dec.T for initial encoding + Matryoshka.

    The initial encoding uses W_dec.T instead of W_enc. Since decoder
    columns are the feature directions matched to GT features (via cosine
    similarity), projecting inputs onto these directions directly ties
    encoding quality to decoder quality. W_corr handles the harder task
    of residual correction.

    This creates gradient coupling: W_dec is optimized for BOTH
    reconstruction AND initial feature detection, ensuring decoder columns
    represent directions useful for both tasks.
    """

    cfg: TiedEncoderLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: TiedEncoderLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)

        # Use W_dec.T for initial encoding (tied to decoder directions)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_dec.t() + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        # LISTA refinement with separate W_corr
        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre


# --- Weighted Matryoshka LISTA (Agent 2, EXP14) ---
# Standard Matryoshka weights all inner losses equally. But smaller widths have
# much worse reconstruction (higher MSE), creating disproportionately strong
# gradients that may distort feature layout. Scale inner losses by width/d_sae.

@dataclass
class WeightedMatryoshkaLISTAConfig(MatryoshkaBatchTopKTrainingSAEConfig):
    """LISTA + width-proportional Matryoshka loss weighting."""

    n_ista_steps: int = 5
    ista_step_size: float = 0.3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "weighted_matryoshka_lista"

    @classmethod
    def from_dict(cls, cfg: dict, total_steps: int = 0) -> "WeightedMatryoshkaLISTAConfig":
        d_sae = int(cfg.get("d_sae", 4096))
        default_widths = [256, 512, 1024, 2048, d_sae]
        widths = cfg.get("matryoshka_widths", default_widths)
        return cls(
            d_in=D_IN,
            d_sae=d_sae,
            k=float(cfg.get("k", 25)),
            matryoshka_widths=widths,
            use_matryoshka_aux_loss=cfg.get("use_matryoshka_aux_loss", True),
            n_ista_steps=int(cfg.get("n_ista_steps", 5)),
            ista_step_size=float(cfg.get("ista_step_size", 0.3)),
            dtype="float32",
            device="cuda",
        )


class WeightedMatryoshkaLISTA(MatryoshkaBatchTopKTrainingSAE):
    """LISTA + Matryoshka with width-proportional inner loss weights.

    Inner Matryoshka losses are scaled by (width / d_sae), so smaller widths
    contribute less gradient. This prevents small-width losses from dominating
    and distorting the feature layout.
    """

    cfg: WeightedMatryoshkaLISTAConfig  # type: ignore[assignment]

    def __init__(self, cfg: WeightedMatryoshkaLISTAConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.W_corr = nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
            * (1 / cfg.d_in ** 0.5)
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        W_dec_norms = None
        if self.cfg.rescale_acts_by_decoder_norm:
            W_dec_norms = self.W_dec.norm(dim=-1)
            hidden_pre = hidden_pre * W_dec_norms

        feature_acts = self.activation_fn(hidden_pre)

        for _ in range(self.cfg.n_ista_steps):
            recon = act_times_W_dec(
                feature_acts, self.W_dec, self.cfg.rescale_acts_by_decoder_norm
            )
            residual = sae_in - recon
            correction = residual @ self.W_corr
            if W_dec_norms is not None:
                correction = correction * W_dec_norms
            hidden_pre = hidden_pre + self.cfg.ista_step_size * correction
            feature_acts = self.activation_fn(hidden_pre)

        feature_acts = self.hook_sae_acts_post(feature_acts)
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        # Get base output (includes full-width MSE + aux loss)
        base_output = super().training_forward_pass(step_input)

        # Remove the default equally-weighted inner losses and replace with weighted ones
        d_sae = self.cfg.d_sae
        for width, inner_reconstruction in self._iterable_decode(
            base_output.feature_acts, include_outer_loss=False
        ):
            default_key = f"inner_mse_loss_{width}"
            if default_key in base_output.losses:
                old_loss = base_output.losses[default_key]
                base_output.loss = base_output.loss - old_loss

            inner_mse_loss = (
                self.mse_loss_fn(inner_reconstruction, step_input.sae_in)
                .sum(dim=-1)
                .mean()
            )
            weight = width / d_sae
            weighted_loss = weight * inner_mse_loss
            base_output.losses[default_key] = weighted_loss
            base_output.loss = base_output.loss + weighted_loss

        return base_output
