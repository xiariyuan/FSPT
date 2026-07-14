from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EncoderConfig:
    type: str = "simple"
    in_channels: int = 3
    base_dim: int = 64
    out_dim: int = 128
    backbone_name: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    pretrained_path: str = ""
    train_backbone: bool = False
    out_indices: Tuple[int, ...] = (1, 2, 3)
    fuse_mode: str = "sum"


@dataclass
class LocalMatcherConfig:
    radius: int = 4
    temperature: float = 0.07


@dataclass
class GlobalRelocatorConfig:
    type: str = "memory_similarity"
    topk: int = 5
    temperature: float = 0.07
    memory_size: int = 8
    hidden_dim: int = 256
    position_bias_strength: float = 0.0
    position_bias_sigma: float = 0.25


@dataclass
class PosteriorFusionConfig:
    hidden_dim: int = 64
    type_embed_dim: int = 8
    min_visibility: float = 0.0
    max_offset: float = 0.08
    prior_residual_mix: float = 0.7


@dataclass
class TrackingConfig:
    feature_stride: int = 4
    forward_only: bool = True
    visibility_update_threshold: float = 0.5
    template_momentum: float = 0.8
    anchor_mix: float = 0.35
    global_use_margin: float = 0.05
    global_refine_radius: int = 2
    global_refine_center: str = "expected"
    global_output_mode: str = "top1_refine"
    global_refine_template_mode: str = "point"
    global_refine_template_radius: int = 1
    memory_write_mode: str = "append_visible"
    safe_write_agreement_px: float = 6.0
    commit_mode: str = "heuristic"
    global_commit_consistency_px: float = 10.0
    global_commit_min_quality: float = 0.10
    long_occ_min_frames: int = 3
    long_occ_recovery_window: int = 1
    selector_threshold: float = 0.5
    selector_target_margin: float = 0.01
    candidate_routing_mode: str = "flat"
    candidate_gate_threshold: float = 0.5
    candidate_gate_positive_weight: float = 1.0
    candidate_gate_focal_gamma: float = 0.0
    candidate_gate_margin: float = 0.0
    candidate_gate_margin_weight: float = 0.0
    candidate_rank_soft_temperature: float = 0.0
    commit_threshold: float = 0.5
    commit_target_margin: float = 0.01
    deferred_commit_min_quality: float = 0.10
    deferred_commit_confirm_px: float = 12.0
    deferred_commit_reconfirm_px: float = 12.0
    deferred_stage_min_disagreement_px: float = 8.0
    enable_multi_hypothesis_diagnostics: bool = False
    belief_transition_sigma: float = 0.08
    belief_prior_strength: float = 1.0
    belief_evidence_temperature: float = 1.0
    belief_birth_mass: float = 0.05
    belief_collapse_min_top1_weight: float = 0.65
    belief_collapse_max_normalized_entropy: float = 0.45
    belief_collapse_min_top1_margin: float = 0.20


@dataclass
class MMPTrackerConfig:
    variant: str = "posterior"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    local_matcher: LocalMatcherConfig = field(default_factory=LocalMatcherConfig)
    global_relocator: GlobalRelocatorConfig = field(default_factory=GlobalRelocatorConfig)
    posterior_fusion: PosteriorFusionConfig = field(default_factory=PosteriorFusionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
