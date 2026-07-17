"""Comparator-only training and evaluation for frozen Route-D CMCP P0h tokens."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from datasets.metrics import compute_tapvid_metrics

from .routeD_cmcp_pairwise_cache import (
    CMCP_PAIRWISE_DYNAMIC_SUMMARY_DIM,
    CMCP_PAIRWISE_STATIC_TOKEN_DIM,
    inject_dynamic_summary,
    load_complete_pairwise_token_index,
    verify_pairwise_token_artifact,
)
from .routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
)
from .routeD_kubric_cache import file_sha256
from .routeD_musr_training import (
    _tracks_from_xy,
    _visible_error_stats,
    paired_video_bootstrap_ci,
    state_dict_sha256,
    visible_post_query_mask,
)


@dataclass(frozen=True)
class PairwiseSafetyLossConfig:
    utility_bce_weight: float = 1.0
    risk_bce_weight: float = 0.5
    pairwise_preference_weight: float = 1.0
    abstention_weight: float = 1.0
    no_harm_weight: float = 1.0
    harmful_multiplier: float = 10.0
    pairwise_margin: float = 0.25
    catastrophe_threshold_px: float = 16.0
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class StaticTokenNormalization:
    mean: torch.Tensor
    std: torch.Tensor

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass(frozen=True)
class PairwiseTokenVideo:
    source_index: int
    video_name: str
    candidate_tokens: torch.Tensor
    candidate_coords_xy_px: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_valid_mask: torch.Tensor
    frame_valid: torch.Tensor
    native_coords_xy_px: torch.Tensor
    tensors: Mapping[str, torch.Tensor]
    sidecar: str
    base_sidecar: str


def load_pairwise_token_video(row: Mapping[str, Any]) -> PairwiseTokenVideo:
    sidecar=Path(str(row["sidecar"])).resolve()
    if file_sha256(sidecar)!=str(row["sidecar_sha256"]):
        raise ValueError("pairwise token sidecar hash mismatch")
    artifact=verify_pairwise_token_artifact(
        sidecar,
        expected_generator_state_sha256=str(row["generator_model_state_sha256"]),
        expected_base_sidecar_sha256=str(row["base_sidecar_sha256"]),
    )
    base_path=Path(str(artifact["provenance"]["base_sidecar"])).resolve()
    if file_sha256(base_path)!=str(row["base_sidecar_sha256"]):
        raise ValueError("base sidecar hash mismatch")
    base=torch.load(base_path,map_location="cpu",weights_only=False)["tensors"]
    t=artifact["tensors"]
    return PairwiseTokenVideo(
        source_index=int(row["source_index"]),
        video_name=str(row["sample_identity"]["video_name"]),
        candidate_tokens=t["candidate_tokens"],
        candidate_coords_xy_px=t["candidate_coords_xy_px"],
        candidate_scores=t["candidate_scores"],
        candidate_valid_mask=t["candidate_valid_mask"],
        frame_valid=t["frame_valid"],
        native_coords_xy_px=t["native_coords_xy_px"],
        tensors=base,
        sidecar=str(sidecar),
        base_sidecar=str(base_path),
    )


def compute_static_token_normalization(index_path: str | Path) -> StaticTokenNormalization:
    index=load_complete_pairwise_token_index(index_path,expected_partition="fit")
    total=torch.zeros(CMCP_PAIRWISE_STATIC_TOKEN_DIM,dtype=torch.float64)
    total2=torch.zeros_like(total); count=0
    for row in index["videos"]:
        video=load_pairwise_token_video(row)
        mask=video.candidate_valid_mask & video.frame_valid.unsqueeze(-1)
        values=video.candidate_tokens[...,:CMCP_PAIRWISE_STATIC_TOKEN_DIM][mask].double()
        total += values.sum(dim=0); total2 += values.square().sum(dim=0); count += int(values.shape[0])
    if count<=0: raise ValueError("no valid static tokens")
    mean=(total/count).float(); variance=(total2/count-total.square()/(count*count)).clamp_min(0)
    std=variance.sqrt().float().clamp_min(1e-5)
    return StaticTokenNormalization(mean,std)


def normalize_static_tokens(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    normalization: StaticTokenNormalization,
) -> torch.Tensor:
    static=(tokens[...,:CMCP_PAIRWISE_STATIC_TOKEN_DIM]-normalization.mean.to(tokens.device,tokens.dtype))/normalization.std.to(tokens.device,tokens.dtype)
    result=torch.cat([static,tokens[...,-CMCP_PAIRWISE_DYNAMIC_SUMMARY_DIM:]],dim=-1)
    return torch.where(valid.unsqueeze(-1),result,torch.zeros_like(result))


def _utility_from_error(error_px: torch.Tensor, config: CMCPLocalSafetyConfig) -> torch.Tensor:
    thresholds=torch.tensor(config.thresholds_px,device=error_px.device,dtype=error_px.dtype)
    weights=torch.tensor(config.utility_weights,device=error_px.device,dtype=error_px.dtype); weights=weights/weights.sum()
    return ((error_px[...,None] <= thresholds)*weights).sum(dim=-1)


def pairwise_safety_frame_loss(
    output: Mapping[str,torch.Tensor],
    candidate_coords_xy_px: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    gt_xy_px: torch.Tensor,
    supervised: torch.Tensor,
    model_config: CMCPLocalSafetyConfig,
    loss_config: PairwiseSafetyLossConfig,
) -> dict[str,torch.Tensor]:
    error=torch.linalg.vector_norm(candidate_coords_xy_px-gt_xy_px[:,None],dim=-1)
    thresholds=torch.tensor(model_config.thresholds_px,device=error.device,dtype=error.dtype)
    correctness=(error[...,None] <= thresholds).float()
    valid_supervised=candidate_valid_mask & supervised[:,None]
    mask5=valid_supervised.unsqueeze(-1).float()
    utility_bce=F.binary_cross_entropy_with_logits(output["utility_logits"],correctness,reduction="none")
    utility_bce=(utility_bce*mask5).sum()/mask5.sum().clamp_min(1)
    risk_target=(error>=float(loss_config.catastrophe_threshold_px)).float()
    mask=valid_supervised.float()
    risk_bce=F.binary_cross_entropy_with_logits(output["risk_logit"],risk_target,reduction="none")
    risk_bce=(risk_bce*mask).sum()/mask.sum().clamp_min(1)
    utility=_utility_from_error(error,model_config)
    native_utility=utility[:,:1]
    better=(utility[:,1:] > native_utility+1e-6).float()
    harmful=utility[:,1:] + 1e-6 < native_utility
    pair_mask=(candidate_valid_mask[:,1:] & supervised[:,None]).float()
    pair_weight=torch.where(harmful,torch.full_like(pair_mask,float(loss_config.harmful_multiplier)),torch.ones_like(pair_mask))*pair_mask
    pair_logit=output["preference_logit"][:,1:]-output["preference_logit"][:,:1]
    pair_bce=F.binary_cross_entropy_with_logits(pair_logit,better,reduction="none")
    pair_bce=(pair_bce*pair_weight).sum()/pair_weight.sum().clamp_min(1)
    has_better=((utility[:,1:] > native_utility+1e-6) & candidate_valid_mask[:,1:]).any(dim=1)
    abstain_target=(~has_better).float()
    abstention=F.binary_cross_entropy_with_logits(output["abstention_logit"],abstain_target,reduction="none")
    abstention=(abstention*supervised.float()).sum()/supervised.float().sum().clamp_min(1)
    score_margin=output["candidate_score"][:,1:]-output["candidate_score"][:,:1]+float(loss_config.pairwise_margin)
    no_harm=F.relu(score_margin)*harmful.float()*pair_mask*float(loss_config.harmful_multiplier)
    no_harm=no_harm.sum()/(harmful.float()*pair_mask*float(loss_config.harmful_multiplier)).sum().clamp_min(1)
    total=(float(loss_config.utility_bce_weight)*utility_bce + float(loss_config.risk_bce_weight)*risk_bce +
           float(loss_config.pairwise_preference_weight)*pair_bce + float(loss_config.abstention_weight)*abstention +
           float(loss_config.no_harm_weight)*no_harm)
    return {"loss":total,"utility_bce":utility_bce,"risk_bce":risk_bce,"pairwise_bce":pair_bce,"abstention_bce":abstention,"no_harm":no_harm}


def _update_summary(
    output: Mapping[str,torch.Tensor],
    candidate_coords: torch.Tensor,
    frame_valid: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    selected=output["selected_candidate_index"].detach()
    score=output["candidate_score"].detach()
    selected_score=score.gather(1,selected[:,None]).squeeze(1); native_score=score[:,0]
    selected_coord=candidate_coords.gather(1,selected[:,None,None].expand(-1,1,2)).squeeze(1)
    displacement=torch.linalg.vector_norm(selected_coord-candidate_coords[:,0],dim=-1)/255.0
    current=torch.stack([(selected>0).float(),selected.float()/float(max(candidate_coords.shape[1]-1,1)),selected_score-native_score,displacement],dim=-1)
    return torch.where(frame_valid[:,None],current,previous).detach()


def predict_pairwise_video(
    model: CMCPLocalPairwiseSafetyComparator,
    video: PairwiseTokenVideo,
    normalization: StaticTokenNormalization,
    *,
    device: str,
    point_batch_size: int,
) -> tuple[dict[str,torch.Tensor],dict[str,float]]:
    all_selected=[]; all_index=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,video.candidate_tokens.shape[0],point_batch_size):
            end=min(video.candidate_tokens.shape[0],start+point_batch_size)
            tokens=video.candidate_tokens[start:end].to(device)
            coords=video.candidate_coords_xy_px[start:end].to(device)
            valid=video.candidate_valid_mask[start:end].to(device)
            frame_valid=video.frame_valid[start:end].to(device)
            tokens=normalize_static_tokens(tokens,valid,normalization)
            summary=torch.zeros(end-start,4,device=device)
            selected_frames=[]; index_frames=[]
            for frame in range(tokens.shape[1]):
                frame_tokens=inject_dynamic_summary(tokens[:,frame],summary)
                output=model(frame_tokens,valid[:,frame],coords[:,frame])
                summary=_update_summary(output,coords[:,frame],frame_valid[:,frame],summary)
                selected_frames.append(output["selected_coord_xy_px"].cpu()); index_frames.append(output["selected_candidate_index"].cpu())
            all_selected.append(torch.stack(selected_frames,dim=1)); all_index.append(torch.stack(index_frames,dim=1))
    selected_xy=torch.cat(all_selected); selected_index=torch.cat(all_index)
    gt_xy=video.tensors["gt_tracks_yx"][...,[1,0]].float()*255.0
    error=torch.linalg.vector_norm(video.candidate_coords_xy_px-gt_xy[...,None,:],dim=-1).masked_fill(~video.candidate_valid_mask,float("inf"))
    oracle_index=error.argmin(dim=-1)
    oracle_xy=video.candidate_coords_xy_px.gather(2,oracle_index[...,None,None].expand(-1,-1,1,2)).squeeze(2)
    mask=visible_post_query_mask(video.tensors)
    native_error=torch.linalg.vector_norm(video.native_coords_xy_px-gt_xy,dim=-1)
    selected_error=torch.linalg.vector_norm(selected_xy-gt_xy,dim=-1)
    native_utility=_utility_from_error(native_error,model.config); selected_utility=_utility_from_error(selected_error,model.config)
    oracle_utility=_utility_from_error(error.min(dim=-1).values,model.config)
    nonnative=selected_index>0; harmful=nonnative & (selected_utility+1e-6<native_utility)
    available=oracle_utility>native_utility+1e-6; beneficial=nonnative & (selected_utility>native_utility+1e-6)
    behavior={
        "rows":int(mask.sum()),
        "selected_non_native_rate":float(nonnative[mask].float().mean()),
        "harmful_non_native_rate":float(harmful[mask].float().mean()),
        "beneficial_candidate_available_rate":float(available[mask].float().mean()),
        "beneficial_candidate_recall":float(beneficial[mask].sum().item()/max(int(available[mask].sum()),1)),
    }
    return {"selected_coords_xy_px":selected_xy,"selected_candidate_index":selected_index,"oracle_coords_xy_px":oracle_xy,"oracle_candidate_index":oracle_index},behavior


def evaluate_pairwise_index(
    model: CMCPLocalPairwiseSafetyComparator,
    index_path: str | Path,
    normalization: StaticTokenNormalization,
    *,
    expected_partition: str,
    device: str,
    point_batch_size: int=128,
    bootstrap_samples: int=5000,
    bootstrap_seed: int=1701,
    max_videos: int=0,
) -> dict[str,Any]:
    index=load_complete_pairwise_token_index(index_path,expected_partition=expected_partition)
    rows=index["videos"][:max_videos] if max_videos>0 else index["videos"]
    native_tracks=[]; selected_tracks=[]; oracle_tracks=[]; gt_tracks=[]; pred_visibility=[]; gt_visibility=[]; queries=[]; per_video=[]
    totals={"rows":0,"selected":0.0,"harmful":0.0,"available":0.0,"beneficial":0.0}
    for row in rows:
        video=load_pairwise_token_video(row); prediction,behavior=predict_pairwise_video(model,video,normalization,device=device,point_batch_size=point_batch_size)
        t=video.tensors; native=video.native_coords_xy_px; selected=prediction["selected_coords_xy_px"]; oracle=prediction["oracle_coords_xy_px"]
        visibility=t["native_visibility"]; gt_vis=~t["gt_occluded"]; args=(t["gt_tracks_yx"],visibility,gt_vis,t["query_points_tyx"])
        nm=compute_tapvid_metrics(_tracks_from_xy(native,256),*args,resolution=256,query_mode="first")
        sm=compute_tapvid_metrics(_tracks_from_xy(selected,256),*args,resolution=256,query_mode="first")
        om=compute_tapvid_metrics(_tracks_from_xy(oracle,256),*args,resolution=256,query_mode="first")
        ne=_visible_error_stats(native,t,raster=256); se=_visible_error_stats(selected,t,raster=256); oe=_visible_error_stats(oracle,t,raster=256)
        per_video.append({"source_index":video.source_index,"video_name":video.video_name,"native_AJ":float(nm["AJ"]),"selected_AJ":float(sm["AJ"]),"oracle_AJ":float(om["AJ"]),
                          "selected_AJ_gain_points":100*(float(sm["AJ"])-float(nm["AJ"])),"oracle_AJ_gain_points":100*(float(om["AJ"])-float(nm["AJ"])),
                          "native_delta_average":float(nm["<avg"]),"selected_delta_average":float(sm["<avg"]),"oracle_delta_average":float(om["<avg"]),
                          "selected_delta_gain_points":100*(float(sm["<avg"])-float(nm["<avg"])),"oracle_delta_gain_points":100*(float(om["<avg"])-float(nm["<avg"])),
                          "native_error":ne,"selected_error":se,"oracle_error":oe,"behavior":behavior})
        n=behavior["rows"]; totals["rows"]+=n; totals["selected"]+=behavior["selected_non_native_rate"]*n; totals["harmful"]+=behavior["harmful_non_native_rate"]*n
        totals["available"]+=behavior["beneficial_candidate_available_rate"]*n; totals["beneficial"]+=behavior["beneficial_candidate_recall"]*behavior["beneficial_candidate_available_rate"]*n
        native_tracks.append(_tracks_from_xy(native,256)); selected_tracks.append(_tracks_from_xy(selected,256)); oracle_tracks.append(_tracks_from_xy(oracle,256))
        gt_tracks.append(t["gt_tracks_yx"]); pred_visibility.append(visibility); gt_visibility.append(gt_vis); queries.append(t["query_points_tyx"])
    pooled=(torch.cat(gt_tracks),torch.cat(pred_visibility),torch.cat(gt_visibility),torch.cat(queries))
    nm=compute_tapvid_metrics(torch.cat(native_tracks),*pooled,resolution=256,query_mode="first")
    sm=compute_tapvid_metrics(torch.cat(selected_tracks),*pooled,resolution=256,query_mode="first")
    om=compute_tapvid_metrics(torch.cat(oracle_tracks),*pooled,resolution=256,query_mode="first")
    saj=[r["selected_AJ_gain_points"] for r in per_video]; oaj=[r["oracle_AJ_gain_points"] for r in per_video]; sd=[r["selected_delta_gain_points"] for r in per_video]
    weights=[r["native_error"]["rows"] for r in per_video]
    ns=float(np.average([r["native_error"]["severe_16px_rate"] for r in per_video],weights=weights)); ss=float(np.average([r["selected_error"]["severe_16px_rate"] for r in per_video],weights=weights)); os=float(np.average([r["oracle_error"]["severe_16px_rate"] for r in per_video],weights=weights))
    n=max(totals["rows"],1)
    return {"partition":expected_partition,"videos":len(per_video),"native_metrics":nm,"selected_metrics":sm,"oracle_metrics":om,
            "selected_gain_points":{"AJ":100*(float(sm["AJ"])-float(nm["AJ"])),"delta_average":100*(float(sm["<avg"])-float(nm["<avg"])),"OA":100*(float(sm["OA"])-float(nm["OA"]))},
            "oracle_gain_points":{"AJ":100*(float(om["AJ"])-float(nm["AJ"])),"delta_average":100*(float(om["<avg"])-float(nm["<avg"])),"OA":100*(float(om["OA"])-float(nm["OA"]))},
            "paired_video_selected_AJ_gain_CI":paired_video_bootstrap_ci(saj,seed=bootstrap_seed,samples=bootstrap_samples),
            "paired_video_oracle_AJ_gain_CI":paired_video_bootstrap_ci(oaj,seed=bootstrap_seed+1,samples=bootstrap_samples),
            "paired_video_selected_delta_gain_CI":paired_video_bootstrap_ci(sd,seed=bootstrap_seed+2,samples=bootstrap_samples),
            "severe_16px_rate":{"native":ns,"selected":ss,"oracle":os,"selected_delta":ss-ns,"oracle_delta":os-ns},
            "behavior":{"rows":totals["rows"],"selected_non_native_rate":totals["selected"]/n,"harmful_non_native_rate":totals["harmful"]/n,
                        "beneficial_candidate_available_rate":totals["available"]/n,"beneficial_candidate_recall":totals["beneficial"]/max(totals["available"],1)},
            "per_video":per_video,"cache_index_sha256":index["_index_sha256"],"candidate_coordinate_combined_sha256":index["candidate_coordinate_combined_sha256"]}


def train_pairwise_epoch(
    model: CMCPLocalPairwiseSafetyComparator,
    index_path: str | Path,
    normalization: StaticTokenNormalization,
    optimizer: torch.optim.Optimizer,
    loss_config: PairwiseSafetyLossConfig,
    *,
    device: str,
    point_batch_size: int,
    generator: torch.Generator,
    max_videos: int=0,
) -> dict[str,float]:
    index=load_complete_pairwise_token_index(index_path,expected_partition="fit")
    rows=index["videos"][:max_videos] if max_videos>0 else index["videos"]
    order=torch.randperm(len(rows),generator=generator).tolist(); totals={}; sequences=0
    model.train()
    for row_id in order:
        video=load_pairwise_token_video(rows[row_id]); point_order=torch.randperm(video.candidate_tokens.shape[0],generator=generator)
        for start in range(0,len(point_order),point_batch_size):
            ids=point_order[start:start+point_batch_size]
            tokens=video.candidate_tokens[ids].to(device); coords=video.candidate_coords_xy_px[ids].to(device); valid=video.candidate_valid_mask[ids].to(device); frame_valid=video.frame_valid[ids].to(device)
            tokens=normalize_static_tokens(tokens,valid,normalization)
            gt=video.tensors["gt_tracks_yx"][ids][...,[1,0]].to(device)*255.0; occluded=video.tensors["gt_occluded"][ids].to(device); query=video.tensors["query_points_tyx"][ids,0].round().long().to(device)
            summary=torch.zeros(len(ids),4,device=device); losses=[]
            for frame in range(tokens.shape[1]):
                frame_tokens=inject_dynamic_summary(tokens[:,frame],summary)
                output=model(frame_tokens,valid[:,frame],coords[:,frame])
                supervised=(~occluded[:,frame]) & (frame>query) & frame_valid[:,frame]
                losses.append(pairwise_safety_frame_loss(output,coords[:,frame],valid[:,frame],gt[:,frame],supervised,model.config,loss_config))
                summary=_update_summary(output,coords[:,frame],frame_valid[:,frame],summary)
            keys=losses[0].keys(); batch_losses={key:torch.stack([v[key] for v in losses]).mean() for key in keys}
            optimizer.zero_grad(set_to_none=True); batch_losses["loss"].backward()
            if not torch.isfinite(batch_losses["loss"]): raise FloatingPointError("non-finite pairwise loss")
            nn.utils.clip_grad_norm_(model.parameters(),float(loss_config.grad_clip_norm)); optimizer.step()
            b=len(ids); sequences+=b
            for key,value in batch_losses.items(): totals[key]=totals.get(key,0.0)+float(value.detach())*b
    return {key:value/max(sequences,1) for key,value in totals.items()}


def safety_feasible(validation: Mapping[str,Any]) -> bool:
    return bool(validation["behavior"]["harmful_non_native_rate"]<=0.01 and validation["severe_16px_rate"]["selected_delta"]<=1e-12)
