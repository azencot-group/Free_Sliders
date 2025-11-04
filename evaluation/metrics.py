import torch

import lpips
from transformers import CLIPProcessor, CLIPModel
import numpy as np

from utils.utils import image_loader


def compute_span(scales, pos_clip, neg_clip):
    """Measuring how well the slider spans the concept."""

    vals_neg = [100]
    vals_pos = [0]
    for s in scales:
        if s < 0:
            vals_neg.append(neg_clip[s])
        elif s > 0:
            vals_pos.append(pos_clip[s])
        else:  # s == 0
            # use neg if coming from neg side, pos if coming from pos side
            vals_neg.append(neg_clip[0])
            vals_neg.append(0)
            vals_pos.append(pos_clip[0])

    vals_pos.append(100)
    vals_neg.reverse()

    vals_pos = [v[0] / 100 if isinstance(v, tuple) else v / 100 for v in vals_pos]
    vals_neg = [v[0] / 100 if isinstance(v, tuple) else v / 100 for v in vals_neg]

    vals_pos = np.array(vals_pos)
    vals_neg = np.array(vals_neg)

    diffs_pos = vals_pos[1:] - vals_pos[:-1]
    diffs_neg = vals_neg[1:] - vals_neg[:-1]
    diffs_pos = diffs_pos[diffs_pos >= 0]
    diffs_neg = diffs_neg[diffs_neg >= 0]

    diffs = np.concatenate([diffs_neg, diffs_pos])

    cv = np.std(diffs) # / np.abs(np.mean(diffs))
    return cv


def compute_range(scales, pos_clip, neg_clip):
    """
    Computes the directional range of the slider using cross-alignment scores:
    - Difference between most positive image's CLIP score and negative image with positive prompt
    - Difference between most negative image's CLIP score and positive image with negative prompt

    """
    # Calculate the minimum and maximum scales for positive and negative ranges
    min_scale, max_scale = min(scales), max(scales)

    # Compute the range using the corresponding scales
    pos_range = pos_clip[max_scale] - pos_clip[min_scale]
    neg_range = neg_clip[min_scale] - neg_clip[max_scale]

    return (pos_range + neg_range) / 2  # average range


def compute_concept_alignment(clip_scores, neutral_index):
    """
    Measures weighted CLIP alignment:
    Each score is weighted by its distance from the neutral scale (scale 0).
    """
    clip_scores = np.array(clip_scores)
    distances = np.abs(np.arange(len(clip_scores)) - neutral_index)
    weights = distances / distances.sum() if distances.sum() > 0 else np.zeros_like(distances)
    return np.sum(clip_scores * weights)


def compute_identity_preservation(lpips_scores):
    """
    Measures the average LPIPS distance to the base image (scale 0).
    Lower is better (identity preserved).
    """
    return np.mean(lpips_scores)


def calculate_lpips(scaled_imgs):
    loss_fn_alex = lpips.LPIPS(net='alex')
    lpips_scores = {}
    scales = list(scaled_imgs.keys())
    original = image_loader(scaled_imgs[0])

    for scale in scales:
        edited = image_loader(scaled_imgs[scale])
        l = loss_fn_alex(original, edited)
        lpips_scores[scale] = l.item()

    return lpips_scores

def calculate_delta_clip(clip_scores, prompt):

    delta_clip_scores = {}
    scales = list(clip_scores.keys())

    source_score = clip_scores[0]
    for scale in scales:
        delta_clip_scores[scale] = np.abs(clip_scores[scale] - source_score)

    # Exclude scale 0 and average the other values
    filtered_scores = [v for k, v in delta_clip_scores.items() if k != 0]  # Exclude scale 0
    delta_clip_avg = sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0

    return delta_clip_avg

def get_prompt(prompts, key):
    return prompts[key] if isinstance(prompts, dict) else getattr(prompts, key)

def calculate_clip(scaled_imgs, prompts):
    """
    Computes CLIP scores for each image in the slider using the appropriate prompt:
    - negative scales → negative prompt
    - scale 0         → neutral prompt
    - positive scales → positive prompt

    Args:
        scaled_imgs (dict[int, PIL.Image]): Mapping from scale to image
        prompts (dict[str, str]): {'positive': str, 'neutral': str, 'negative': str}

    Returns:
        dict[int, float]: Mapping from scale to delta CLIP score relative to scale 0
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    contextual_scores = {}
    positive_scores = {}
    negative_scores = {}
    scales = list(scaled_imgs.keys())

    for scale in scales:
        im = scaled_imgs[scale]

        # Choose prompt based on scale
        if scale < 0:
            context_prompt = get_prompt(prompts, "negative")
        elif scale > 0:
            context_prompt = get_prompt(prompts, "positive")
        else:
            context_prompt = get_prompt(prompts, "neutral")

        inputs = processor(text=[context_prompt], images=im, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            clip_score = outputs.logits_per_image[0][0].detach().cpu().item()
            contextual_scores[scale] = clip_score

        inputs = processor(text=[get_prompt(prompts, "positive")], images=im, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            clip_score = outputs.logits_per_image[0][0].detach().cpu().item()
            positive_scores[scale] = clip_score

        inputs = processor(text=[get_prompt(prompts, "negative")], images=im, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            clip_score = outputs.logits_per_image[0][0].detach().cpu().item()
            negative_scores[scale] = clip_score

    # plot_clip_scores(clip_scores)

    return contextual_scores, positive_scores, negative_scores

import matplotlib.pyplot as plt

def plot_clip_scores(clip_scores):
    """
    Plots the CLIP scores against the scales.

    Args:
        clip_scores (dict[int, float]): Mapping from scale to CLIP score.
    """
    scales = list(clip_scores.keys())
    scores = list(clip_scores.values())

    plt.figure(figsize=(10, 6))
    plt.plot(scales, scores, marker='o', linestyle='-', color='b')
    plt.title("CLIP Scores vs Scales")
    plt.xlabel("Scales")
    plt.ylabel("CLIP Scores")
    plt.grid(True)
    plt.show()

def compute_overall_video_scores(scales,
                                 pos_clip_scores,
                                    pos_viclip_scores,
                                    neg_clip_scores,
                                    neg_viclip_scores,
                                  lpips_scores,
                                  motion_scores,
                                  tag=None,
                                  logger=None):
    """
    Processes video scores across all scales and computes extended metrics.
    Returns a DataFrame with per-scale scores and 'Overall' scores per metric.
    """

    # Range
    range_static = compute_range(scales, pos_clip_scores, neg_clip_scores)
    range_dynamic = compute_range(scales, pos_viclip_scores, neg_viclip_scores)

    # Preservation
    static_preservation = compute_identity_preservation([v for k, v in lpips_scores.items() if k != 0])
    dynamic_preservation = compute_identity_preservation([v for k, v in motion_scores.items() if k != 0])

    # Span
    span_static = compute_span(scales, pos_clip_scores, neg_clip_scores)
    span_dynamic = compute_span(scales, pos_viclip_scores, neg_viclip_scores)

    overall_scores = {
        "range_static": range_static,
        "range_dynamic": range_dynamic,
        "span_static": span_static,
        "span_dynamic": span_dynamic,
        "preservation_static": static_preservation,
        "preservation_dynamic": dynamic_preservation
    }

    # Reshape the dictionary into a nested dict grouped by metric
    metrics = ["range", "span", "preservation"]
    derived_metrics = {
        metric: {
            "static": overall_scores[f"{metric}_static"],
            "dynamic": overall_scores[f"{metric}_dynamic"]
        }
        for metric in metrics
    }

    for name, score in derived_metrics.items():
        if logger:
            logger[f"evaluation/{name}"].log(score)

    return derived_metrics


def compute_overall_img_scores(clip_scores, lpips_scores, positive_clip, negative_clip, tag, logger=None):
    """
    Compute the overall score across scales.

    Parameters:
        scores_dict (dict): Keys are scales (e.g., -3, -1, 0, 1, 3), values are metric scores.
    Returns:
        float: The overall score.
    """
    scales = list(clip_scores.keys())
    # ------------------------------ DELTA CLIP (old) ------------------------------------------------
    logger.add_tags("pos_delta_clip")
    delta_clip = calculate_delta_clip(positive_clip, tag)
    delta_clip = calculate_delta_clip(positive_clip, tag)

    # ------------------------------ SMOOTHNESS ----------------------------------------
    span = compute_span(scales,
                              positive_clip,
                              negative_clip)

    # ------------------------------ RANGE ---------------------------------------------
    range_score = compute_range(scales,
                                positive_clip,
                                negative_clip)

    # ------------------------------ IDENTITY PRESERVATION -----------------------------
    lpips_filtered_scores = [v for k, v in lpips_scores.items() if k != 0] # Exclude scale 0
    identity_preservation = compute_identity_preservation(list(lpips_filtered_scores))

    overall_scores = {
        "delta_clip": delta_clip,
        "span_score": span,
        "range_score": range_score,
        "identity_preservation": identity_preservation
    }

    if logger is not None:
        for name, score in overall_scores.items():
            logger[f"{tag}/{name}"].log(score)

    return overall_scores
