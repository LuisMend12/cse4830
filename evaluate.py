"""
evaluate.py — Rank-1 accuracy and mean Average Precision (mAP) for ReID.

Standard Market-1501 evaluation protocol:
  - For each query, rank all gallery images by similarity.
  - Discard gallery images that share BOTH the same PID AND the same camera
    as the query (these are "same-camera same-identity" junk images).
  - Rank-1 : fraction of queries whose top-1 valid result is a true match.
  - mAP    : mean of Average Precision across all queries.

Optional: k-reciprocal re-ranking (Zhong et al. 2017) for a free ~5–10%
mAP boost. Enabled with `use_rerank=True` in evaluate(), or
`--rerank` on the command line.
"""

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device, use_keypoints=False):
    """
    Run the model in eval mode and collect embeddings + metadata.

    Returns:
        feats   : np.ndarray (N, D) — L2-normalized BN embeddings
        pids    : np.ndarray (N,)   — integer person IDs
        cam_ids : np.ndarray (N,)   — integer camera IDs
        paths   : list[str]         — source image paths (for debugging)
    """
    model.eval()
    all_feats, all_pids, all_cams, all_paths = [], [], [], []

    for batch in tqdm(loader, desc="Extracting features", leave=False):
        imgs = batch["img"].to(device)
        kps  = batch["keypoints"].to(device) if use_keypoints else None

        # model.forward() in eval mode returns L2-normalized (B, D) tensor
        feats = model(imgs, kps)

        all_feats.append(feats.cpu().numpy())
        all_pids.append(batch["pid"].numpy())
        all_cams.append(batch["cam"].numpy())
        all_paths.extend(batch["path"])

    return (
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_pids,  axis=0),
        np.concatenate(all_cams,  axis=0),
        all_paths,
    )


# ---------------------------------------------------------------------------
# k-reciprocal re-ranking (Zhong et al. 2017)
# ---------------------------------------------------------------------------

def re_ranking(q_feats: np.ndarray,
               g_feats: np.ndarray,
               k1: int = 20,
               k2: int = 6,
               lambda_value: float = 0.3) -> np.ndarray:
    """
    k-reciprocal re-ranking. Returns a (Q, G) distance matrix where smaller
    means more similar. Combine with argsort to rank gallery items.

    The intuition: if A and B are mutual k-NNs of each other, they're more
    likely to be the same identity than ordinary k-NNs. We encode each
    sample as a sparse weighted vector over its k-reciprocal neighbors,
    then use Jaccard distance between those vectors. The final distance
    is a weighted mix of original and Jaccard distances.

    Standard hyperparameters from the paper:
        k1=20, k2=6, lambda_value=0.3

    Memory: O((Q+G)^2) floats. For Market-1501 (~23k total) this is ~2 GB.

    Args:
        q_feats      : (Q, D) query features (L2-normalized)
        g_feats      : (G, D) gallery features (L2-normalized)
        k1, k2       : neighborhood sizes for reciprocal search and local QE
        lambda_value : weight on original distance (0 = pure Jaccard,
                       1 = no re-ranking)

    Returns:
        final_dist : (Q, G) re-ranked distance matrix
    """
    query_num = q_feats.shape[0]
    all_num   = query_num + g_feats.shape[0]
    feat      = np.concatenate([q_feats, g_feats], axis=0).astype(np.float32)

    # Pairwise squared Euclidean distances on the combined set
    sq_norms = np.sum(feat ** 2, axis=1, keepdims=True)
    original_dist = sq_norms + sq_norms.T - 2.0 * (feat @ feat.T)
    original_dist = np.clip(original_dist, 0.0, None)

    # Normalize each column to [0, 1] by its max — required for the
    # exp(-d) weighting below to be well-conditioned.
    col_max = original_dist.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    original_dist = original_dist / col_max

    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    # ---- Build k-reciprocal feature vectors V ------------------------------
    V = np.zeros_like(original_dist, dtype=np.float32)
    half_k1 = int(np.around(k1 / 2))

    for i in range(all_num):
        # k-reciprocal neighbors: forward & backward k1-NN intersection
        fwd = initial_rank[i, : k1 + 1]
        bwd = initial_rank[fwd, : k1 + 1]
        recip = fwd[np.where(bwd == i)[0]]
        expansion = recip.copy()

        # Local expansion: add k/2-reciprocal neighbors of each recip item
        # if they substantially overlap with the original recip set
        for cand in recip:
            cand_fwd = initial_rank[cand, : half_k1 + 1]
            cand_bwd = initial_rank[cand_fwd, : half_k1 + 1]
            cand_recip = cand_fwd[np.where(cand_bwd == cand)[0]]
            if (len(np.intersect1d(cand_recip, recip))
                    > (2.0 / 3.0) * len(cand_recip)):
                expansion = np.append(expansion, cand_recip)

        expansion = np.unique(expansion)
        w = np.exp(-original_dist[i, expansion])
        V[i, expansion] = w / w.sum()

    # ---- Local query expansion: average V over top-k2 neighbors ------------
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i] = V[initial_rank[i, :k2]].mean(axis=0)
        V = V_qe

    # ---- Jaccard distance via inverted index --------------------------------
    inv_index = [np.where(V[:, j] != 0)[0] for j in range(all_num)]

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(all_num, dtype=np.float32)
        nonzero  = np.where(V[i] != 0)[0]
        for j in nonzero:
            idx = inv_index[j]
            temp_min[idx] += np.minimum(V[i, j], V[idx, j])
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)

    final_dist = jaccard_dist * (1.0 - lambda_value) + original_dist * lambda_value
    return final_dist[:query_num, query_num:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ap(good_mask: np.ndarray) -> float:
    """Average Precision for a single query."""
    num_good = good_mask.sum()
    if num_good == 0:
        return 0.0
    positions  = np.where(good_mask)[0] + 1
    precisions = np.arange(1, num_good + 1) / positions
    return precisions.mean()


def evaluate(model, query_loader, gallery_loader, device,
             use_keypoints: bool = False,
             use_rerank:    bool = False,
             rerank_k1:     int  = 20,
             rerank_k2:     int  = 6,
             rerank_lambda: float = 0.3):
    """
    Compute Rank-1 and mAP using the standard Market-1501 protocol.

    Args:
        model           : GaitReIDNet (switched to eval internally)
        query_loader    : DataLoader for the query split
        gallery_loader  : DataLoader for the test (gallery) split
        device          : torch device
        use_keypoints   : whether to pass keypoints to the model
        use_rerank      : apply k-reciprocal re-ranking on top of cosine
                          similarity (Zhong et al. 2017)
        rerank_k1, k2,  : re-ranking hyperparameters (defaults from paper)
        rerank_lambda

    Returns:
        rank1 : float — Rank-1 accuracy in percent
        mAP   : float — mean Average Precision in percent
    """
    print("Extracting query features...")
    q_feats, q_pids, q_cams, _ = extract_features(
        model, query_loader, device, use_keypoints
    )
    print("Extracting gallery features...")
    g_feats, g_pids, g_cams, _ = extract_features(
        model, gallery_loader, device, use_keypoints
    )

    # Build a "score matrix" where a *higher* value means the gallery item
    # ranks more highly. For cosine similarity that's the similarity itself;
    # for re-ranking distance it's the negated distance.
    if use_rerank:
        print(f"Running re-ranking (k1={rerank_k1}, k2={rerank_k2}, "
              f"λ={rerank_lambda})...")
        rerank_dist = re_ranking(q_feats, g_feats,
                                 k1=rerank_k1, k2=rerank_k2,
                                 lambda_value=rerank_lambda)
        score_matrix = -rerank_dist          # smaller dist → higher score
    else:
        # L2-normalized features → dot product = cosine similarity
        score_matrix = q_feats @ g_feats.T

    num_queries = q_feats.shape[0]
    rank1_hits  = 0
    ap_scores   = []

    for q_idx in range(num_queries):
        q_pid  = q_pids[q_idx]
        q_cam  = q_cams[q_idx]

        scores = score_matrix[q_idx]
        ranked = np.argsort(-scores)         # descending

        ranked_pids = g_pids[ranked]
        ranked_cams = g_cams[ranked]

        # Junk: same PID AND same camera as query → exclude
        junk_mask = (ranked_pids == q_pid) & (ranked_cams == q_cam)
        # Good: same PID, different camera → true positive
        good_mask = (ranked_pids == q_pid) & (ranked_cams != q_cam)

        valid_idx  = ~junk_mask
        good_valid = good_mask[valid_idx]

        if good_valid.sum() == 0:
            continue

        rank1_hits += int(good_valid[0])
        ap_scores.append(compute_ap(good_valid))

    rank1 = 100.0 * rank1_hits / num_queries
    mAP   = 100.0 * float(np.mean(ap_scores)) if ap_scores else 0.0

    return rank1, mAP


# ---------------------------------------------------------------------------
# Standalone entry point — evaluate a saved checkpoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from dataset import SkeletonReIDDataset, build_transforms
    from model import GaitReIDNet

    parser = argparse.ArgumentParser(description="Evaluate a saved ReID checkpoint")
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to a saved .pth state_dict or full checkpoint")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of training identities (used to build the model)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_h",      type=int, default=256)
    parser.add_argument("--img_w",      type=int, default=128)

    # Modality / model flags — must match the training run!
    parser.add_argument("--use_keypoints", action="store_true")
    parser.add_argument("--use_ibn",       action="store_true",
                        help="Set if checkpoint was trained with --use_ibn")
    parser.add_argument("--use_skeleton",  action="store_true",
                        help="Legacy: evaluate against rendered skeleton images")

    # Re-ranking
    parser.add_argument("--rerank",        action="store_true",
                        help="Enable k-reciprocal re-ranking (~+5-10%% mAP)")
    parser.add_argument("--rerank_k1",     type=int,   default=20)
    parser.add_argument("--rerank_k2",     type=int,   default=6)
    parser.add_argument("--rerank_lambda", type=float, default=0.3)

    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    img_size = (args.img_h, args.img_w)
    use_rgb  = not args.use_skeleton

    def make_loader(split):
        ds = SkeletonReIDDataset(
            data_root     = args.data_root,
            split         = split,
            transform     = build_transforms(split, img_size, use_rgb),
            use_rgb       = use_rgb,
            use_keypoints = args.use_keypoints,
            # Don't filter at eval time — it would skew the protocol
            skip_no_detection = False,
        )
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

    model = GaitReIDNet(
        num_classes   = args.num_classes,
        pretrained    = False,
        use_keypoints = args.use_keypoints,
        use_ibn       = args.use_ibn,
    ).to(device)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt)   # handle both full ckpt and plain state_dict
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    rank1, mAP = evaluate(
        model,
        make_loader("query"),
        make_loader("test"),
        device,
        use_keypoints = args.use_keypoints,
        use_rerank    = args.rerank,
        rerank_k1     = args.rerank_k1,
        rerank_k2     = args.rerank_k2,
        rerank_lambda = args.rerank_lambda,
    )
    print(f"\nRank-1: {rank1:.2f}%")
    print(f"mAP   : {mAP:.2f}%")