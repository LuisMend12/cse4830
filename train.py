"""
train.py — Training loop for the ReID model.

Loss combination (Bag of Tricks recipe):
    L_total = L_id  +  L_triplet  +  λ * L_center

    L_id      : cross-entropy with label smoothing (epsilon=0.1)
    L_triplet : batch-hard triplet loss with soft-plus margin
    L_center  : center loss (Wen et al. 2016) for tighter intra-class clusters
                — weighted at λ=5e-4, optimized by a separate SGD optimizer

LR schedule: 10-epoch linear warmup → cosine annealing.

The DataLoader uses P×K sampling — P identities × K images per identity
per batch — which guarantees each batch contains positives and negatives
for the triplet mining. Default P=8, K=8 (batch=64, more positives per
anchor than the old P=16/K=4).

Usage:
    python train.py --data_root /path/to/data --output_dir ./checkpoints

Key flags:
    --use_keypoints   feed (33,3) MediaPipe keypoints to the model alongside RGB
    --use_ibn         use ResNet50-IBN-a backbone (better cross-camera generalization)
    --use_skeleton    legacy: use rendered skeleton images instead of RGB
    --num_pids        P: identities per batch (default 8)
    --num_imgs        K: images per identity  (default 8)
"""

import os
import time
import argparse
import logging
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Sampler

from dataset import SkeletonReIDDataset, build_transforms
from model import GaitReIDNet
from evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing. epsilon=0.1 is standard in ReID."""
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon     = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets):
        log_probs = self.log_softmax(logits)
        with torch.no_grad():
            smooth = torch.full_like(log_probs,
                                     self.epsilon / (self.num_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
        return -(smooth * log_probs).sum(dim=1).mean()


class BatchHardTripletLoss(nn.Module):
    """
    For each anchor, mines the hardest positive (same PID, largest distance)
    and hardest negative (different PID, smallest distance) within the batch.
    Soft-plus margin (more stable than hard margin) by default.
    """
    def __init__(self, margin: float = 0.3, soft: bool = True):
        super().__init__()
        self.margin = margin
        self.soft   = soft

    def _pairwise_dist(self, x):
        dot = x @ x.t()
        sq  = dot.diag().unsqueeze(1)
        return (sq + sq.t() - 2 * dot).clamp(min=1e-12).sqrt()

    def forward(self, embeddings, labels):
        dist     = self._pairwise_dist(embeddings)
        same_pid = labels.unsqueeze(1) == labels.unsqueeze(0)
        diff_pid = ~same_pid

        ap = dist.clone(); ap[diff_pid] = -1e9
        an = dist.clone(); an[same_pid] =  1e9

        hardest_pos = ap.max(dim=1).values
        hardest_neg = an.min(dim=1).values

        if self.soft:
            return torch.log1p(torch.exp(hardest_pos - hardest_neg)).mean()
        return torch.relu(hardest_pos - hardest_neg + self.margin).mean()


class CenterLoss(nn.Module):
    """
    Center loss (Wen et al. 2016).

    Maintains a learnable centroid for each identity and pulls each
    embedding toward its class centroid. Tightens intra-class clusters
    in feature space, which complements triplet loss (which controls
    inter-class margin but not within-class compactness).

    Per BoT recipe: weight=5e-4 in the total loss, with a separate SGD
    optimizer for the centers themselves (lr=0.5). The center gradients
    are rescaled by 1/λ in train_one_epoch so the effective center
    update size doesn't depend on the loss weighting.
    """
    def __init__(self, num_classes: int, feat_dim: int = 2048):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        return ((features - centers_batch) ** 2).sum(dim=1).mean() / 2


# ---------------------------------------------------------------------------
# P × K sampler
# ---------------------------------------------------------------------------

class PKSampler(Sampler):
    """
    Yields indices such that each batch contains exactly P identities
    with K images each (total batch size = P*K).
    """
    def __init__(self, dataset, num_pids: int = 8, num_imgs: int = 8):
        self.P = num_pids
        self.K = num_imgs
        self.pid_index = defaultdict(list)
        for idx, (_, _, pid, _) in enumerate(dataset.samples):
            self.pid_index[pid].append(idx)
        self.pids = list(self.pid_index.keys())

    def __len__(self):
        return len(self.pids) * self.K

    def __iter__(self):
        random.shuffle(self.pids)
        batch = []
        for pid in self.pids:
            idxs    = self.pid_index[pid]
            replace = len(idxs) < self.K
            chosen  = (random.choices(idxs, k=self.K) if replace
                       else random.sample(idxs, self.K))
            batch.extend(chosen)
            if len(batch) == self.P * self.K:
                yield from batch
                batch = []


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader,
                    id_loss_fn, tri_loss_fn, center_loss_fn,
                    center_loss_weight,
                    optimizer, optimizer_center,
                    device, epoch, use_keypoints):
    model.train()
    total_loss = total_id = total_tri = total_cen = correct = total = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        imgs      = batch["img"].to(device)
        labels    = batch["pid"].to(device)
        keypoints = batch["keypoints"].to(device) if use_keypoints else None

        logits, pool_feat, bn_feat = model(imgs, keypoints)

        loss_id  = id_loss_fn(logits, labels)
        loss_tri = tri_loss_fn(pool_feat, labels)
        loss_cen = center_loss_fn(pool_feat, labels)
        loss     = loss_id + loss_tri + center_loss_weight * loss_cen

        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()

        # Rescale center gradients by 1/λ so the actual update size on
        # centers is independent of the loss weighting term (BoT trick).
        for p in center_loss_fn.parameters():
            if p.grad is not None:
                p.grad.data *= (1.0 / center_loss_weight)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        optimizer_center.step()

        total_loss += loss.item()
        total_id   += loss_id.item()
        total_tri  += loss_tri.item()
        total_cen  += loss_cen.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if (step + 1) % 50 == 0:
            n = step + 1
            log.info(
                f"Epoch {epoch:3d} | step {n:4d}/{len(loader)} | "
                f"loss={total_loss/n:.4f} "
                f"(id={total_id/n:.4f}, tri={total_tri/n:.4f}, "
                f"cen={total_cen/n:.4f}) | "
                f"acc={100*correct/total:.1f}%"
            )

    n = len(loader)
    log.info(
        f"Epoch {epoch:3d} done [{time.time()-t0:.0f}s] | "
        f"avg loss={total_loss/n:.4f} | train acc={100*correct/total:.1f}%"
    )
    return total_loss / n


# ---------------------------------------------------------------------------
# Argument parsing + entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ReID baseline (RGB + optional keypoints)")
    p.add_argument("--data_root",   required=True,
                   help="Root dir containing processed/, skeleton_images/, keypoints/")
    p.add_argument("--output_dir",  default="./checkpoints")
    p.add_argument("--epochs",        type=int, default=120)
    p.add_argument("--warmup_epochs", type=int, default=10,
                   help="Linear LR warmup before cosine annealing kicks in")
    p.add_argument("--num_pids",  type=int, default=8,
                   help="P: identities per training batch")
    p.add_argument("--num_imgs",  type=int, default=8,
                   help="K: images per identity per batch")
    p.add_argument("--eval_bs",   type=int, default=64,
                   help="Batch size for test/query evaluation")
    p.add_argument("--lr",            type=float, default=3.5e-4)
    p.add_argument("--center_lr",     type=float, default=0.5,
                   help="Separate SGD LR for the center-loss centroids")
    p.add_argument("--center_weight", type=float, default=5e-4,
                   help="λ: weight of center loss in the total loss")
    p.add_argument("--img_h",      type=int, default=256)
    p.add_argument("--img_w",      type=int, default=128)
    p.add_argument("--eval_every", type=int, default=5,
                   help="Run Rank-1/mAP evaluation every N epochs")

    # Modality flags
    p.add_argument("--use_keypoints", action="store_true",
                   help="Also feed (33,3) MediaPipe keypoints to the model")
    p.add_argument("--use_ibn", action="store_true",
                   help="Use ResNet50-IBN-a backbone (better cross-camera)")
    p.add_argument("--use_skeleton", action="store_true",
                   help="Legacy: use rendered skeleton images instead of RGB")
    p.add_argument("--include_no_detection", action="store_true",
                   help="Include samples where MediaPipe found no pose "
                        "(only relevant in skeleton mode)")

    # Re-ranking at evaluation time
    p.add_argument("--rerank", action="store_true",
                   help="Apply k-reciprocal re-ranking at every eval (~+5-10%% mAP)")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    img_size = (args.img_h, args.img_w)
    use_rgb  = not args.use_skeleton

    # ---- Datasets -----------------------------------------------------------
    def make_ds(split):
        return SkeletonReIDDataset(
            data_root         = args.data_root,
            split             = split,
            transform         = build_transforms(split, img_size, use_rgb),
            use_rgb           = use_rgb,
            use_keypoints     = args.use_keypoints,
            # In RGB mode we keep all samples — RGB is informative regardless
            # of whether MediaPipe found a pose.
            skip_no_detection = args.use_skeleton and not args.include_no_detection,
        )

    train_ds = make_ds("train")
    test_ds  = make_ds("test")
    query_ds = make_ds("query")

    num_classes = len(train_ds.pid2label)
    log.info(f"Training identities: {num_classes} | "
             f"train samples: {len(train_ds)} | "
             f"test: {len(test_ds)} | query: {len(query_ds)}")

    # ---- Loaders ------------------------------------------------------------
    sampler      = PKSampler(train_ds, args.num_pids, args.num_imgs)
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.num_pids * args.num_imgs,
        sampler     = sampler,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_bs,
                              shuffle=False, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_ds, batch_size=args.eval_bs,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ---- Model --------------------------------------------------------------
    model = GaitReIDNet(
        num_classes   = num_classes,
        pretrained    = True,
        use_keypoints = args.use_keypoints,
        use_ibn       = args.use_ibn,
    ).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Losses -------------------------------------------------------------
    id_loss_fn     = LabelSmoothingCE(num_classes).to(device)
    tri_loss_fn    = BatchHardTripletLoss().to(device)
    center_loss_fn = CenterLoss(num_classes, feat_dim=model.feat_dim).to(device)

    # ---- Optimizers ---------------------------------------------------------
    # Lower LR for pretrained backbone, full LR for new layers (BoT)
    optimizer = optim.Adam([
        {"params": model.backbone.parameters(),    "lr": args.lr * 0.1},
        {"params": model.bottleneck.parameters(),  "lr": args.lr},
        {"params": model.classifier.parameters(),  "lr": args.lr},
    ], weight_decay=5e-4)

    if args.use_keypoints:
        kp_params = list(model.kp_encoder.parameters()) + list(model.fusion.parameters())
        optimizer.add_param_group({"params": kp_params, "lr": args.lr})

    # Centers get their own SGD optimizer (BoT recipe)
    optimizer_center = optim.SGD(center_loss_fn.parameters(), lr=args.center_lr)

    # ---- LR schedule: linear warmup → cosine annealing ----------------------
    if args.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max  = args.epochs - args.warmup_epochs,
                                   eta_min= 1e-7)
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # ---- Training loop ------------------------------------------------------
    best_rank1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader,
                        id_loss_fn, tri_loss_fn, center_loss_fn,
                        args.center_weight,
                        optimizer, optimizer_center,
                        device, epoch, args.use_keypoints)
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rank1, mAP = evaluate(model, query_loader, test_loader,
                                  device, args.use_keypoints,
                                  use_rerank=args.rerank)
            log.info(f"[Eval] Epoch {epoch:3d} — Rank-1: {rank1:.2f}%  "
                     f"mAP: {mAP:.2f}%")

            ckpt = os.path.join(args.output_dir, f"epoch_{epoch:03d}.pth")
            torch.save({
                "epoch":              epoch,
                "rank1":              rank1,
                "mAP":                mAP,
                "state_dict":         model.state_dict(),
                "optimizer":          optimizer.state_dict(),
                "optimizer_center":   optimizer_center.state_dict(),
                "center_loss_state":  center_loss_fn.state_dict(),
            }, ckpt)

            if rank1 > best_rank1:
                best_rank1 = rank1
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "best_model.pth"))
                log.info(f"  -> New best Rank-1: {rank1:.2f}%")

    log.info(f"Training complete. Best Rank-1: {best_rank1:.2f}%")


if __name__ == "__main__":
    main()