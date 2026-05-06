import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Weight initialisers
# ---------------------------------------------------------------------------

def _init_kaiming(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias,   0.0)


def _init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# Backbone selection (ResNet50 or ResNet50-IBN-a)
# ---------------------------------------------------------------------------

def _build_backbone(use_ibn: bool, pretrained: bool):
    """
    Returns a ResNet50 (or ResNet50-IBN-a) backbone module with the same
    interface (conv1, bn1, relu, maxpool, layer1..layer4, fc).

    IBN-Net (Pan et al. 2018) interleaves Instance Normalization with
    Batch Normalization in the early stages, which empirically helps a
    lot with cross-camera generalization in ReID — exactly the kind of
    overfitting symptom we're trying to fix.
    """
    if use_ibn:
        try:
            backbone = torch.hub.load(
                "XingangPan/IBN-Net",
                "resnet50_ibn_a",
                pretrained=pretrained,
            )
            print("[Model] Using ResNet50-IBN-a backbone")
            return backbone
        except Exception as e:
            print(f"[Model] IBN-Net load failed ({e}); falling back to ResNet50")

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    print("[Model] Using ResNet50 backbone")
    return models.resnet50(weights=weights)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GaitReIDNet(nn.Module):
    """
    ResNet50 (optionally IBN-a) + BN-neck for ReID.

    Args:
        num_classes   : number of training identities
        feat_dim      : backbone output channels (2048 for ResNet50)
        last_stride   : 1 keeps more spatial resolution in layer4 (BoT paper)
        pretrained    : load ImageNet weights for the backbone
        use_keypoints : if True, accepts (B, 99) keypoint vector and fuses
                        it into the embedding before the BN neck
        use_ibn       : if True, use ResNet50-IBN-a instead of ResNet50
    """

    def __init__(self,
                 num_classes:   int,
                 feat_dim:      int  = 2048,
                 last_stride:   int  = 1,
                 pretrained:    bool = True,
                 use_keypoints: bool = False,
                 use_ibn:       bool = False):
        super().__init__()
        self.feat_dim      = feat_dim
        self.num_classes   = num_classes
        self.use_keypoints = use_keypoints

        # ---- Backbone -------------------------------------------------------
        backbone = _build_backbone(use_ibn, pretrained)

        # Lower stride in the last residual stage to preserve spatial detail
        if last_stride == 1:
            backbone.layer4[0].conv2.stride         = (1, 1)
            backbone.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # ---- Pooling --------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ---- Optional keypoint fusion branch --------------------------------
        if use_keypoints:
            self.kp_encoder = nn.Sequential(
                nn.Linear(99, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
            )
            self.fusion = nn.Linear(feat_dim + 256, feat_dim)
            self.fusion.apply(_init_kaiming)

        # ---- BN neck (no bias — BoT paper) ----------------------------------
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_init_kaiming)

        # ---- ID-loss classifier (training only) -----------------------------
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(_init_classifier)

    # ------------------------------------------------------------------

    def forward(self, x, keypoints=None):
        """
        Args:
            x         : (B, 3, H, W) image tensor (RGB or skeleton)
            keypoints : (B, 99) MediaPipe keypoint array — only used when
                        use_keypoints=True and keypoints is not None

        Returns:
            Training : (logits, pool_feat, bn_feat)
            Eval     : L2-normalized bn_feat, shape (B, feat_dim)
        """
        feat_map  = self.backbone(x)
        pool_feat = self.gap(feat_map).flatten(1)

        if self.use_keypoints and keypoints is not None:
            kp_feat   = self.kp_encoder(keypoints)
            pool_feat = self.fusion(torch.cat([pool_feat, kp_feat], dim=1))

        bn_feat = self.bottleneck(pool_feat)

        if not self.training:
            return F.normalize(bn_feat, dim=1)

        logits = self.classifier(bn_feat)
        return logits, pool_feat, bn_feat

    def get_embedding(self, x, keypoints=None):
        """Always returns the L2-normalized BN feature regardless of training state."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            emb = self.forward(x, keypoints)
        if was_training:
            self.train()
        return emb