#!/usr/bin/env python
import copy
import os
import sys
from typing import Any, Dict

import torch
from einops.einops import rearrange

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "LoFTR"))

from src.loftr import LoFTR, default_cfg

DEFAULT_CFG = copy.deepcopy(default_cfg)
DEFAULT_CFG["coarse"]["temp_bug_fix"] = True


class LoFTRWrapper(LoFTR):
    def __init__(
        self,
        config: Dict[str, Any] = DEFAULT_CFG,
    ):
        LoFTR.__init__(self, config)

    def forward(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        data = {
            "image0": image0,
            "image1": image1,
        }
        del image0, image1

        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(
                torch.cat([data["image0"], data["image1"]], dim=0)
            )
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(
                data["bs"]
            ), feats_f.split(data["bs"])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(
                data["image0"]
            ), self.backbone(data["image1"])

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), "n c h w -> n (h w) c")
        feat_c1 = rearrange(self.pos_encoding(feat_c1), "n c h w -> n (h w) c")

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"].flatten(-2), data["mask1"].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data
        )
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(
                feat_f0_unfold, feat_f1_unfold
            )

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        rename_keys: Dict[str, str] = {
            "mkpts0_f": "keypoints0",
            "mkpts1_f": "keypoints1",
            "mconf": "confidence",
        }
        out: Dict[str, torch.Tensor] = {}
        for k, v in rename_keys.items():
            _d = data[k]
            if isinstance(_d, torch.Tensor):
                out[v] = _d
            else:
                raise TypeError(
                    f"Expected torch.Tensor for item `{k}`. Gotcha {type(_d)}"
                )
        del data

        return out
