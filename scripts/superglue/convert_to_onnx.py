#!/usr/bin/env python
import os

import torch

import onnxruntime
from superglue_wrapper import SuperGlueWrapper as SuperGlue


def main():
    config = {
        "descriptor_dim": 256,
        "weights": "indoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    }

    model = SuperGlue(config=config)
    model.eval()

    batch_size = 1
    height = 480
    width = 640
    num_keypoints = 382
    data = {}
    for i in range(2):
        data[f"image{i}_shape"] = torch.tensor(
            [batch_size, 1, height, width], dtype=torch.float32
        )
        data[f"scores{i}"] = torch.randn(batch_size, num_keypoints)
        data[f"keypoints{i}"] = torch.randn(batch_size, num_keypoints, 2)
        data[f"descriptors{i}"] = torch.randn(batch_size, 256, num_keypoints)

    torch.onnx.export(
        model,
        data,
        "super_glue.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=list(data.keys()),
        output_names=["matches0", "matches1", "matching_scores0", "matching_scores1"],
        dynamic_axes={
            "keypoints0": {0: "batch_size", 1: "num_keypoints0"},
            "scores0": {0: "batch_size", 1: "num_keypoints0"},
            "descriptors0": {0: "batch_size", 2: "num_keypoints0"},
            "keypoints1": {0: "batch_size", 1: "num_keypoints1"},
            "scores1": {0: "batch_size", 1: "num_keypoints1"},
            "descriptors1": {0: "batch_size", 2: "num_keypoints1"},
            "matches0": {0: "batch_size", 1: "num_keypoints0"},
            "matches1": {0: "batch_size", 1: "num_keypoints1"},
            "matching_scores0": {0: "batch_size", 1: "num_keypoints0"},
            "matching_scores1": {0: "batch_size", 1: "num_keypoints1"},
        },
    )
    print(f"\nonnx model is saved to: {os.getcwd()}/super_glue.onnx")

    print("\ntest inference using onnxruntime")
    sess = onnxruntime.InferenceSession("super_glue.onnx")
    for input in sess.get_inputs():
        print("input: ", input)

    print("\n")
    for output in sess.get_outputs():
        print("output: ", output)


if __name__ == "__main__":
    main()
