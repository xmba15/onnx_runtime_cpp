#!/usr/bin/env python
import os

import torch
import torch.onnx

from SuperPointPretrainedNetwork.demo_superpoint import SuperPointNet

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
_WEIGHTS_PATH = os.path.join(
    _CURRENT_DIR, "SuperPointPretrainedNetwork/superpoint_v1.pth"
)


def main():
    assert os.path.isfile(_WEIGHTS_PATH)
    model = SuperPointNet()
    model.load_state_dict(torch.load(_WEIGHTS_PATH))
    model.eval()

    batch_size = 1
    height = 16
    width = 16
    x = torch.randn(batch_size, 1, height, width)

    torch.onnx.export(
        model,
        x,
        "super_point.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size"},
        },
    )
    print(f"\nonnx model is saved to: {os.getcwd()}/super_point.onnx")


if __name__ == "__main__":
    main()
