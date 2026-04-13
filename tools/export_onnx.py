import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.svtr import build_model
from engine.augment import get_val_transforms
from configs.config import CONFIGS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  required=True)
    p.add_argument("--variant",  default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--out_dir",  default="exports")
    p.add_argument("--quantize", action="store_true", help="Also export INT8")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt    = torch.load(args.weights, map_location="cpu")
    cfg     = ckpt.get("cfg", CONFIGS[args.variant])
    model   = build_model(args.variant, cfg["img_h"], cfg["img_w"],
                           len(cfg["chars"]) + 1,
                           use_frm=cfg.get("use_frm", True),
                           use_sgm=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy      = torch.randn(1, 3, cfg["img_h"], cfg["img_w"])
    onnx_path  = os.path.join(args.out_dir, f"transformers_ocr_{args.variant}.onnx")

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"],
        output_names=["log_probs"],
        dynamic_axes={"image": {0: "batch"}, "log_probs": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"exported  {onnx_path}")

    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        onnx.checker.check_model(onnx_path)
        print(f"onnx check  OK")

        if args.quantize:
            q_path = onnx_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(onnx_path, q_path,
                              weight_type=QuantType.QInt8)
            print(f"quantized   {q_path}")

            import os as _os
            fp32_mb = _os.path.getsize(onnx_path) / 1e6
            int8_mb = _os.path.getsize(q_path) / 1e6
            print(f"size  FP32={fp32_mb:.1f}MB  INT8={int8_mb:.1f}MB  "
                  f"ratio={fp32_mb/int8_mb:.1f}x")
    except ImportError:
        print("onnx / onnxruntime not installed — skipping check + quantize")

    print("\ndone.")


if __name__ == "__main__":
    main()

