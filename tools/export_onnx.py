
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import CONFIGS
from models.svtr import build_model


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

    ckpt = torch.load(args.weights, map_location="cpu")
    cfg  = ckpt.get("cfg", CONFIGS[args.variant])

    # FIX: load with use_sgm from checkpoint config, not hard-coded False.
    # Previously use_sgm=False was passed unconditionally — if the checkpoint
    # was trained with SGM the state_dict keys would mismatch and load_state_dict
    # would raise an error (or silently drop weights with strict=False).
    use_sgm = cfg.get("use_sgm", True)
    use_frm = cfg.get("use_frm", True)

    model = build_model(
        args.variant, cfg["img_h"], cfg["img_w"],
        len(cfg["chars"]) + 1,
        use_frm=use_frm,
        use_sgm=use_sgm,
    )
    model.load_state_dict(ckpt["model_state"])
    # Export in inference mode: SGM branch is only active when return_sgm=True,
    # and we never call with return_sgm=True during ONNX export, so the graph
    # only contains the main CTC head regardless of use_sgm.
    model.eval()

    dummy     = torch.randn(1, 3, cfg["img_h"], cfg["img_w"])
    onnx_path = os.path.join(args.out_dir, f"transformers_ocr_{args.variant}.onnx")

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
        from onnxruntime.quantization import QuantType, quantize_dynamic
        onnx.checker.check_model(onnx_path)
        print("onnx check  OK")

        if args.quantize:
            q_path = onnx_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
            print(f"quantized   {q_path}")
            fp32_mb = os.path.getsize(onnx_path) / 1e6
            int8_mb = os.path.getsize(q_path) / 1e6
            print(f"size  FP32={fp32_mb:.1f}MB  INT8={int8_mb:.1f}MB  " f"ratio={fp32_mb/int8_mb:.1f}x")
    except ImportError:
        print("onnx / onnxruntime not installed — skipping check + quantize")

    print("\ndone.")


if __name__ == "__main__":
    main()
