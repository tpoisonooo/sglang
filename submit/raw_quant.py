"""
RTN (Round-To-Nearest) W4A16 quantization for MiniCPM-SALA.
No calibration data, no forward pass. Saves in standard GPTQ format for SGLang.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file


def quantize_weight_rtn(weight: torch.Tensor, bits: int = 4, group_size: int = 1024):
    """
    RTN quantize a single linear layer weight to GPTQ-compatible packed format.

    weight shape : [out_features, in_features]
    Returns      : qweight, scales, qzeros, g_idx
    """
    out_features, in_features = weight.shape
    pack_factor = 32 // bits
    maxq = 2 ** bits - 1

    if group_size <= 0:
        group_size = in_features
    num_groups = (in_features + group_size - 1) // group_size

    if in_features % group_size != 0:
        pad = group_size * num_groups - in_features
        weight = torch.nn.functional.pad(weight, (0, pad))
        in_features = weight.shape[1]

    w = weight.float()
    w_grouped = w.reshape(out_features, num_groups, group_size)
    w_min = w_grouped.min(dim=2, keepdim=True).values
    w_max = w_grouped.max(dim=2, keepdim=True).values

    scales = (w_max - w_min).clamp(min=1e-10) / maxq
    zeros = -w_min / scales

    qw = torch.clamp(torch.round(w_grouped / scales + zeros), 0, maxq).to(torch.int32)
    qw = qw.reshape(out_features, in_features)

    # qweight: [in_features // pack_factor, out_features]
    qw_t = qw.t().contiguous()
    qweight = torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32)
    for j in range(pack_factor):
        qweight |= qw_t[j::pack_factor, :] << (bits * j)

    # scales: [num_groups, out_features]
    scales_out = scales.squeeze(2).t().contiguous().half()

    # qzeros: [num_groups, out_features // pack_factor]  (packed along output dim)
    zeros_int = torch.clamp(torch.round(zeros.squeeze(2)), 0, maxq).to(torch.int32)
    zeros_t = zeros_int.t().contiguous()
    qzeros = torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32)
    for j in range(pack_factor):
        qzeros |= zeros_t[:, j::pack_factor] << (bits * j)

    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)

    return qweight, scales_out, qzeros, g_idx


LINEAR_SUFFIXES = (
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
    ".o_proj.weight", ".o_gate.weight", ".z_proj.weight",
    ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
)


def main():
    parser = argparse.ArgumentParser(description="RTN W4A16 quantization")
    parser.add_argument("--input", required=True, help="Original model directory")
    parser.add_argument("--output", required=True, help="Quantized model output directory")
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    # ---- copy non-weight files ----
    for f in src.iterdir():
        if f.suffix in (".json", ".py", ".model", ".txt") or f.name == "tokenizer.json":
            shutil.copy2(f, dst / f.name)

    # ---- patch config.json ----
    with open(src / "config.json") as f:
        config = json.load(f)
    config["quantization_config"] = {
        "bits": args.bits,
        "group_size": args.group_size,
        "quant_method": "gptq",
        "desc_act": False,
        "sym": False,
    }
    config["torch_dtype"] = "float16"
    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- write quantize_config.json (SGLang reads this separately) ----
    with open(dst / "quantize_config.json", "w") as f:
        json.dump({
            "bits": args.bits,
            "group_size": args.group_size,
            "desc_act": False,
            "sym": False,
            "lm_head": False,
            "dynamic": {},
        }, f, indent=2)

    # ---- quantize weight shards ----
    with open(src / "model.safetensors.index.json") as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    new_weight_map = {}
    total_quantized = 0

    for shard_name in shard_files:
        print(f"[quantize] Processing {shard_name} ...")
        shard = load_file(str(src / shard_name))
        new_tensors = {}

        for name, tensor in shard.items():
            if name.endswith(LINEAR_SUFFIXES):
                w = tensor.to(torch.float16)
                qweight, scales, qzeros, g_idx = quantize_weight_rtn(w, args.bits, args.group_size)

                base = name.removesuffix(".weight")
                for suffix, t in [(".qweight", qweight), (".scales", scales),
                                   (".qzeros", qzeros), (".g_idx", g_idx)]:
                    new_tensors[base + suffix] = t
                    new_weight_map[base + suffix] = shard_name
                total_quantized += 1
                print(f"  {name}: {list(w.shape)} -> qweight {list(qweight.shape)}")
            else:
                new_tensors[name] = tensor.to(torch.float16)
                new_weight_map[name] = shard_name

        save_file(new_tensors, str(dst / shard_name))

    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": new_weight_map}, f, indent=2)

    print(f"\n[quantize] Done! {total_quantized} layers -> W{args.bits}A16 (group_size={args.group_size})")


if __name__ == "__main__":
    main()
