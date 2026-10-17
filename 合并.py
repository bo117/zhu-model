# -*- coding: utf-8 -*-
"""把分卷的 adapter 权重合并回单文件。

用法：python 合并.py
产出：adapter/adapter_model.safetensors
"""
import glob, os, sys

PARTS = sorted(glob.glob(os.path.join("adapter", "adapter_model.safetensors.part*")))
if not PARTS:
    sys.exit("没有找到分卷文件（adapter/adapter_model.safetensors.part*）")

dst = os.path.join("adapter", "adapter_model.safetensors")
tmp = dst + ".tmp"
total = 0
with open(tmp, "wb") as out:
    for p in PARTS:
        with open(p, "rb") as f:
            while True:
                b = f.read(8 << 20)
                if not b:
                    break
                out.write(b)
                total += len(b)
        print("+", os.path.basename(p))
os.replace(tmp, dst)
print(f"合并完成：{dst}  {total/1e6:.1f} MB  （{len(PARTS)} 卷）")
