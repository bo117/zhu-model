# 竹 · 模型产物

「竹」中文作文写作模型的 LoRA 产物。代码在 [bo117/zhu](https://github.com/bo117/zhu)，语料在 [bo117/zhu-data](https://github.com/bo117/zhu-data)。

## 内容

| 路径 | 说明 |
|---|---|
| `adapter/` | LoRA adapter（r=32），约 148 MB |
| `adapter/adapter_model.safetensors.part1~3` | 权重分卷（GitHub 单文件上限 100 MB，故切成 3 卷） |
| `合并.py` | 把分卷还原成 `adapter_model.safetensors` |

**先合并再加载**：

```bash
python 合并.py     # 产出 adapter/adapter_model.safetensors
```

## 基座与训练参数

| 项 | 值 |
|---|---|
| 基座 | `Qwen/Qwen2.5-1.5B-Instruct` |
| LoRA | r=32 / alpha=64 / dropout=0.05 |
| target_modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| 训练量 | 120,000 样本 × 1 epoch，lr=2e-4，约 330 分钟（单卡） |
| 最终 loss | ≈ 3.32（几乎未下降，见下方说明） |

## 用法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "bo117/zhu-model", subfolder="adapter")
```

或直接用仓库里的推理服务：

```bash
python 本地模型服务.py      # 127.0.0.1:8766，OpenAI 兼容
```

## 合并成完整权重

```python
merged = model.merge_and_unload()
merged.save_pretrained("merged")
tok.save_pretrained("merged")
```

## 诚实说明

- 这一版 adapter 的 loss 停在 3.32，**基本没有降**。原因是当时的训练语料是规则引擎
  自己的输出（3 万篇同模板生成，句级重复 98.9%），模型学到的是拼接器，不是写作能力。
  这一批语料已停用，新数据见 [bo117/zhu-data](https://github.com/bo117/zhu-data)。
- 推理质量请以「是否复读 / 是否听懂指令」两项人工评估为准，不要看 loss。
- 训练脚本支持 CUDA / ROCm / CPU 自动适配；MoE 基座只挂 attention，见 `训练.py` 注释。

## 许可

个人项目，仅供学习与研究使用。
