# BiLLMv2

BiLLMv2 是面向 decoder-only 语言模型的纯后训练量化（PTQ）实现。本仓库只包含 BiLLMv2 核心代码、最小运行脚本与测试；不包含或分发对比方法、模型权重、数据集、缓存、实验日志和量化产物。

## 环境

当前代码面向 Linux、具备 CUDA 的 NVIDIA GPU，以及 Python 3.10 或更高版本。先安装与本机 CUDA 驱动匹配的 PyTorch，再安装其余依赖：

```bash
conda create -n billmv2 python=3.10 -y
conda activate billmv2
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

BiLLMv2 的层发现与困惑度评估依赖的 [BiLLM](https://github.com/htqin/BiLLM) 源码（`datautils.py`、`eval_ppl_utils.py`、`modelutils.py`）已按 MIT 许可原样 vendor 在 `external/BiLLM/` 内并随仓库提交，无需外部 checkout。可选地链接本地模型、数据集和 Hugging Face 缓存：

```bash
bash scripts/setup_links.sh /path/to/models /path/to/datasets /path/to/hf-cache
```

该脚本只在本地创建 `external/` 及可选软链接。模型与数据集的 Hugging Face 缓存默认位于 `/autodl-fs/data/cclanro/huggingface`，token 缓存位于 `/autodl-fs/data/cclanro/billm-v2-output/cache`；也可设置 `HF_HOME` 或对应环境变量覆盖。

## 运行

正式核心配置为 `billmv2_flr_f2`：W≈1.1A16、全局非对称分裂、activation-aware k-center 校准和 INT8 functional low-rank 补偿。

```bash
bash scripts/run_ptq.sh \
  huggyllama/llama-7b c4 \
  --preset billmv2_flr_f2 \
  --device cuda:0 \
  --eval_dataset wikitext2 \
  --output_dir outputs/llama7b_flr \
  --validate_reload
```

在 PTQ 后进行紧凑的仅校准数据微调：

```bash
bash scripts/run_ptq_ft.sh \
  huggyllama/llama-7b c4 \
  --preset billmv2_flr_f2 \
  --ft_steps 200 \
  --ft_lr 1e-4 \
  --ft_amp \
  --output_dir outputs/llama7b_flr_ft
```

使用 `python evaluate.py --artifact_dir outputs/llama7b_flr --dataset wikitext2` 评估已保存的紧凑 artifact；使用 `python -m pytest -q` 运行测试。

## 方法设计

BiLLMv2 以 BiLLM 的结构化二值表示为起点，并围绕量化器与校准流程组织为五个核心组成部分。

1. **量化器。** 相比 BiLLM，BiLLMv2 使用 residual-Hessian 显著性、有限候选的全局非对称分裂搜索、加权尺度求解、静态激活量化与紧凑 artifact 打包。纯 PTQ 在 `torch.no_grad()` 下执行，不更新原模型权重。

2. **INT8 因子存储。** 对被选中的低秩因子，BiLLMv2 将因子本体量化为 INT8，并保存对应尺度，从而将补偿项以紧凑且可重载的形式写入 artifact。

3. **低秩残差补偿。** 正式配置在 `o_proj` 和 `down_proj` 分支执行 functional-branch low-rank 搜索，并从候选秩中选择补偿量，以恢复二值量化后的功能误差。

4. **校准策略。** BiLLMv2 不将校准序列视为可互换样本，而是先提取 activation 或 joint quantization-error 特征，再进行逐层重建。正式 preset 从 512 条候选序列中基于 activation 特征以 k-center 选择 128 条代表性样本。

5. **校准样本优化。** 除 k-center 外，BiLLMv2 还提供 D-optimal 与 hybrid selector，在覆盖度和信息增益之间取舍。相较 BiLLM 的直接校准使用方式，BiLLMv2 将子集选择显式化并可复现：模型、数据集、随机种子、特征配置与选中索引均随 artifact 保存。

artifact 目录包含配置、校准索引、打包二值载荷、低秩因子、旋转参数、BPW 统计和指标，可通过 `billmv2.utils.artifacts.load_billmv2_artifacts` 与 `apply_billmv2_artifacts` 重载。

## 许可与致谢

实现来源与第三方许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。模型权重与数据集遵循各自许可，本仓库不再分发。
