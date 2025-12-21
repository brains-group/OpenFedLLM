# OpenFedLLM Windows 安装和运行指南（CPU 版本）

本指南专门针对 Windows 系统且仅有集成显卡（无独立显卡）的环境。

## 系统要求

### 最低配置
- **操作系统**：Windows 10/11 64位
- **处理器**：现代多核 CPU（推荐 8 核心以上）
- **内存**：至少 16GB RAM（推荐 32GB+）
- **存储**：至少 50GB 可用空间

### 注意事项
⚠️ **重要**：由于使用 CPU 训练，速度会比 GPU 慢很多（可能慢 10-100 倍）。建议：
- 使用更小的模型（如 1B-3B 参数的模型）
- 减少训练轮数和数据量
- 考虑使用量化技术

## 安装步骤

### 1. 安装前提软件

#### 1.1 安装 Anaconda 或 Miniconda

**Miniconda（推荐，体积更小）：**
1. 访问：https://docs.conda.io/en/latest/miniconda.html
2. 下载 Windows 64位安装包
3. 运行安装程序，建议勾选 "Add Miniconda3 to PATH"
4. 安装完成后，打开新的命令提示符窗口验证：
   ```cmd
   conda --version
   ```

#### 1.2 安装 Git

1. 访问：https://git-scm.com/download/win
2. 下载 Windows 安装包
3. 安装时选择：
   - "Git from the command line and also from 3rd-party software"
   - "Use Windows' default console window"
4. 验证安装：
   ```cmd
   git --version
   ```

### 2. 克隆项目

打开 **命令提示符** 或 **PowerShell**：

```cmd
git clone --recursive --shallow-submodules https://github.com/rui-ye/OpenFedLLM.git
cd OpenFedLLM
```

### 3. 创建 Python 环境

```cmd
conda create -n fedllm python=3.10 -y
conda activate fedllm
```

**注意**：每次使用项目前都需要激活环境：
```cmd
conda activate fedllm
```

### 4. 安装依赖包（CPU 版本）

#### 4.1 安装 PyTorch CPU 版本

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 4.2 修改 requirements.txt

某些包在 Windows CPU 环境下需要调整或跳过：

**创建 `requirements-cpu.txt` 文件**（已为你准备好，见下文）

然后安装：
```cmd
pip install -r requirements-cpu.txt
```

**跳过或可选的包**：
- `bitsandbytes`：仅支持 CUDA，CPU 环境不需要
- `deepspeed`：在 Windows 上支持有限，且主要用于 GPU
- `triton`：CUDA 优化库，CPU 不需要

### 5. 设置环境变量

#### 方法 1：使用提供的批处理脚本（推荐）

运行：
```cmd
setup.bat
```

#### 方法 2：手动设置（临时）

**在 CMD 中**：
```cmd
set PYTHONPATH=%PYTHONPATH%;%CD%\evaluation\FinGPT
```

**在 PowerShell 中**：
```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD\evaluation\FinGPT"
```

### 6. 运行训练（CPU 优化版本）

#### 6.1 使用提供的 Windows 训练脚本

```cmd
training_scripts\run_sft_cpu.bat
```

#### 6.2 手动运行（自定义参数）

```cmd
python main_sft.py ^
  --learning_rate 5e-5 ^
  --model_name_or_path "facebook/opt-1.3b" ^
  --dataset_name "vicgalle/alpaca-gpt4" ^
  --dataset_sample 1000 ^
  --fed_alg "fedavg" ^
  --num_clients 5 ^
  --sample_clients 2 ^
  --max_steps 5 ^
  --num_rounds 10 ^
  --batch_size 2 ^
  --gradient_accumulation_steps 4 ^
  --seq_length 256 ^
  --peft_lora_r 8 ^
  --peft_lora_alpha 16 ^
  --use_peft ^
  --output_dir ./output ^
  --template "alpaca"
```

**参数说明（CPU 优化）**：
- `model_name_or_path`: 使用较小的模型（1B-3B 参数）
  - 推荐：`facebook/opt-1.3b`, `gpt2-large`, `EleutherAI/pythia-1.4b`
- `dataset_sample`: 1000（仅使用 1000 个样本，而不是 20000）
- `num_clients`: 5（减少客户端数量）
- `batch_size`: 2（减小批次大小）
- `gradient_accumulation_steps`: 4（通过梯度累积模拟更大的批次）
- `seq_length`: 256（减少序列长度）
- `peft_lora_r`: 8（减少 LoRA 参数）
- `max_steps`: 5（减少每轮训练步数）
- `num_rounds`: 10（减少总轮数）
- **移除**：`--load_in_8bit`（仅支持 CUDA）

## CPU 训练优化建议

### 1. 使用更小的模型
```
facebook/opt-125m        # 125M 参数（测试用）
facebook/opt-1.3b        # 1.3B 参数（推荐）
EleutherAI/pythia-1.4b   # 1.4B 参数
gpt2-large               # 774M 参数
```

### 2. 启用多线程
在运行前设置：
```cmd
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
```
（将 8 改为你的 CPU 核心数）

### 3. 减少内存使用
- 减小 `batch_size`（建议 1-4）
- 减小 `seq_length`（建议 128-512）
- 减少 `num_clients`
- 使用 `gradient_accumulation_steps` 来模拟更大的批次

### 4. 使用量化（如果可能）
虽然不能使用 8-bit 量化（需要 CUDA），但可以：
- 选择已经量化的模型
- 使用 PyTorch 的动态量化（需要额外配置）

## 预期性能

**训练速度参考**（基于 Intel i7-12700K）：
- **小模型（125M-350M）**：~1-5 秒/步
- **中等模型（1B-1.4B）**：~10-30 秒/步
- **大模型（3B+）**：~1-5 分钟/步

**完整训练时间估算**：
- 10 轮 × 5 步/轮 × 20 秒/步 = ~16 分钟（1.3B 模型）

## 常见问题

### Q1: 内存不足 (Out of Memory)
**解决方案**：
- 减小 `batch_size` 至 1
- 减小 `seq_length` 至 128
- 使用更小的模型
- 关闭其他应用程序

### Q2: 训练速度太慢
**解决方案**：
- 使用更小的模型和数据集
- 这是 CPU 训练的正常现象
- 考虑使用云端 GPU 服务（如 Google Colab, Kaggle）

### Q3: 找不到 CUDA
**解决方案**：
这是正常的，确保：
- 安装了 CPU 版本的 PyTorch
- 没有使用 `--load_in_8bit` 或 `--load_in_4bit` 参数
- 验证 PyTorch 使用 CPU：
  ```python
  import torch
  print(torch.cuda.is_available())  # 应该输出 False
  print(torch.__version__)
  ```

### Q4: 某些包安装失败
**可以跳过的包**（CPU 环境不需要）：
- `bitsandbytes`
- `triton`
- `deepspeed`（可选）

### Q5: 想在 GPU 服务器上运行
**推荐的免费 GPU 选项**：
- Google Colab：https://colab.research.google.com/
- Kaggle Kernels：https://www.kaggle.com/
- 各大学的计算集群（如果你是学生）

## 评估模型

评估代码在 `evaluation/` 目录下。大多数评估脚本也需要转换为 Windows 命令。

### 示例：运行开放式评估
```cmd
cd evaluation\open_ended
python eval_mt_bench.py --model-path ..\..\output\final_model
```

## 推荐工作流（CPU 环境）

1. **开发和测试**：在本地 Windows CPU 上使用小模型和小数据集
2. **完整训练**：使用云端 GPU 服务（Colab, Kaggle 等）
3. **评估**：可以在本地 CPU 上进行推理评估（速度较慢但可行）

## 额外资源

- **项目主页**：https://github.com/rui-ye/OpenFedLLM
- **论文**：https://arxiv.org/abs/2402.06954
- **Hugging Face 模型库**：https://huggingface.co/models
- **PyTorch 文档**：https://pytorch.org/docs/stable/index.html

## 获得帮助

如果遇到问题：
1. 检查本文档的"常见问题"部分
2. 查看项目的 GitHub Issues
3. 确保所有依赖包都正确安装
4. 验证 Python 环境配置正确

---

**最后更新**：2024-12
