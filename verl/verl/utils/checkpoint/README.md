# FSDP模型保存与合并指南

## 保存FSDP模型

在训练过程中，`FSDPCheckpointManager`会以分片形式保存模型状态，同时也会保存模型配置和tokenizer。这些保存的文件包括：

- `model_world_size_{world_size}_rank_{rank}.pt`: 每个rank的模型分片
- `optim_world_size_{world_size}_rank_{rank}.pt`: 每个rank的优化器分片
- `extra_state_world_size_{world_size}_rank_{rank}.pt`: 额外状态信息
- `huggingface/`: 包含模型配置和tokenizer的目录

由于FSDP会将模型分片到不同的进程，直接尝试在保存时获取完整模型可能导致内存溢出或卡死。

## 合并FSDP模型分片

要获取完整的Hugging Face模型，请使用`scripts/model_merger.py`脚本合并分片模型。这个脚本会读取所有分片，将它们合并成一个完整的模型，然后保存为Hugging Face格式。

### 使用方法

```bash
python scripts/model_merger.py --local_dir /path/to/checkpoint/directory
```

参数说明：
- `--local_dir`: 包含模型分片的目录路径（不要包含huggingface子目录）
- `--hf_upload_path`(可选): 如果需要将合并后的模型上传到Hugging Face Hub，指定仓库路径

### 示例

```bash
# 合并模型分片并保存到本地
python scripts/model_merger.py --local_dir ./checkpoints/global_step_1000

# 合并模型分片并上传到Hugging Face Hub
python scripts/model_merger.py --local_dir ./checkpoints/global_step_1000 --hf_upload_path username/model-name
```

### 注意事项

1. 合并过程会消耗大量内存，确保有足够的CPU内存执行此操作
2. 合并后的模型将保存在`{local_dir}/huggingface/`目录下
3. 如果上传到Hugging Face Hub，需要安装`huggingface_hub`库并登录
4. 脚本会自动检测模型类型并使用适当的AutoModel类加载模型

## 故障排除

如果遇到内存问题，可以尝试在更大内存的机器上运行，或修改`model_merger.py`脚本启用更多的内存优化。

如果遇到模型架构不匹配的问题，检查`huggingface`目录下的`config.json`文件，确保`architectures`字段指向正确的模型类。 