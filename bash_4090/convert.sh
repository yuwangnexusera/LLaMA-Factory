#!/bin/bash

# 确保脚本在执行过程中遇到错误时停止
set -e

# 提取，转换，微调，合并，导出
# 1. 提取
echo "开始提取任务..."
python parse_ds/full_dose/full_dose_ds.py
echo "训练任务完成。"

# 2. 执行评估命令
echo "开始评估任务..."
python tests/eval/ipynb_test.py
echo "评估任务完成。"

# # 3. 执行导出命令
echo "开始导出任务..."
llamafactory-cli export examples/merge_lora/qwen_lora_sft.yaml
echo "导出任务完成。"

# echo "所有任务已完成。"