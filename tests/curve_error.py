import json
import math
import os
from typing import List, Dict, Any

from transformers.trainer import TRAINER_STATE_NAME


import matplotlib.pyplot as plt


def merge_loss_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_data = []
    epoch_dict = {}  # 用于存储每个 epoch 的数据

    for entry in data:
        epoch = entry.get("epoch")

        if epoch is not None:
            if epoch not in epoch_dict:
                epoch_dict[epoch] = {"epoch": epoch}

            # 如果当前条目包含训练损失，则添加到当前 epoch 数据中
            if "loss" in entry:
                epoch_dict[epoch]["loss"] = entry["loss"]

            # 如果当前条目包含验证损失，则添加到当前 epoch 数据中
            if "eval_loss" in entry:
                epoch_dict[epoch]["eval_loss"] = entry["eval_loss"]

    # 将合并后的数据添加到 merged_data 列表中
    for epoch_data in epoch_dict.values():
        merged_data.append(epoch_data)

    return merged_data


def plot_loss(save_dictionary: os.PathLike, keys: List[str] = ["loss"]) -> None:
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)
    merged_data = merge_loss_data(data["log_history"])
    epochs = [entry["epoch"] for entry in merged_data]
    train_losses = [entry["loss"] for entry in merged_data if "loss" in entry]
    train_losses.append(merged_data[-1]["train_loss"] if "train_loss" in merged_data[-1] else None)
    eval_losses = [entry["eval_loss"] for entry in merged_data if "eval_loss" in entry]
    plt.figure()
    plt.plot(epochs, train_losses, color="#0000CD", label="Train Loss")
    plt.plot(epochs, eval_losses, color="#00FA9A", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    figure_path = os.path.join(save_dictionary, "loss.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print("Figure saved at:", figure_path)


if __name__ == "__main__":
    plot_loss(r"C:\code\LLaMA-Factory\tests")
