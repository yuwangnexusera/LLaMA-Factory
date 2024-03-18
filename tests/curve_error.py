import json
import math
import os
from typing import List

from transformers.trainer import TRAINER_STATE_NAME


import matplotlib.pyplot as plt


def plot_loss_error_curve(
    save_dictionary: os.PathLike, keys: List[str] = ["error_curve"]
) -> None:
    try:
        with open(
            os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
            # 遍历data["log_history"] 列表中每个字典 将key为epoch和steps都相同的字典合并
            data["log_history"] = [
                {**data["log_history"][i], **data["log_history"][i + 1]}
                for i in range(0, len(data["log_history"]) - 1, 2)
            ]

        # 初始化用于收集每个epoch的平均误差的字典
        epoch_error = {}

        # 遍历log_history，计算loss和eval_loss的差值
        for entry in data["log_history"]:
            if "epoch" in entry and "step" in entry:
                epoch = entry["epoch"]
                steps = entry["step"]
                # 确保每个epoch的数据只被计算一次
                if (epoch, steps) not in epoch_error:
                    # 检查是否同时有loss和eval_loss
                    if "loss" in entry and "eval_loss" in entry:
                        error = entry["loss"] - entry["eval_loss"]
                        epoch_error[(epoch, steps)] = error

        epochs = [k[0] for k in sorted(epoch_error.keys())]
        errors = [epoch_error[k] for k in sorted(epoch_error.keys())]

        if len(errors) == 0:
            return

        plt.figure()
        plt.plot(epochs, errors, color="#FF6347", alpha=0.4, label="Error Curve")
        plt.title("Error Curve over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Error (Loss - Eval Loss)")
        plt.legend()
        figure_path = os.path.join(save_dictionary, "error_curve.png")
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)
    except Exception as e:
        print(f"Failed to plot loss error curve: {e}")


if __name__ == "__main__":
    plot_loss_error_curve(r"C:\Users\Administrator\Downloads")
