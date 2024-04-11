import json
import math
import os
from typing import List

from transformers.trainer import TRAINER_STATE_NAME

from .logging import get_logger
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.pyplot as plt


logger = get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: List[str] = ["loss"]) -> None:
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []

        for i in range(len(data["log_history"])):

            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#FFDAB9", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#FFD700", label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)


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
