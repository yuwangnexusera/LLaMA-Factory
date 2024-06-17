# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from typing import Any, Dict, Any, Dict, List

from transformers.trainer import TRAINER_STATE_NAME

from .logging import get_logger
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.figure
    import matplotlib.pyplot as plt


logger = get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def gen_loss_plot(trainer_log: List[Dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""
    Plots loss curves in LlamaBoard.
    """
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    for log in trainer_log:
        if log.get("loss", None):
            steps.append(log["current_steps"])
            losses.append(log["loss"])

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig


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
    r"""
    Plots loss curves and saves the image.
    """
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
