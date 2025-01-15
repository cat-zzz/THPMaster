"""
@project: THPMaster
@File   : ex4.py
@Desc   :
@Author : gql
@Date   : 2024/10/16 19:53
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_training_loss(epochs, initial_loss=1000, decay_rate=0.05, noise_level=50):
    """
    模拟训练过程中损失值的变化。

    :param epochs: 训练的总轮数
    :param initial_loss: 初始损失值
    :param decay_rate: 损失值衰减的速率
    :param noise_level: 噪声的幅度
    :return: 每个epoch的损失值
    """
    np.random.seed(43)  # 固定随机数种子，确保结果可复现
    loss_values = []

    for epoch in range(epochs):
        # 使用指数衰减函数来模拟损失的下降
        decay = initial_loss * np.exp(-decay_rate * epoch)
        # 添加随机噪声
        noise = np.random.normal(0, noise_level / np.sqrt(epoch + 1))
        # 总损失为衰减后的值加上噪声
        current_loss = decay + noise
        loss_values.append(max(current_loss, 0))  # 损失不能为负值
    # 将损失值缩放到 [50, 300] 范围内
    # loss_min, loss_max = np.min(loss_values), np.max(loss_values)
    # loss_scaled = 50 + (np.array(loss_values) - loss_min) * (300 - 50) / (loss_max - loss_min)
    loss_values[900] = 12.13
    loss_values[823] = 16.13
    loss_values[873] = 11.13
    loss_values[910] = 17.355
    loss_values[700] = 12.213
    loss_values[713] = 15.312
    loss_values[940] = 16.463
    loss_values[930] = 17.92
    loss_values[967] = 16.4
    loss_values[968] = 16.43
    loss_values[970] = 16.31
    loss_values[980] = 16.53
    loss_values[955] = 16.21
    loss_values = np.array(loss_values) + 42
    return loss_values


def plot_loss_curve(loss_values, epochs, sample_points=100):
    """
    绘制损失值的曲线图。

    :param loss_values: 模拟的损失值列表
    """
    sampled_indices = np.linspace(0, epochs - 1, sample_points, dtype=int)

    # 采样的损失值和对应的 epoch
    sampled_loss_values = np.array(loss_values)[sampled_indices]
    sampled_epochs = np.arange(epochs)[sampled_indices]

    plt.figure(figsize=(10, 6))
    # plt.plot(loss_values, label='Training Loss', linewidth=2)
    plt.plot(sampled_epochs, sampled_loss_values, label='Training Loss', linewidth=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Training loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_func.pdf')
    plt.show()


# 模拟训练1000个epoch的损失值
epochs = 1000
loss_values = simulate_training_loss(epochs, initial_loss=300, decay_rate=0.01, noise_level=100)

# 绘制损失曲线
plot_loss_curve(loss_values, epochs, sample_points=200)
