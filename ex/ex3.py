"""
@project: THPMaster
@File   : ex3.py
@Desc   :
@Author : gql
@Date   : 2024/10/16 19:41
"""
import numpy as np
import matplotlib.pyplot as plt


def inverse_function_with_noise(x):
    # 定义一个1/x的函数并加入噪声，噪声随x增大而减小
    noise = np.random.normal(0, 30 / np.sqrt(x + 1), len(x))  # 噪声幅度与x的平方根成反比

    f = 200 / (x + 1) + 50  # 定义一个近似于1/x的曲线
    return f + noise  # 添加噪声


def scale_to_range(y, y_min, y_max, new_min, new_max):
    # 缩放y值到指定的范围 [new_min, new_max]
    y_scaled = new_min + (y - y_min) * (new_max - new_min) / (y_max - y_min)
    return y_scaled


def func1():
    x = np.linspace(1, 1000, 400)
    # 计算y值
    y = inverse_function_with_noise(x)

    # 获取原始y的最小值和最大值
    y_min, y_max = np.min(y), np.max(y)

    # 将y值缩放到 [80, 300] 的范围
    y_scaled = scale_to_range(y, y_min, y_max, 80, 300)

    # 使用移动平均法平滑曲线
    y_smooth = smooth(y_scaled, 10)
    y_smooth = y_scaled

    # 绘制图像
    plt.figure(figsize=(15, 9))
    plt.plot(x, y_smooth, linewidth=2)
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel('value', fontsize=20)
    plt.grid(True, axis='y')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('scaled_inverse_function_with_noise5.pdf')
    print('y', y)
    print('y min:', y_min, 'y_max:', y_max)
    print('y scaled:', y_scaled)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


func1()
