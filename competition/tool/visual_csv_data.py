"""
@File   : visual_csv_data.py
@Desc   : 
@Author : gql
@Date   : 2023/7/20 17:25
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visual_hands_win_rate_csv():
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    # 数据读取
    df = pd.read_csv('hands_win_rate.csv', header=0, usecols=range(1, 14))
    data = df.values  # 转为ndarray类型
    X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # 点数
    Y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # 点数
    X, Y = np.meshgrid(X, Y)
    print("网格化后的X=", X)
    print("X维度信息", X.shape)
    print("网格化后的Y=", Y)
    print("Y维度信息", Y.shape)
    Z = data[:, :]
    print("维度调整前的Z轴数据维度", Z.shape)
    Z = Z.T
    print("维度调整后的Z轴数据维度", Z.shape)
    # 绘制三维散点图
    colors = X + Y
    ax.scatter3D(X, Y, Z, c=colors, cmap='rainbow', alpha=1)
    # ax.plot_surface(X, Y, Z)
    # 设置三个坐标轴信息
    ax.set_xlabel('first card', color='b')
    ax.set_ylabel('second card', color='g')
    ax.set_zlabel('win rate', color='r')

    ax.set_title('hand cards win rate')
    plt.draw()
    plt.show()
    # plt.savefig('hands_win_rate.jpg')


if __name__ == '__main__':
    visual_hands_win_rate_csv()
