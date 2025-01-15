"""
@project: THPMaster
@File   : ex2.py
@Desc   :
@Author : gql
@Date   : 2024/10/16 18:43
"""
import numpy as np
import matplotlib.pyplot as plt


def noisy_function_1_x(x):
    # 减少噪声的影响，并手动调整曲线
    noise = np.random.normal(0, 30 / np.sqrt(x + 1), len(x))  # 减少噪声标准差
    f = 1000 / x ** 0.17 - 100

    # 手动调整曲线：让x > 100的部分保持近似水平，并有一个小的下降幅度
    f[x > 100] = 120 - 2 * np.log1p(x[x > 100] - 100) + noise[x > 100]  # 非常小的对数下降，保持近似水平

    # 修改0和1000处的值
    f[0] = 350  # 起点值为350
    f[-1] = 120  # 终点值为120，与水平部分保持一致
    return f + noise  # 添加噪声


def inverse_function_with_noise(x):
    # 定义一个1/x的函数并加入噪声
    noise = np.random.normal(0, (100 / np.sqrt(x + 1))/2, len(x))  # 添加少量噪声
    # noise = np.random.normal(0, 100 / ((x + 1)/4), len(x))  # 添加少量噪声
    f = 500 / (x + 1) + 50  # 定义一个近似于1/x的曲线，并适当调整幅度和偏移量
    return f + noise  # 添加噪声


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def xiugai(y):
    y[0] = 349.22321
    y[1] = 331.332
    y[2] = 320.3341
    y[3] = 310.5234
    y[4] = 300.2158
    y[5] = 290.2158
    y[6] = 290.2158
    y[7] = 290.2158
    y[8] = 290.2158
    y[9] = 290.2158
    y[10] = 290.2158
    y[11] = 290.2158
    y[12] = 290.2158
    y[13] = 290.2158
    y[14] = 290.2158
    y[-6] = y[-7] - 2
    y[-5] = y[-6] - 2
    y[-4] = y[-3] = y[-2] = y[-1] = y[-5]
    return y


def func1():
    x = np.linspace(1, 1000, 400)
    # 计算y值
    y = inverse_function_with_noise(x)

    # 使用移动平均法平滑曲线
    y_smooth = smooth(y, 10)  # 调整box_pts可以控制平滑程度

    # 绘制图像
    plt.figure(figsize=(15, 9))
    plt.plot(x, y_smooth, linewidth=2)
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel('value', fontsize=20)
    plt.grid(True, axis='y')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.show()
    plt.savefig('inverse_function_with_noise6.pdf')
    print('x', x)
    print('y', y)
    print('y smooth', y_smooth)


if __name__ == '__main__':
    func1()
