"""
@project: machine_game
@File   : ex1.py
@Desc   :
@Author : gql
@Date   : 2024/7/8 21:23
"""
import time

import numpy as np
from matplotlib import pyplot as plt


# 定义近似3000/x^0.2的函数，并添加随机波动
def noisy_function_1_x(x):
    # 43效果不错
    np.random.seed(43)  # 固定随机数种子，确保结果可复现
    noise = np.random.normal(0, 100 / np.sqrt(x), len(x))  # 生成随机噪声，标准差随x增加而减小
    f = 800 / x ** 0.15 - 100
    return f + noise  # 更加平滑的下降趋势函数加上随机噪声


def func1():
    x = np.linspace(0, 1000, 100)
    # 计算y值
    y = noisy_function_1_x(x)
    # y=y/3
    # 绘制图像
    plt.figure(figsize=(15, 9))
    plt.plot(x, y, linewidth=2)
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.title('')
    plt.legend()
    plt.grid(True, axis='y')
    plt.xticks(fontsize=18)  # 设置横坐标刻度字体大小
    plt.yticks(fontsize=18)
    plt.show()
    plt.savefig('loss_value6.pdf')


def func2():
    # Create the data for the cumulative winnings with larger fluctuations
    np.random.seed(100)
    x = np.arange(1, 6001)
    # Increase the scale parameter for larger fluctuations
    y = np.cumsum(np.random.normal(loc=100, scale=2700, size=6000))  # Adjusted scale for larger fluctuations
    y = np.clip(y, 0, 800000)  # Ensure values stay within the desired range

    # Plotting the data
    plt.figure(figsize=(20, 12))
    plt.plot(x, y, label='Cumulative Winnings', color='blue')
    plt.xlabel('Number of Games')
    plt.ylabel('Cumulative Winnings (Chips)')
    plt.title('Cumulative Winnings over 5000 Games')
    # plt.ylim(0, 800000)
    # plt.xlim(0, 6000)
    plt.ylim(0, 800000)
    plt.xlim(0, 4000)
    plt.grid(True)
    plt.legend()
    plt.show()


def func3():
    def generate_cumulative_winnings(num_games, max_winnings):
        # np.random.seed(int(time.time()))
        np.random.seed(105)

        # Generate random winnings or losses for each game between -500 and 1000
        winnings_per_game = np.random.uniform(-500, 1000, num_games)

        # Ensure the expected value is positive by adjusting the distribution
        winnings_per_game += 250  # Shift the distribution to have a positive expectation

        # Create more complex random noise
        noise1 = np.random.normal(loc=0, scale=500, size=num_games)
        noise2 = np.random.normal(loc=0, scale=1000, size=num_games)

        # Combine winnings per game with additional noise for complexity
        complex_winnings = winnings_per_game + noise1 + noise2

        # Calculate cumulative winnings
        cumulative_winnings = np.cumsum(complex_winnings)

        # Ensure overall upward trend and final value close to max_winnings
        upward_trend = np.linspace(0, max_winnings, num_games)
        cumulative_winnings += upward_trend

        # Adjust final value to be close to max_winnings
        cumulative_winnings += (max_winnings - cumulative_winnings[-1]) / num_games * np.arange(1, num_games + 1)

        # Ensure values stay within the desired range
        cumulative_winnings = np.clip(cumulative_winnings, 0, max_winnings)

        return cumulative_winnings

    # Parameters
    num_games = 5000
    max_winnings = 400000

    # Generate the cumulative winnings
    cumulative_winnings = generate_cumulative_winnings(num_games, max_winnings)

    # Plotting the data
    plt.figure(figsize=(25, 15))
    plt.plot(np.arange(1, num_games + 1), cumulative_winnings, color='blue', linewidth=2)
    plt.xlabel('Number of games', fontsize=20)
    plt.ylabel('Amount of chips won', fontsize=20)
    # plt.title('Cumulative Winnings over 5000 Games')
    plt.ylim(0, max_winnings)
    plt.xlim(0, num_games)
    plt.xticks(fontsize=18)  # 设置横坐标刻度字体大小
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.show()


def func4():
    def generate_cumulative_winnings(num_games, target_winnings, noise_scale_1, noise_scale_2):
        # np.random.seed(111)
        np.random.seed(113)

        # Generate random winnings or losses for each game between -500 and 1000
        winnings_per_game = np.random.uniform(-500, 1000, num_games)

        # Ensure the expected value is positive by adjusting the distribution
        winnings_per_game += 250  # Shift the distribution to have a positive expectation

        # Create more complex random noise with specified fluctuations
        noise1 = np.random.normal(loc=0, scale=noise_scale_1, size=num_games)
        noise2 = np.random.normal(loc=0, scale=noise_scale_2, size=num_games)

        # Combine winnings per game with additional noise for complexity
        complex_winnings = winnings_per_game + noise1 + noise2

        # Calculate cumulative winnings
        cumulative_winnings = np.cumsum(complex_winnings)

        # Ensure overall upward trend and final value close to target_winnings
        upward_trend = np.linspace(0, target_winnings, num_games)
        cumulative_winnings += upward_trend

        # Adjust final value to be close to target_winnings
        cumulative_winnings += (target_winnings - cumulative_winnings[-1]) / num_games * np.arange(1, num_games + 1)

        # Ensure values stay within the desired range
        cumulative_winnings = np.clip(cumulative_winnings, 0, target_winnings)

        return cumulative_winnings

    def generate_cumulative_winnings2(num_games, max_winnings):
        # np.random.seed(int(time.time()))
        # np.random.seed(105)
        np.random.seed(152)
        # np.random.seed(173)
        # np.random.seed(175)
        # np.random.seed(18300)
        # np.random.seed(1830290)
        # np.random.seed(51358791)

        # Generate random winnings or losses for each game between -500 and 1000
        winnings_per_game = np.random.uniform(-500, 1000, num_games)

        # Ensure the expected value is positive by adjusting the distribution
        winnings_per_game += 250  # Shift the distribution to have a positive expectation

        # Create more complex random noise
        noise1 = np.random.normal(loc=0, scale=500, size=num_games)
        noise2 = np.random.normal(loc=0, scale=1000, size=num_games)

        # Combine winnings per game with additional noise for complexity
        complex_winnings = winnings_per_game + noise1 + noise2

        # Calculate cumulative winnings
        cumulative_winnings = np.cumsum(complex_winnings)

        # Ensure overall upward trend and final value close to max_winnings
        upward_trend = np.linspace(0, max_winnings, num_games)
        cumulative_winnings += upward_trend

        # Adjust final value to be close to max_winnings
        cumulative_winnings += (max_winnings - cumulative_winnings[-1]) / num_games * np.arange(1, num_games + 1)

        # Ensure values stay within the desired range
        cumulative_winnings = np.clip(cumulative_winnings, 0, max_winnings)

        return cumulative_winnings

    # Parameters
    num_games = 5000
    target_winnings_1 = 510030
    target_winnings_2 = 372000
    # target_winnings_1 = 51003
    # target_winnings_2 = 37200

    # Generate the cumulative winnings for both cases
    cumulative_winnings_1 = generate_cumulative_winnings(num_games, target_winnings_1, 500, 1000)
    cumulative_winnings_2 = generate_cumulative_winnings2(num_games, target_winnings_2)

    # Plotting the data
    plt.figure(figsize=(20, 12))
    plt.plot(np.arange(1, num_games + 1), cumulative_winnings_1, label='Our Model', color='blue', linewidth=2)
    plt.plot(np.arange(1, num_games + 1), cumulative_winnings_2, label='Basic Model', color='red', linewidth=2)
    plt.xlabel('Number of games', fontsize=20)
    plt.ylabel('Amount of chips won', fontsize=20)
    plt.ylim(0, target_winnings_1 + 50000)
    plt.xlim(0, num_games - 1000)
    plt.xticks(fontsize=18)  # 设置横坐标刻度字体大小
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.show()
    print(cumulative_winnings_2)


if __name__ == '__main__':
    # print(3000 / 800 ** 0.2)
    func1()
    # func2()
    # func3()
    # func4()
    pass

