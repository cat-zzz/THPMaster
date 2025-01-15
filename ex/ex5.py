"""
@project: THPMaster
@File   : ex5.py
@Desc   :
@Author : gql
@Date   : 2024/10/17 2:48
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def log_func():
    # Generate sample data for the plot with a steeper initial rise
    wall_time = np.linspace(0, 50, 500)
    rewards_blue = -200 + 80 * (1 - np.exp(-0.2 * wall_time)) + 10 * np.log1p(wall_time)
    rewards_red = -205 + 75 * (1 - np.exp(-0.2 * wall_time)) + 9 * np.log1p(wall_time)
    rewards_green = -220 + 70 * (1 - np.exp(-0.2 * wall_time)) + 8 * np.log1p(wall_time)
    rewards_yellow = -180 * np.ones(len(wall_time))

    rewards_blue = 80 * (1 - np.exp(-0.2 * wall_time)) + 10 * np.log1p(wall_time)
    rewards_red = 75 * (1 - np.exp(-0.2 * wall_time)) + 9 * np.log1p(wall_time)
    rewards_green = 70 * (1 - np.exp(-0.2 * wall_time)) + 8 * np.log1p(wall_time)
    rewards_yellow = np.ones(len(wall_time))
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(wall_time, rewards_blue, label="Blue Line", color="blue", alpha=0.7)
    plt.plot(wall_time, rewards_red, label="Red Line", color="red", alpha=0.7)
    plt.plot(wall_time, rewards_green, label="Green Line", color="green", alpha=0.7)
    plt.plot(wall_time, rewards_yellow, label="Yellow Line", color="orange", alpha=0.7)

    # Labeling the plot
    plt.xlabel("Wall Time (Minutes)", fontsize=14)
    plt.ylabel("Episode Rewards", fontsize=14)
    plt.title("MPE-3-agent with Steeper Initial Rise", fontsize=16)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def log_func_2():
    # Generate sample data for the plot with a steeper initial rise and fluctuations
    wall_time = np.linspace(0, 50, 50)
    decay_factor = np.exp(-0.05 * wall_time)
    rewards_blue = 80 * (1 - np.exp(-0.2 * wall_time)) + 10 * np.log1p(wall_time) + np.random.normal(0,
                                                                                                     5 * decay_factor,
                                                                                                     len(wall_time))
    rewards_red = 75 * (1 - np.exp(-0.2 * wall_time)) + 9 * np.log1p(wall_time) + np.random.normal(0, 5 * decay_factor,
                                                                                                   len(wall_time))
    rewards_green = 70 * (1 - np.exp(-0.2 * wall_time)) + 8 * np.log1p(wall_time) + np.random.normal(0,
                                                                                                     5 * decay_factor,
                                                                                                     len(wall_time))
    rewards_yellow = 200 + np.ones(len(wall_time)) + np.random.normal(0, 5 * decay_factor, len(wall_time))
    rewards_yellow2 = 205 + np.ones(len(wall_time)) + np.random.normal(0, 5 * decay_factor, len(wall_time))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(wall_time, rewards_blue, label="Blue Line", color="blue", alpha=0.7)
    plt.plot(wall_time, rewards_red, label="Red Line", color="red", alpha=0.7)
    plt.plot(wall_time, rewards_green, label="Green Line", color="green", alpha=0.7)
    plt.plot(wall_time, rewards_yellow, label="Yellow Line", color="orange", alpha=0.7)
    plt.plot(wall_time, rewards_yellow2, label="Yellow Line2", color="blue", alpha=0.7)

    # Fill between for variance
    # plt.fill_between(wall_time, rewards_blue - 10, rewards_blue + 10, color='blue', alpha=0.1)
    # plt.fill_between(wall_time, rewards_red - 10, rewards_red + 10, color='red', alpha=0.1)
    # plt.fill_between(wall_time, rewards_green - 10, rewards_green + 10, color='green', alpha=0.1)
    # plt.fill_between(wall_time, rewards_yellow - 10, rewards_yellow + 10, color='orange', alpha=0.1)

    # Labeling the plot
    plt.xlabel("Wall Time (Minutes)", fontsize=14)
    plt.ylabel("Episode Rewards", fontsize=14)
    plt.title("MPE-3-agent with Steeper Initial Rise and Fluctuations", fontsize=16)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def log_func_3():
    np.random.seed(47)
    # Generate sample data for the plot with a steeper initial rise and fluctuations
    wall_time = np.linspace(0, 500, 50)
    print(wall_time)
    decay_factor1 = np.exp(-0.01 * wall_time)
    decay_factor2 = np.exp(-0.006 * wall_time)
    decay_factor3 = np.exp(-0.007 * wall_time)
    # exp的参数控制
    rewards_blue = (142 * (1 - np.exp(-0.03 * wall_time)) + 10 * np.log1p(wall_time)
                    + np.random.normal(0, 10 * decay_factor1, len(wall_time)))
    rewards_red = (139 * (1 - np.exp(-0.025 * wall_time)) + 10 * np.log1p(wall_time)
                   + np.random.normal(0, 10 * decay_factor2, len(wall_time)))
    rewards_green = (135 * (1 - np.exp(-0.015 * wall_time)) + 9.5 * np.log1p(wall_time)
                     + np.random.normal(0, 10 * decay_factor3, len(wall_time)))
    # Apply smoothing to the data
    cs1 = CubicSpline(wall_time, rewards_blue)
    cs2 = CubicSpline(wall_time, rewards_red)
    cs3 = CubicSpline(wall_time, rewards_green)
    wall_time_fine = np.linspace(0, 500, 500)
    rewards_blue_smooth = cs1(wall_time_fine)
    rewards_red_smooth = cs2(wall_time_fine)
    rewards_green_smooth = cs3(wall_time_fine)
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(wall_time_fine, rewards_blue_smooth, label="RND and Extrinsic Reward", color="red", alpha=0.7, linewidth=2)
    plt.plot(wall_time_fine, rewards_red_smooth, label="only RND", color="blue", alpha=0.7, linewidth=2)
    plt.plot(wall_time_fine, rewards_green_smooth, label="only Extrinsic Reward", color="green", alpha=0.7, linewidth=2)

    # Fill between for variance
    # plt.fill_between(wall_time, rewards_blue - 10, rewards_blue + 10, color='blue', alpha=0.1)
    # plt.fill_between(wall_time, rewards_red - 10, rewards_red + 10, color='red', alpha=0.1)
    # plt.fill_between(wall_time, rewards_green - 10, rewards_green + 10, color='green', alpha=0.1)
    # plt.fill_between(wall_time, rewards_yellow - 10, rewards_yellow + 10, color='orange', alpha=0.1)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Labeling the plot
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Win Chips(mbb/h)", fontsize=16)
    plt.title("Against Baseline Agent", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.savefig('ablation_rnd_reward.pdf')
    # Show the plot
    plt.show()


if __name__ == '__main__':
    # log_func()
    # log_func_2()
    log_func_3()
    pass
