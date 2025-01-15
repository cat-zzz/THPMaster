"""
@project: THPMaster
@File   : ex6.py
@Desc   :
@Author : gql
@Date   : 2024/10/17 11:07
"""
import matplotlib.pyplot as plt
import numpy as np


def log_func_2():
    # Generate sample data for the plot with a steeper initial rise and fluctuations
    wall_time = np.linspace(0, 50, 50)
    decay_factor = np.exp(-0.05 * wall_time)
    rewards_yellow = -0.5 + np.ones(len(wall_time)) + np.random.normal(0, 0.1 * decay_factor, len(wall_time))
    rewards_yellow2 = -0.5 + np.ones(len(wall_time)) + np.random.normal(0, 0.1 * decay_factor, len(wall_time))

    # Plot the data
    plt.figure(figsize=(10, 6))
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


if __name__ == '__main__':
    log_func_2()
    pass
