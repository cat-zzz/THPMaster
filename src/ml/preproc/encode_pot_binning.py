"""
@project: THPMaster
@File   : encode_pot_binning.py
@Desc   : 底池筹码分箱和编码
@Author : gql
@Date   : 2024/9/8 15:10
"""
import math
import os

import numpy as np
from matplotlib import pyplot as plt


def statics_pgn_pot_distribution(data_dir):
    """
    统计data_dir目录下的所有对局文件，计算每个底池筹码出现的次数
    """
    '''
    返回值pot_mapping的说明信息
    1. 统计的是最后输赢筹码的两倍作为底池，所以会有平局的情况导致输赢筹码为0
    2. pot_mapping的index对应的是底池大小，value对应的是该底池大小出现的次数
    '''
    pot_mapping = np.zeros(40001, dtype=int)
    length = len(os.listdir(data_dir))
    # 1 读取文件
    for i, filename in enumerate(os.listdir(data_dir)):
        # 2 遍历每个文件下的历史对局
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r') as file:
            histories = file.readlines()
        histories = histories[4:]  # 前4行是一些说明信息，不需要
        drop_list = [0, 1, 2, 3, 5]  # 原数据集中不需要的列
        histories = [np.delete(_.split(':'), drop_list).tolist() for _ in histories]
        histories[0][0].split('|')
        for history in histories:
            pot = history[0].split('|')[0]
            pot = int(math.fabs(int(pot)) * 2)
            pot_mapping[pot] += 1
        # 统计信息
        if (i + 1) % 30 == 0 or i >= length - 10:
            print(f'----已读取{i + 1}/{length}个文件----')
            n = 50
            # 使用 argpartition 获取前 n 个最大值的索引（无序）
            top_n_indices_unordered = np.argpartition(-pot_mapping, n - 1)[:n]
            # 对前 n 个最大值的索引按大小进行排序（从大到小）
            top_n_indices = top_n_indices_unordered[np.argsort(-pot_mapping[top_n_indices_unordered])]
            # 获取对应的值
            top_n_values = pot_mapping[top_n_indices]
            print('indices:', top_n_indices)
            print('values:', top_n_values)
    # 保存为 .npy 文件
    np.save('pot_mapping_array.npy', pot_mapping)
    return pot_mapping


def load_pot_mapping_array(filepath='pot_mapping_array.npy'):
    """
    加载pot_mapping_array.npy文件
    """
    pot_mapping = np.load(filepath, 'r')
    n = 500
    # 使用 argpartition 获取前 n 个最大值的索引（无序）
    top_n_indices_unordered = np.argpartition(-pot_mapping, n - 1)[:n]
    # 对前 n 个最大值的索引按大小进行排序（从大到小）
    top_n_indices = top_n_indices_unordered[np.argsort(-pot_mapping[top_n_indices_unordered])]
    # 获取对应的值
    top_n_values = pot_mapping[top_n_indices]
    print('indices:', top_n_indices)
    print('values:', top_n_values)
    return pot_mapping


def fixed_interval_pot_mapping(weights, interval=200, output_file="fix_interval_binning_results.npy"):
    """
    按照固定间隔对筹码进行分箱，并将结果保存为npy文件。此处只是初步分箱，否则200~40000之间的数量过多，难以统计
    :param weights: 表示筹码值的数组，下标为筹码值，元素为出现次数。
    :param interval: 每隔多少筹码分一个箱，默认值为200
    :param output_file: 保存结果的npy文件名
    :return: 每个区间的权重（区间内所有筹码出现次数之和）
    """
    min_chip = 200
    max_chip = 40000

    # 计算分箱边界，每隔固定间隔(200)设置一个分箱边界
    bin_edges = np.array(list(range(min_chip, max_chip + 2, interval)))
    # 确保最大值40000包含在最后一个分箱边界中
    if bin_edges[-1] != max_chip:
        bin_edges = np.append(bin_edges, max_chip)
    print("固定间隔的分箱边界:", bin_edges)

    # 统计每个分箱的权重
    binned_indices = np.digitize(np.arange(min_chip, max_chip + 1), bin_edges) - 1
    binned_indices[-1] = binned_indices[-2]
    bin_counts = np.array(
        [np.sum(weights[min_chip:max_chip + 1][binned_indices == i]) for i in range(len(bin_edges) - 1)])
    print("每个分箱的权重统计:", bin_counts)

    # 保存分箱边界和权重统计为 NumPy 数组到 npy 文件
    np.save(output_file, bin_counts)
    print(f"分箱边界和权重统计已保存为 NumPy 数组，文件: {output_file}")
    return bin_counts, bin_edges


def find_weighted_median(weights):
    """
    找出筹码权重的中点（累计总权重达到一半时的筹码值）
    :param weights: 表示筹码值的数组，下标为筹码值，元素为出现次数
    :return: 中点筹码值
    """
    min_chip = 200
    max_chip = 40000
    # 计算总权重
    total_weight = np.sum(weights[min_chip:max_chip + 1])
    print('total weight:', total_weight)
    half_weight = total_weight * 2 / 3  # 找到总权重的2/3
    # 累积权重并找到中点
    cumulative_weight = 0
    for chip in range(min_chip, max_chip + 1):
        cumulative_weight += weights[chip]
        if cumulative_weight >= half_weight:
            print(f"中点筹码值: {chip}")
            return chip  # 返回中点筹码值
    return None  # 如果未找到，返回None


def chip_encoding_adaptive(chip_value, bins, alpha=0.5, threshold=100):
    chip_value = np.array(chip_value)

    # Step 1: 计算每个分箱的中心点作为 μ
    mu = bins  # 将 bins 作为分箱中心点的数组

    # Step 2: 根据分箱间距调节 distance_factor
    distance_factor = np.diff(bins, prepend=0)  # 计算每个分箱之间的间距
    # distance_factor[0] = 200  # 起始筹码（200）需要特殊设置
    # distance_factor = distance_factor / 2
    # distance_factor[-2] = distance_factor[-2] / 2
    # distance_factor[-1] = distance_factor[-1] / 3
    # distance_factor = distance_factor / 4
    # distance_factor = np.sqrt(distance_factor*4)
    distance_factor = np.log(distance_factor) + 0.2 * distance_factor

    distance_factor = [40, 50, 50, 100, 200, 400, 1800, 4500]
    # distance_factor = [40, 50, 50, 100, 200, 400, 1500, 6000]
    # todo (理论上不可行，)加上一个很小的正数偏移，避免 distance_factor 为零的情况
    distance_factor = np.where(distance_factor == 0, 1e-8, distance_factor)

    # Step 3: 根据筹码值是否低于阈值，决定不同的权重计算方式
    if chip_value < threshold:
        # 当筹码值小于阈值时，仅考虑密集分箱的权重
        dense_bins = bins[:7]  # 仅考虑 200 到 3400 分箱
        mu = dense_bins
        distance_factor = distance_factor[:7]
        distances = np.abs(chip_value - mu)
        print(f'distances:{distances}\ndistance_factor:{distance_factor}')
        print(
            f'x:{-alpha * distances / distance_factor}, exp:{np.exp(-alpha * distances / distance_factor)}')
        weights = np.exp(-alpha * distances / distance_factor)  # 使用指数衰减函数
        weights /= weights.sum()  # 归一化处理

        # 扩展到8维，将10800和40000的权重设为0
        extended_weights = np.zeros(len(bins))
        extended_weights[:7] = weights
        return extended_weights
    else:
        # 当筹码值大于阈值时，考虑所有分箱
        distances = np.abs(chip_value - mu)
        weights = np.exp(-alpha * distances / distance_factor)  # 使用指数衰减函数
        weights /= weights.sum()  # 归一化处理
        print(f'distances:{distances}\ndistance_factor:{distance_factor}')
        print(
            f'x:{-alpha * distances / distance_factor}\nexp:{np.exp(-alpha * distances / distance_factor)}')
        return weights


def chip_encoding_dynamic(chip_value, bins, alpha=0.5, threshold=100):
    chip_value = np.array(chip_value)

    # Step 1: 计算每个分箱的中心点作为 μ
    mu = bins  # 将 bins 作为分箱中心点的数组

    # Step 2: 计算每个分箱的大小
    bin_size = np.diff(bins, prepend=0)  # 计算每个分箱之间的大小

    # Step 3: 动态调整 distance_factor，根据筹码值与分箱的距离
    # f(chip_value, mu) 是根据筹码值调整的非线性部分
    distance_factor = np.log(bin_size + 1) + np.abs(chip_value - mu) / bin_size

    # 防止 distance_factor 太小
    distance_factor = np.where(distance_factor == 0, 1e-8, distance_factor)

    # Step 4: 根据筹码值是否低于阈值，决定不同的权重计算方式
    if chip_value < threshold:
        # 当筹码值小于阈值时，仅考虑密集分箱的权重
        dense_bins = bins[:7]  # 仅考虑 200 到 3400 分箱
        mu = dense_bins
        distance_factor = distance_factor[:7]
        distances = np.abs(chip_value - mu)
        weights = np.exp(-alpha * distances / distance_factor)  # 使用指数衰减函数
        weights /= weights.sum()  # 归一化处理

        # 扩展到8维，将10800和40000的权重设为0
        extended_weights = np.zeros(len(bins))
        extended_weights[:7] = weights
        return extended_weights
    else:
        # 当筹码值大于阈值时，考虑所有分箱
        distances = np.abs(chip_value - mu)
        weights = np.exp(-alpha * distances / distance_factor)  # 使用指数衰减函数
        weights /= weights.sum()  # 归一化处理
        return weights


# 采用此方法
def chip_encoding_with_distance(chip_value, bins=None, alpha=1):
    if bins is None:
        bins = [200, 400, 600, 1000, 1800, 3400, 10800, 40000]
    chip_value = np.array(chip_value)
    distances = np.abs(chip_value - bins)
    distances = distances / chip_value * 8
    weights = np.exp(-alpha * distances)
    # print(f'-----------------------\noriginal:{np.abs(chip_value - bins)}\noriginal2:'
    #       f'{np.abs(chip_value - bins) / chip_value}\ndistances:{distances}\nweights:{weights}')
    weights /= weights.sum()  # 归一化处理
    return weights


def run_chip_encoding():
    np.set_printoptions(suppress=True)  # 禁用科学计数法，确保显示为普通小数形式
    bins = [200, 400, 600, 1000, 1800, 3400, 10800, 40000]
    chip = 200
    distance_factor = np.diff(bins, prepend=bins[0])  # 计算每个分箱之间的间距
    print('distance factor:', distance_factor)
    for _ in range(50):
        result = chip_encoding_with_distance(chip, bins)
        print(f'--------chip={chip}, result={result}\n')
        chip += 10
    for _ in range(99):
        result = chip_encoding_with_distance(chip, bins)
        print(f'--------chip={chip}, result={result}\n')
        chip += 400
    chip = 40000
    result = chip_encoding_with_distance(chip, bins)
    print(f'chip={chip}, result={result}\n')


if __name__ == '__main__':
    run_chip_encoding()
    pass
