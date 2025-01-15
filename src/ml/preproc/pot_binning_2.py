"""
@project: THPMaster
@File   : pot_binning_2.py
@Desc   :
@Author : gql
@Date   : 2024/9/4 15:52
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


def analyze_pot_mapping_7(weights, num_bins=10):
    min_chip = 200
    max_chip = 40000

    # 计算总权重
    total_weight = np.sum(weights[min_chip:max_chip + 1])
    target_weight_per_bin = total_weight / (num_bins - 1)  # 目标分配的权重，每个分箱应包含的权重
    print(weights[200:250])

    # 初始化分箱边界
    bin_edges = [min_chip]
    current_weight = 0

    # 标记前一个非零权重区间的终点
    previous_nonzero_chip = min_chip

    for chip in range(min_chip, max_chip + 1):
        if weights[chip] == 0:
            print(f'跳过{chip}')
            # 遇到权重为0时，跳过，但保持在前一个有效分箱内
            continue

        current_weight += weights[chip]

        # 当达到目标权重时，添加分箱边界
        if current_weight >= target_weight_per_bin:
            # 在前一个非零区间处设置边界
            bin_edges.append(previous_nonzero_chip)
            current_weight = 0

        # 更新最近的非零权重区间的终点
        previous_nonzero_chip = chip

    # 最后添加最大筹码值的边界
    if bin_edges[-1] != max_chip:
        bin_edges.append(max_chip)

    # 确保分箱数量达到目标
    while len(bin_edges) - 1 < num_bins:
        # 计算每个区间的权重
        bin_weights = [np.sum(weights[bin_edges[i]:bin_edges[i + 1]]) for i in range(len(bin_edges) - 1)]
        max_weight_index = np.argmax(bin_weights)

        # 分割权重最大的区间
        start, end = bin_edges[max_weight_index], bin_edges[max_weight_index + 1]
        mid = (start + end) // 2
        bin_edges.insert(max_weight_index + 1, mid)

    # 确保分箱边界唯一且有序
    bin_edges = sorted(set(bin_edges))

    print("最终的分箱边界:", bin_edges)

    # 统计每个分箱的权重
    binned_indices = np.digitize(np.arange(min_chip, max_chip + 1), bin_edges) - 1
    bin_counts = [np.sum(weights[min_chip:max_chip + 1][binned_indices == i]) for i in range(len(bin_edges) - 1)]
    print("每个分箱的权重统计:", bin_counts)


def fixed_interval_pot_mapping(weights, interval=200, output_file="fix_interval_binning_results.npy"):
    """
    按照固定间隔对筹码进行分箱，并将结果保存为npy文件
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


def pot_mapping_binning(binning_results, num_bins=9, alpha=10):
    """
    根据权重数组对区间进行分箱，使前面的区间分配更大的权重，后面的区间分配更小的权重。
    :param binning_results: 表示每个区间权重之和的数组
    :param num_bins: 需要的分箱数量
    :param alpha: 前后区间权重比例衰减的系数，默认值为10。越大表示前面的区间分配更多的权重。
    :return: 分箱边界以及每个分箱的权重统计
    """
    # 计算总权重
    total_weight = np.sum(binning_results)

    # 使用指数函数来生成前大后小的目标权重比例
    weight_factors = np.exp(np.linspace(np.log(alpha), 0, num_bins))
    weight_factors = weight_factors / np.sum(weight_factors)  # 归一化，使其加和为1
    print('weight factors:', weight_factors)

    # 计算每个分箱的目标权重
    target_weights_per_bin = total_weight * weight_factors
    print(f"总权重: {total_weight}")
    print(f"每个分箱的目标权重: {target_weights_per_bin}")
    # 初始化变量
    current_weight = 0
    bin_edges = [0]  # 起始边界为第一个区间
    bin_weights = []  # 用来存储每个分箱的权重
    current_bin = 0  # 记录当前处理的是第几个分箱

    # 累积权重并进行分箱
    for i, weight in enumerate(binning_results):
        current_weight += weight
        # 如果累积的权重大于等于当前分箱的目标权重，划分一个新的分箱
        if current_weight >= target_weights_per_bin[current_bin]:
            bin_edges.append(i + 1)  # 将当前区间作为新的分箱边界
            bin_weights.append(current_weight)  # 保存当前分箱的权重
            current_weight = 0  # 重置累积权重
            current_bin += 1  # 进入下一个分箱

            # 如果已经完成所有分箱，提前结束
            if current_bin >= num_bins:
                break
    # 如果还有剩余权重，合并到最后一个分箱
    if current_weight > 0 and current_bin < num_bins:
        bin_weights.append(current_weight)
        bin_edges.append(len(binning_results))
    print('分箱边界:', bin_edges)
    print('每个分箱的权重:', bin_weights)
    return bin_edges, bin_weights


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


def pot_encoding_with_gaussian(chips, intervals, k=0.1):
    print()
    # 计算每个区间的中心点
    # centers = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]
    centers = intervals

    # 初始化权重列表
    weights = np.zeros(len(intervals))

    # 计算每个区间的长度
    lengths = [(intervals[i + 1] - intervals[i]) for i in range(len(intervals) - 1)]

    # 处理第一个区间的边界情况（使用第一个区间的边界值）
    print('intervals:', intervals)
    print('lengths:', lengths)
    sigma = k * (intervals[1] - intervals[0])  # 动态 sigma 值
    distance = (chips - intervals[0]) / max(intervals[0], 1)
    weights[0] = np.exp(-0.5 * (distance / sigma) ** 2)
    print('sigma1:', sigma, 'weight1:', weights[0])

    # 计算筹码值与每个中心点的距离，并应用高斯函数进行加权
    for i in range(0, len(intervals) - 1):
        if chips >= 20000:
            sigma = 0.64 * k * k * k * lengths[i - 1]  # 根据区间长度调整 sigma
        elif chips >= 12000:
            sigma = 0.65 * k * k * k * lengths[i - 1]
        else:
            sigma = 0.66 * k * k * k * lengths[i - 1]
        distance = (chips - centers[i - 1]) / max(centers[i - 1], 1)
        weights[i] = np.exp(-0.5 * (distance / sigma) ** 2)
        if chips == 17800 or chips == 17400:
            print('distance:', distance, 'sigma2:', sigma, 'exp2:', -0.5 * (distance / sigma) ** 2,
                  np.exp(-0.5 * (distance / sigma) ** 2), 'weight1:', weights[i])

    # 处理最后一个区间的边界情况（使用最后一个区间的边界值）
    sigma = k * (intervals[-1] - intervals[-2])  # 动态 sigma 值
    distance = (chips - intervals[-1]) / max(intervals[-1], 1)
    weights[-1] = np.exp(-0.5 * (distance / sigma) ** 2)
    # print('sigma3:', sigma, 'weight3:', weights[-1])

    # 归一化权重，确保权重和为1
    weights = weights / np.sum(weights)

    # 保留小数点后四位
    weights = np.round(weights, 4)

    return weights


def pot_encoding_with_gaussian_5(chip, intervals, k=0.6):
    print(f'intervals:{intervals}')
    weights = np.zeros(len(intervals))
    distances = np.abs(np.array(intervals) - chip)

    # 找到最小距离对应的索引
    closest_index = np.argmin(distances)
    a = intervals[closest_index]
    for i in range(len(intervals)):
        if chip >= 10800:
            sigma = k * ((chip + intervals[i]) / 4000) ** 1
        else:
            sigma = k * ((chip + intervals[i]) / 4000) ** 1
        # if chip < 5000:
        #     sigma = k * (intervals[i] / 1000) ** 1.05
        #     # sigma = k * intervals[i] / 1000
        # elif chip > 20000:
        #     sigma = k * (intervals[i] / 1000) ** 1.05
        #     # sigma = k * intervals[i] / 950
        # else:
        #     sigma = k * (intervals[i] / 1000) ** 1.05
        # sigma = k * intervals[i] / 1000
        # distance = (chip - intervals[i]) / max(intervals[i], 1)
        # 以chip最靠近的区间为分母
        distance = (chip - intervals[i]) / chip * 20
        # distance = (chip - intervals[i]) / a * 20
        weights[i] = np.exp(-0.5 * (distance / sigma) ** 2)
        print(
            f'i:{i}, a:{a}, distance:{distance}, sigma:{sigma}, exp: {-0.5 * (distance / sigma) ** 2}, weight:{weights[i]}')
    print('原始weights:', weights)
    weights = weights / np.sum(weights)
    return weights


def pot_encoding_with_gaussian_4(chips, intervals, k=0.1, sigma=200, threshold=5000):
    # 初始化权重列表
    weights = np.zeros(len(intervals) - 1)

    for i in range(len(intervals) - 1):
        # 计算筹码与区间左边界的距离
        # distance = chips - intervals[i]
        if chips > threshold:
            distance = (chips - intervals[i]) / max(intervals[i - 1], 0)
        else:
            distance = (chips - intervals[i]) / (intervals[i + 1] - intervals[i])
        # 根据索引判断密集区间和稀疏区间
        if i < 5:  # 假设前5个区间为密集区间
            # 密集区间：使用双曲正切函数计算权重
            weight = 0.5 * (1 + np.tanh(-k * distance))
        else:
            # 稀疏区间：使用高斯分布计算权重
            weight = np.exp(-0.5 * (distance / sigma) ** 2)

        # 如果距离非常大，设置权重为 0
        if distance > threshold:
            weight = 0
        weights[i] = weight
        print('i:', i, 'distance:', distance, 'exp:', -0.5 * (distance / sigma) ** 2, 'weight: ', weights[i])

    # 归一化权重
    weights = weights / np.sum(weights)
    return weights


def pot_encoding_with_gaussian_3(chips, intervals, k=0.1):
    # 计算每个区间的中心点
    centers = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]
    # centers = intervals

    # 初始化权重列表
    weights = np.zeros(len(intervals))

    # 计算每个区间的长度
    lengths = [(intervals[i + 1] - intervals[i]) for i in range(len(intervals) - 1)]

    # 处理第一个区间的边界情况（使用第一个区间的边界值）
    print('intervals:', intervals)
    print('lengths:', lengths)
    sigma = k * (intervals[1] - intervals[0])  # 动态 sigma 值
    distance = (chips - intervals[0]) / max(intervals[0], 1)
    weights[0] = np.exp(-0.5 * (distance / sigma) ** 2)
    print('sigma1:', sigma, 'weight1:', weights[0])

    # 计算筹码值与每个中心点的距离，并应用高斯函数进行加权
    for i in range(1, len(intervals) - 1):
        sigma = k * k * k * lengths[i - 1]  # 根据区间长度调整 sigma
        distance = (chips - centers[i - 1]) / max(centers[i - 1], 1)
        weights[i] = np.exp(-0.5 * (distance / sigma) ** 2)
        # todo np.exp(-0.5 * (distance / sigma) ** 2)计算有问题
        print('distance:', distance, 'sigma2:', sigma, 'exp2:', -0.5 * (distance / sigma) ** 2,
              np.exp(-0.5 * (distance / sigma) ** 2), 'weight1:', weights[i])

    # 处理最后一个区间的边界情况（使用最后一个区间的边界值）
    sigma = k * (intervals[-1] - intervals[-2])  # 动态 sigma 值
    distance = (chips - intervals[-1]) / max(intervals[-1], 1)
    weights[-1] = np.exp(-0.5 * (distance / sigma) ** 2)
    print('sigma3:', sigma, 'weight3:', weights[-1])

    # 归一化权重，确保权重和为1
    weights = weights / np.sum(weights)

    # 保留小数点后四位
    weights = np.round(weights, 4)

    return weights


# 分箱区间


# 编码算法
def chip_encoding(chip_value, bins, alpha=0.1):
    # Step 1: 确定所在分箱
    bin_idx = np.digitize(chip_value, bins) - 1  # 确定筹码所在的区间索引
    if bin_idx >= len(bins) - 1:  # 边界处理
        bin_idx = len(bins) - 2

    # 获取当前分箱的上下界
    lower_bound = bins[bin_idx]
    upper_bound = bins[bin_idx + 1]

    # Step 2: 计算筹码在区间中的比例
    ratio = (chip_value - lower_bound) / (upper_bound - lower_bound)

    # 步幅调整: 使用指数变换来增大步幅
    step = 1 - np.exp(-alpha * ratio)

    # Step 3: 计算当前区间和相邻区间的权重
    current_weight = 1 - step  # 当前区间的权重
    previous_weight = step  # 前一个区间的权重

    # Step 4: 根据步幅调整权重
    encoding_result = np.zeros(len(bins) - 1)
    encoding_result[bin_idx] = current_weight
    if bin_idx > 0:
        encoding_result[bin_idx - 1] = previous_weight
    print(f'chip={chip_value}, 编码结果:{encoding_result}')

    return encoding_result


def chip_encoding_2(chip_value, bins):
    chip_value = np.array(chip_value)
    # Step 1: 计算每个分箱的中心点作为 μ
    mu = bins  # 将 bins 作为分箱中心点的数组

    # Step 2: 根据分箱间距调节 distance_factor
    distance_factor = np.diff(bins, prepend=bins[0])  # 计算每个分箱之间的间距

    # 加上一个很小的正数偏移，避免 distance_factor 为零的情况
    distance_factor = np.where(distance_factor == 0, 1e-8, distance_factor)

    # Step 3: 计算权重衰减，基于距离和分箱间距控制
    distances = np.abs(chip_value - mu)
    weights = (1 / distance_factor) * np.exp(-distances / distance_factor)  # 使用距离与间距结合控制权重

    # Step 4: 对权重进行归一化
    weights /= weights.sum()

    return weights


def chip_encoding_3(chip_value, bins, alpha=1):
    chip_value = np.array(chip_value)

    # Step 1: 计算每个分箱的中心点作为 μ
    mu = bins  # 将 bins 作为分箱中心点的数组

    # Step 2: 根据分箱间距调节 distance_factor
    distance_factor = np.diff(bins, prepend=bins[0])  # 计算每个分箱之间的间距

    # 加上一个很小的正数偏移，避免 distance_factor 为零的情况
    distance_factor = np.where(distance_factor == 0, 1e-8, distance_factor)

    # Step 3: 使用非线性距离处理（比如距离平方），加大距离差异的影响
    distances = np.abs(chip_value - mu) ** 2  # 使用距离的平方来增强距离影响
    weights = np.exp(-alpha * distances / distance_factor)  # 使用指数衰减函数，但放大距离影响

    # Step 4: 对权重进行归一化，确保分箱总和为 1
    weights /= weights.sum()

    return weights


def chip_encoding_adaptive(chip_value, bins, alpha=0.5, threshold=10800):
    chip_value = np.array(chip_value)

    # Step 1: 计算每个分箱的中心点作为 μ
    mu = bins  # 将 bins 作为分箱中心点的数组

    # Step 2: 根据分箱间距调节 distance_factor
    distance_factor = np.diff(bins, prepend=bins[0])  # 计算每个分箱之间的间距
    distance_factor = distance_factor / 2
    # todo 修改
    distance_factor[0] = 200

    # 加上一个很小的正数偏移，避免 distance_factor 为零的情况
    distance_factor = np.where(distance_factor == 0, 1e-8, distance_factor)

    # Step 3: 根据筹码值是否低于阈值，决定不同的权重计算方式
    if chip_value < threshold:
        # 当筹码值小于阈值时，仅考虑密集分箱的权重
        dense_bins = bins[:6]  # 仅考虑 200 到 3400 分箱
        mu = dense_bins
        distance_factor = distance_factor[:6]
        distances = np.abs(chip_value - mu)
        print(f'distances:{distances}, distance_factor:{distance_factor}')
        print(
            f'x:{-alpha * distances / distance_factor}, exp:{np.exp(-alpha * distances / distance_factor)}')
        weights = np.exp(-alpha * distances / distance_factor)  # 使用指数衰减函数
        weights /= weights.sum()  # 归一化处理

        # 扩展到8维，将10800和40000的权重设为0
        extended_weights = np.zeros(len(bins))
        extended_weights[:6] = weights
        return extended_weights
    else:
        # 当筹码值大于阈值时，考虑所有分箱
        distances = np.abs(chip_value - mu)
        weights = np.exp(-alpha * distances / distance_factor)  # 使用指数衰减函数
        weights /= weights.sum()  # 归一化处理
        return weights


def chip_encoding_adaptive_2(chip_value, bins, alpha=0.5, dense_alpha=1, threshold=3400):
    chip_value = np.array(chip_value)

    # Step 1: 计算每个分箱的中心点作为 μ
    mu = bins  # 将 bins 作为分箱中心点的数组

    # Step 2: 根据分箱间距调节 distance_factor
    distance_factor = np.diff(bins, prepend=bins[0])  # 计算每个分箱之间的间距

    # 加上一个很小的正数偏移，避免 distance_factor 为零的情况
    distance_factor = np.where(distance_factor == 0, 1e-8, distance_factor)

    # Step 3: 根据筹码值是否低于阈值，决定不同的权重计算方式
    if chip_value < threshold:
        # 当筹码值小于阈值时，仅考虑密集分箱的权重
        dense_bins = bins[:6]  # 仅考虑 200 到 3400 分箱
        mu = dense_bins
        distance_factor = distance_factor[:6]
        distances = np.abs(chip_value - mu)

        # 使用专门为密集区间设置的 dense_alpha 来调整权重
        weights = np.exp(-dense_alpha * distances / distance_factor)
        weights /= weights.sum()  # 归一化处理

        # 扩展到8维，将10800和40000的权重设为0
        extended_weights = np.zeros(len(bins))
        extended_weights[:6] = weights
        return extended_weights
    else:
        # 当筹码值大于阈值时，考虑所有分箱
        distances = np.abs(chip_value - mu)
        weights = np.exp(-alpha * distances / distance_factor)  # 使用原始的 alpha 控制衰减
        weights /= weights.sum()  # 归一化处理
        return weights


def run():
    # 根据即使ACPC数据计算底池分布，确定底池筹码间隔
    # pm = load_pot_mapping_array() # 后续用不到pot_mapping_array.npy，而是使用fix_interval_binning_results.npy
    # 统计pm中非零元素，仅用于查看信息
    # non_zero_indices = np.nonzero(pm)
    # non_zero_indices = np.array(non_zero_indices)
    # non_zero_values = pm[non_zero_indices]
    # 生成fix_interval_binning_results.npy文件，只需运行一次
    # counts, edges = fixed_interval_pot_mapping(pm)

    # 加载fix_interval_binning_results.npy文件
    fib_pot = np.load('fix_interval_binning_results.npy', 'r')
    # todo 此处设置的num_bins=9, 但是最后结果只能分出8个箱，这是由于后面的权重较小，即使把后面的权重全加一起也凑不够一个箱的权重
    # 我们虽然已经设置了动态权重，目前权重是基于指数函数的分配的，每前面的箱的权重较大，后面的箱的权重较小，
    # 但是由于后面区间的权重普遍只有几千（40000筹码区间除外），而开头的权重有一百多万，导致后面的区间权重全加一起也凑不够一个箱的权重
    # 运行的结果为[0, 1, 2, 4, 8, 16, 53, 199]，对应筹码值为[200 400 600 1000 1800 3400 10800 40000]
    pot_mapping_binning(fib_pot, alpha=10)  # 基于固定间隔区间的动态权重分箱算法
    fib_pot = fib_pot[::-1]
    print('倒叙之后的结果')
    pot_mapping_binning(fib_pot, alpha=10)


def pot_encoding_with_gaussian_2(chips, intervals, k=0.1):
    # 计算每个区间的中心点
    centers = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]

    # 初始化权重列表
    weights = np.zeros(len(intervals))

    # 计算每个区间的长度
    lengths = [(intervals[i + 1] - intervals[i]) for i in range(len(intervals) - 1)]

    # 处理第一个区间的边界情况
    if chips <= intervals[0]:
        weights[0] = 1.0
    # 处理最后一个区间的边界情况
    elif chips >= intervals[-1]:
        weights[-1] = 1.0
    else:
        # 处理筹码落在区间内部的情况
        for i in range(len(intervals) - 1):
            if intervals[i] <= chips < intervals[i + 1]:
                # 给当前区间一个较大的权重
                weights[i] = 0.5

                # 应用高斯分布衰减相邻区间的影响
                # 距离中心的扩散程度由区间长度决定
                sigma = k * lengths[i]  # 使用区间长度动态调整 sigma

                # 对相邻的下一个区间进行加权
                if i < len(intervals) - 1:
                    distance_next = (chips - intervals[i + 1]) / max(intervals[i + 1], 1)
                    weights[i + 1] = np.exp(-0.5 * (distance_next / sigma) ** 2)

                # 对相邻的上一个区间进行加权
                if i > 0:
                    distance_prev = (chips - intervals[i - 1]) / max(intervals[i - 1], 1)
                    weights[i - 1] = np.exp(-0.5 * (distance_prev / sigma) ** 2)

                break

    # 归一化权重，确保权重和为1
    weights = weights / np.sum(weights)

    # 保留小数点后四位
    weights = np.round(weights, 4)

    return weights


def run_encode_pot():
    # 禁用科学计数法，确保显示为普通小数形式
    np.set_printoptions(suppress=True)
    intervals = [200, 400, 600, 1000, 1800, 3400, 10800, 40000]
    chip = 400
    for _ in range(100):
        print(f'chip: {chip}, 编码结果为{pot_encoding_with_gaussian_5(chip, intervals)}\n')
        chip += 400


def exp_func():
    # 固定距离
    # distance = 0.5

    # 不同的sigma值
    # sigma_values = np.linspace(0.1, 5, 100)
    sigma_values = 1
    distance = np.linspace(0.1, 10, 200)

    # 计算每个sigma对应的f值
    f_values = np.exp(-0.5 * (distance / sigma_values) ** 2)

    # 绘制f与sigma的关系曲线
    # plt.plot(sigma_values, f_values)
    plt.plot(distance, f_values)
    plt.xlabel('distance')
    plt.ylabel('f')
    # plt.title(f'distance:{distance}')
    plt.title('gaussian func')
    plt.grid(True)
    plt.show()


def run_chip_encoding():
    np.set_printoptions(suppress=True)
    bins = [200, 400, 600, 1000, 1800, 3400, 10800, 40000]
    chip = 201
    distance_factor = np.diff(bins, prepend=bins[0])  # 计算每个分箱之间的间距
    print('distance factor:', distance_factor)
    for _ in range(50):
        result = chip_encoding_adaptive(chip, bins)
        print(f'chip={chip}, result={result}\n')
        chip += 10
    for _ in range(99):
        result = chip_encoding_adaptive(chip, bins)
        print(f'chip={chip}, result={result}\n')
        chip += 400
    chip = 40000
    result = chip_encoding_adaptive(chip, bins)
    print(f'chip={chip}, result={result}\n')


if __name__ == '__main__':
    # run()
    # run_encode_pot()
    # exp_func()
    run_chip_encoding()
    pass
