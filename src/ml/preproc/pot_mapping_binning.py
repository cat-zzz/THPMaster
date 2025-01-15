"""
@project: machine_game
@File   : pot_mapping_binning.py
@Desc   :
@Author : gql
@Date   : 2024/6/20 18:23
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def encode_pot_sizes(pot_sizes, num_bins=13, dense_region_limit=5000, log_bins=7, kmeans_bins=5):
    """
    将底池筹码数据编码为指定维度的特征向量
    :param pot_sizes: 包含底池筹码数据的数组或列表
    :param num_bins: 最终的特征向量维数，默认为13
    :param dense_region_limit: 密集区域的上限值
    :param log_bins: 对数分箱的初步区间数
    :param kmeans_bins: 稀疏区域的K-means聚类数
    :return: 编码后的13维特征向量
    """
    # 步骤 1: 对数分箱 (Logarithmic Binning)
    min_value = 200
    max_value = 40000
    log_bins = np.logspace(np.log10(min_value), np.log10(max_value), num=log_bins)
    print("对数分箱区间:", log_bins)

    # 步骤 2: 细化密集区域的分箱 (Quantile Binning)
    dense_region = pot_sizes[(pot_sizes >= min_value) & (pot_sizes <= dense_region_limit)]
    num_bins_dense = num_bins - len(log_bins) + 1  # 根据最终维数调整密集区域的分箱数量
    quantile_bins_dense = pd.qcut(dense_region, q=num_bins_dense, duplicates='drop', retbins=True)[1]
    print("密集区域分位数分箱结果:", quantile_bins_dense)

    # 步骤 3: 优化稀疏区域的分箱 (K-means Clustering)
    sparse_region = pot_sizes[(pot_sizes > dense_region_limit) & (pot_sizes <= max_value)].reshape(-1, 1)
    kmeans = KMeans(n_clusters=kmeans_bins).fit(sparse_region)
    kmeans_centers = sorted(kmeans.cluster_centers_.flatten())
    print("稀疏区域K-means聚类中心:", kmeans_centers)

    # 步骤 4: 合并结果
    combined_bins = np.concatenate((quantile_bins_dense, kmeans_centers))
    combined_bins = np.sort(np.unique(combined_bins))  # 去除重复值并排序
    print("初步合并后的区间:", combined_bins)

    # 确保最终区间数为13个
    def merge_intervals_to_target_count(bins, target_count):
        while len(bins) > target_count:
            diffs = np.diff(bins)
            min_diff_index = np.argmin(diffs)
            bins = np.delete(bins, min_diff_index + 1)
        return bins

    final_bins = merge_intervals_to_target_count(combined_bins, num_bins)
    print("最终的区间:", final_bins)

    # 应用温度编码
    def temperature_encoding(chips, centers, temperature=1.0):
        distances = np.abs(centers - chips)
        exp_values = np.exp(-distances / temperature)
        encoded_vector = exp_values / np.sum(exp_values)
        return encoded_vector

    # 示例：编码一个底池筹码量
    encoded_vectors = [temperature_encoding(size, final_bins) for size in pot_sizes[:10]]  # 示例编码前10个筹码量
    for i, vec in enumerate(encoded_vectors):
        print(f"底池筹码: {pot_sizes[i]}, 编码后: {vec}")

    return encoded_vectors


# 2024/07/01新增
bin_edges = [200, 500, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 40000]
# [0.00012340980408667956, 0.0015034391929775724, 0.04393693362340742, 0.32465246735834974, 0.6013449152352569, 0.02843972957506904, 0.0009469910723220414, 3.4811068399043104e-06, 9.342813371521583e-09, 1.8334865483181855e-11, 2.699578503363014e-14, 2.8946403116483005e-17, 2.400196623397581e-20]
# [0.00012340980408667956, 0.0015034391929775724, 0.04393693362340742, 0.32465246735834974, 0.6013449152352569, 0.02843972957506904, 0.0009469910723220414, 3.4811068399043104e-06, 9.342813371521583e-09, 1.8334865483181855e-11, 2.699578503363014e-14, 2.8946403116483005e-17, 2.400196623397581e-20]

def encode_chips_gaussian(chip_amount, bin_edges, sigma=1000.0):
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    encoded = np.array([norm.pdf(chip_amount, loc=center, scale=sigma) for center in bin_centers])
    # 归一化
    encoded /= np.sum(encoded)
    return encoded.tolist()

def encode_chips_with_temperature(chip_amount, bin_edges, temperature=1.0, sigma=1000.0):
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    # 计算每个分箱的高斯分布权重
    encoded = np.array([norm.pdf(chip_amount, loc=center, scale=sigma) for center in bin_centers])
    encoded /= np.sum(encoded)
    # 应用温度参数
    encoded = np.exp(encoded / temperature)
    # 归一化
    encoded /= np.sum(encoded)
    return encoded.tolist()
# 示例
chip_amount = 3500
temperature = 0.01  # 温度参数
encoded_chips = encode_chips_gaussian(chip_amount, bin_edges)
print('高斯权重', encoded_chips)
encoded_chips = encode_chips_with_temperature(chip_amount, bin_edges, temperature)
print('高斯权重+温度编码', encoded_chips)


def analyze_pot_mapping(weights):
    print('weight:', weights)
    print('weight size:', weights.shape)
    # 计算累计权重
    cumulative_weights = np.cumsum(weights[0:6000])
    # 计算总权重
    total_weight = cumulative_weights[-1]
    # 每个分箱应包含的权重
    bin_weight = total_weight / 13
    # 初始化分箱边界列表
    bin_edges = [200]  # 起始边界,未考虑平局（weight[0]）
    print('total weight:', cumulative_weights)
    print('bin_weight:', bin_weight)
    # 当前累计权重
    current_cumulative_weight = 0
    for i in range(len(cumulative_weights)):
        current_cumulative_weight += weights[i + 200]
        # 检查是否超过了当前分箱的权重
        if current_cumulative_weight >= bin_weight:
            # 记录边界
            bin_edges.append(i + 200)
            # 重置当前累计权重
            current_cumulative_weight = 0
    # 如果最后一个边界不是40000，手动添加
    if bin_edges[-1] != 40000:
        bin_edges.append(40000)
    # 确保最终边界数为14个（包括起始和结束边界）
    while len(bin_edges) > 14:
        # 如果分箱数量多于14个，合并最小差值的分箱
        diffs = np.diff(bin_edges)
        min_diff_index = np.argmin(diffs)
        bin_edges.pop(min_diff_index + 1)

    while len(bin_edges) < 14:
        # 如果分箱数量少于14个，可能需要在权重分布均匀的地方插入一个边界
        # 这里可以根据实际情况具体调整
        # 例如，在中间插入一个点
        mid_point = (bin_edges[-2] + bin_edges[-1]) // 2
        bin_edges.insert(-1, mid_point)

    # 确保边界是唯一且有序的
    bin_edges = sorted(set(bin_edges))

    print("最终的分箱边界:", bin_edges)
    # 使用np.digitize进行分箱
    binned_indices = np.digitize(np.arange(200, 40001), bin_edges) - 1
    # 统计每个分箱中的数据和权重
    bin_counts = np.zeros(len(bin_edges) - 1)
    for i in range(len(bin_counts)):
        bin_counts[i] = np.sum(weights[200:40001][binned_indices == i])
    print("每个分箱的权重统计:", bin_counts)


def analyze_pot_mapping_binning(pot_mapping, num_bins=13, dense_region_limit=5000, log_bins=7, kmeans_bins=5):
    # 1 对数分箱
    min_value = 200
    max_value = 40000
    log_bins = np.logspace(np.log10(min_value), np.log10(max_value), num=log_bins)
    print("对数分箱区间:", log_bins)
    # 2 细化密集区域的分箱(Quantile Binning)
    dense_region = pot_mapping[(pot_mapping >= min_value) & (pot_mapping <= dense_region_limit)]
    num_bins_dense = num_bins - len(log_bins) + 1  # 根据最终维数调整密集区域的分箱数量
    quantile_bins_dense = pd.qcut(dense_region, q=num_bins_dense, duplicates='drop', retbins=True)[1]
    print("密集区域分位数分箱结果:", quantile_bins_dense)
    # 3 优化稀疏区域的分箱 (K-means Clustering)
    sparse_region = pot_mapping[(pot_mapping > dense_region_limit) & (pot_mapping <= max_value)].reshape(-1, 1)
    kmeans = KMeans(n_clusters=kmeans_bins).fit(sparse_region)
    kmeans_centers = sorted(kmeans.cluster_centers_.flatten())
    print("稀疏区域K-means聚类中心:", kmeans_centers)


def analyze_dense_pot_mapping(dense_pot_mapping, dense_edges_num, min_dense_region_pot, max_dense_region_pot):
    cumulative_weights = np.cumsum(dense_pot_mapping[min_dense_region_pot:max_dense_region_pot])  # 计算累计和
    total_weight = cumulative_weights[-1]
    bin_weight = total_weight / dense_edges_num  # 每个分箱应包含的权重
    bin_edges = [min_dense_region_pot]  # 起始边界
    current_cumulative_weight = 0  # 当前累计权重
    # todo 200对应的值很大（直接成为一个分箱），而201，202这些都没有值，应把201，202这些归类到200中，而不是下一个分箱中
    #  dense_edges_num是边界数，而不是分箱数，边界数=分箱数+1
    for i in range(len(cumulative_weights)):
        current_cumulative_weight += dense_pot_mapping[i + min_dense_region_pot]
        # 检查是否超过了当前分箱的权重
        if current_cumulative_weight >= bin_weight:
            bin_edges.append(i + min_dense_region_pot)  # 记录边界
            current_cumulative_weight = 0
    # 如果最后一个边界不是max_dense_region_pot，手动添加
    if bin_edges[-1] != max_dense_region_pot:
        bin_edges.append(max_dense_region_pot)
    # 确保最终边界数为 dense_bins_num + 1 个（包括起始和结束边界）
    while len(bin_edges) > dense_edges_num + 1:
        # 如果边界数多于dense_bins_num + 1个，合并最小差值的分箱
        diffs = np.diff(bin_edges)
        min_diff_index = np.argmin(diffs)
        bin_edges.pop(min_diff_index + 1)
    while len(bin_edges) < dense_edges_num + 1:
        # 如果边界数少于dense_edges_num + 1个，可能需要在权重分布均匀的地方插入一个边界
        # 这里可以根据实际情况具体调整，例如，在中间插入一个点
        mid_point = (bin_edges[-2] + bin_edges[-1]) // 2
        bin_edges.insert(-1, mid_point)
    bin_edges = sorted(set(bin_edges))  # 确保边界是唯一且有序的
    print("最终的分箱边界1:", bin_edges)
    # 使用np.digitize进行分箱
    binned_indices = np.digitize(np.arange(min_dense_region_pot, max_dense_region_pot), bin_edges) - 1
    # 统计每个分箱中的数据和权重
    bin_counts = np.zeros(len(bin_edges) - 1)
    for i in range(len(bin_counts)):
        bin_counts[i] = np.sum(dense_pot_mapping[200:max_dense_region_pot][binned_indices == i])
    print("每个分箱的权重统计1:", bin_counts)


def load_pot_mapping_array():
    pot_mapping = np.load('pot_mapping_array.npy', 'r')
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


def gmm_threshold(weights):
    x = np.arange(200, 40001).reshape(-1, 1)
    y = weights[200:40001].reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(x, y)
    means = np.sort(gmm.means_.flatten())
    threshold = (means[0] + means[1]) / 2
    return threshold


def bimodal_threshold(weights):
    x = np.arange(200, 40001)
    y = weights[200:40001]
    y_smooth = np.convolve(y, np.ones(50) / 50, mode='same')
    peaks, _ = find_peaks(y_smooth)
    valleys, _ = find_peaks(-y_smooth)
    if len(peaks) >= 2:
        for valley in valleys:
            if peaks[0] < valley < peaks[1]:
                return x[valley]
    return None


if __name__ == '__main__':
    pm = load_pot_mapping_array()
    analyze_dense_pot_mapping(pm[:5000], 8, 200, 5000)
    analyze_pot_mapping(pm)

    threshold_gmm = gmm_threshold(pm)
    print("Gaussian Mixture Model计算的最佳分割点:", threshold_gmm)
    threshold_bimodal = bimodal_threshold(pm)
    print("Bi-modal Analysis计算的最佳分割点:", threshold_bimodal)
