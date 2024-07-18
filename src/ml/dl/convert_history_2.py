"""
@project: THPMaster
@File   : convert_history_2.py
@Desc   :
@Author : gql
@Date   : 2024/7/12 21:02
"""
import os

import numpy as np

CARDS = ['2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'Tc', 'Jc', 'Qc', 'Kc', 'Ac',
         '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'Td', 'Jd', 'Qd', 'Kd', 'Ad',
         '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'Th', 'Jh', 'Qh', 'Kh', 'Ah',
         '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'Ts', 'Js', 'Qs', 'Ks', 'As']


class HistoryConvertor:
    def __init__(self, source_dir, save_dir, file_count: int, bet_ratio_seq=None, pot_ratio_seq=None):
        self.file_path = None
        if bet_ratio_seq is None:
            bet_ratio_seq = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5]
        if pot_ratio_seq is None:
            pot_ratio_seq = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
        self.source_dir = os.path.join(source_dir)
        self.save_dir = os.path.join(save_dir)
        self.file_count = file_count
        self.prefix_name = 'thp_dataset_h5_'
        self.num_active_players = 2  # 此处只适用于双人德扑，即 值只能为2
        self.histories = []  # 保存一个文件中的所有对局
        self.bet_ratio_seq = bet_ratio_seq  # 相对于最近一次下注的倍率列表
        self.pot_ratio_seq = pot_ratio_seq  # 相对于底池的倍率列表
        self.global_index = 0  # 统计全部对局

    def run(self):
        all_src_files_count = len(os.listdir(self.source_dir))
        for i, filename in enumerate(os.listdir(self.source_dir)):
            self.load_file(self.source_dir, filename)   # 加载一个文件到self.histories

    def load_file(self, path, filename):
        """
        加载一个原始比赛文件(.txt)，通过多次调用此方法以实现多次加载不同的文件
        """
        self.file_path = os.path.join(path, filename)
        with open(self.file_path, 'r') as file:
            self.histories = file.readlines()
        self.histories = self.histories[4:]  # 前4行是一些说明信息，不需要
        drop_list = [0, 1]  # 原数据集中不需要的列
        self.histories = [np.delete(_.split(':'), drop_list).tolist() for _ in self.histories]


if __name__ == '__main__':
    pass
