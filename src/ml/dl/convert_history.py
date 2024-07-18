"""
@project: THPMaster
@File   : convert_history.py
@Desc   :
@Author : gql
@Date   : 2024/7/12 16:10
"""
import math
import os
import re
import time
from datetime import datetime

import numpy as np

CARDS = ['2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'Tc', 'Jc', 'Qc', 'Kc', 'Ac',
         '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'Td', 'Jd', 'Qd', 'Kd', 'Ad',
         '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'Th', 'Jh', 'Qh', 'Kh', 'Ah',
         '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'Ts', 'Js', 'Qs', 'Ks', 'As']


class HistoryConvertor:
    def __init__(self, source_dir, save_dir, one_file_size=1000000, bet_ratio_seq=None, pot_ratio_seq=None):
        """

        :param source_dir:
        :param save_path: 保存路径
        :param one_file_size: 一个h5文件最多保存的对局数
        :param bet_ratio_seq:
        :param pot_ratio_seq:
        """
        self.file_path = None
        if bet_ratio_seq is None:
            bet_ratio_seq = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5]
        if pot_ratio_seq is None:
            pot_ratio_seq = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]
        self.source_dir = os.path.join(source_dir)
        # self.save_dirs = save_dirs
        self.save_dir = save_dir
        self.prefix_name = 'thp_dataset_'
        self.num_active_players = 2  # 此处只适用于双人德扑，即 值只能为2
        self.histories = []
        # self.reload_file(path, filename)
        # 确保self.bet_ratio_seq和self.pot_ratio_seq的长度相等
        # 例如倍率只要大于等于5都被赋值为5，所以5可以认为是一种下注的程度，而不是具体的数值（5倍），或许可以不和前面的值连续，用一个更大的值
        self.bet_ratio_seq = bet_ratio_seq  # 相对于最近一次下注的倍率列表
        self.pot_ratio_seq = pot_ratio_seq  # 相对于底池的倍率列表
        self.global_index = 0  # 统计全部对局

    def reload_file(self, path, filename):
        """
        加载一个原始比赛文件(.txt)，通过多次调用此方法以实现多次加载不同的文件
        """
        self.file_path = os.path.join(path, filename)
        with open(self.file_path, 'r') as file:
            self.histories = file.readlines()
        self.histories = self.histories[4:]  # 前4行是一些说明信息，不需要
        drop_list = [0, 1]  # 原数据集中不需要的列
        self.histories = [np.delete(_.split(':'), drop_list).tolist() for _ in self.histories]

    def save_data_to_h5(self):
        all_src_files_count = len(os.listdir(self.source_dir))
        for i, filename in enumerate(os.listdir(self.source_dir)):
            self.reload_file(self.source_dir, filename)
            # self.convert_one_file_to_numpy(save_dir)
            pass
        pass

    def convert_one_file_to_numpy_2(self):
        pass

    def convert_one_actions_history_to_numpy(self, one_actions_history):

        pass
    # def convert_directory_to_numpy(self):
    # all_src_files_count = len(os.listdir(self.source_dir))
    # files_count = all_src_files_count // len(self.save_dirs)
    # cpu_start_time = time.perf_counter()
    # formatted_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # print(f'--------开始时间：{formatted_start_time}--------')
    # print(f'CPU计时：{cpu_start_time}')
    # for i, filename in enumerate(os.listdir(self.source_dir)):
    #     temp_time = time.perf_counter()
    #     save_dir = self.save_dirs[i // files_count]
    #     self.reload_file(self.source_dir, filename)
    #     self.convert_one_file_to_numpy(save_dir)
    #     print(f'当前文件{filename}，保存路径为{save_dir}')
    #     if (i + 1) % 50 == 0:
    #         cpu_end_time = time.perf_counter()
    #         print(
    #             f'----当前已处理{i + 1}/{all_src_files_count}个文件，最后一个文件名为{filename}，'
    #             f'此批次CPU用时{cpu_end_time - temp_time}----')
    # cpu_end_time = time.perf_counter()
    # print(f'总CPU用时：{cpu_end_time - cpu_start_time}')
    # formatted_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # print(f'--------结束时间：{formatted_end_time}--------')

    def convert_one_file_to_numpy(self, save_dir):
        """
        将一个文件的所有对局全都转为numpy张量形式
        :return:
        """
        suffix = 0
        if self.histories is None or self.histories == []:
            return
        for i, history in enumerate(self.histories):
            earn_chips = history[2].split('|')
            actions = self.convert_one_actions_history(history[0])
            # cards_1和cards_2分别对应双方手牌
            cards_1 = self.convert_one_cards_history(history[1], 0)
            cards_2 = self.convert_one_cards_history(history[1], 1)
            # 以时间戳方式命名文件
            t = time.time()
            npz_name_1 = str(t).replace('.', '') + str(suffix) + '_' + earn_chips[0]
            npz_name_2 = str(t).replace('.', '') + str(suffix + 1) + '_' + earn_chips[1]
            np.savez_compressed(os.path.join(save_dir, npz_name_1), a=actions, b=cards_1)
            np.savez_compressed(os.path.join(save_dir, npz_name_2), a=actions, b=cards_2)
            suffix = (suffix + 2) % 10
            self.global_index += 1

    def convert_one_actions_history(self, one_actions_history):
        """
        提取一局的动作序列，转为numpy张量形式
        :return: [4, 27（每个阶段的动作数 * 表示每个动作所需的向量数）, 13（抽象后的动作种类个数）]
        """
        '''
        # 按照上一位玩家下注筹码量抽象动作，例如1bet（表示比上一位玩家多下注1倍）
        # 每阶段的第一个动作根据底池下注，因为此时没有上一位玩家下注进行参考，而且应更加看重每阶段的第一个动作（比如多加一些特征）
        # 动作列表如下：
        # （已废弃）[fold, check, call, 1bet, 1.25bet, 1.5bet, 1.75bet, 2bet, 2.5bet, 3bet, 4bet, allin]
        # （新版本）[1bet, 1.25bet, 1.5bet, 1.75bet, 2bet, 2.5bet, 3bet, 4bet, 5bet, allin, call, check, fold]
        #         [0.5pot, 0.75pot, 1pot, 1.25pot, 1.5pot, 2pot, 2.5pot, 3pot, 4pot, allin, call, check, fold]
        # 除allin外的最后几位说明该玩家的下注很大，即该玩家...
        # 用一行表示合法的动作
        # 具体实现1：一个动作用2行表示
        # [玩家1的位置，玩家1的动作]
        # [玩家1的位置，玩家1的合法动作]
        # [玩家2的位置，玩家2的动作]
        # [玩家2的位置，玩家2的合法动作]
        # ...以此类推，直到本阶段结束...
        # 具体实现2：一个动作用3行表示
        # [玩家1的动作]
        # [玩家1的合法动作]
        # [玩家1的位置]  位置用全为1或全为2，单独用一行表示位置，想通过位置表明这个矩阵包含了两个不同玩家的下注动作（强化这一认知）
        # ...玩家2也是以此类推...
        # 一个阶段内双方总计最多做9个动作，一共4个阶段，总计4*9=36个动作
        # 一个动作用3行表示，一个阶段需要3*9=27行
        '''
        '''
        改进点（时间点240618）：
        1. 添加了legal actions（已废弃）
        2. 小盲注玩家用1表示，大盲注玩家用2表示，后续未进行下注的位置都用0表示
        3. 如何表示每个阶段的第一次下注？此时下注倍率是相对于底池的倍率
        '''
        actions = np.zeros((4, 27, 4 + len(self.bet_ratio_seq)), dtype=int)  # 最后返回的结果
        one_actions_history_splits = one_actions_history.split('/')
        players_chip = [50, 100, 0]  # 前两个元素表示大小盲注在当前阶段内的下注筹码量，最后一个元素表示除当前阶段下注筹码之外的底池大小
        # legal_actions_idxes = list(range(0, len(self.pot_ratio_seq)))
        # 遍历每个阶段
        for i, one_split in enumerate(one_actions_history_splits):
            one_split = re.findall(r'\d+|f|c', one_split)  # 提取出数字或f或c
            if i == 0:  # flop阶段，小盲注先行动
                j = 0  # j用来表示当前动作的玩家索引（大小盲注索引），用来确定位置那一行的值是0还是1
                last_chip = 100
            else:  # 其他阶段，大盲注先行动
                j = 1
                last_chip = 0
            action_idx = 0
            k = 0  # k用来表示是否为每个阶段内的第一个raise动作
            m = 0  # 区分每个阶段内的第二个c
            # 遍历一个阶段内的所有动作
            for raw_action in one_split:
                if raw_action.isdigit():  # 表示此raw_action是raise动作
                    raise_chip = int(raw_action)
                    if raise_chip >= 20000:  # 原数据集里的allin用r20000表示，对方此时只能call或fold
                        l = -4  # l对应当前动作在动作列表中的索引
                        k = 1
                    elif k == 0 and i != 0:  # 非flop阶段内第一次下注。flop阶段没有第一次下注（因为已经下了大小盲注）
                        l = self.specific_to_abstract_chip(raise_chip, players_chip[-1], k)
                        k = 1
                    else:
                        k = 1  # flop阶段没有第一次下注，k需要先置1
                        l = self.specific_to_abstract_chip(raise_chip, last_chip, k)
                    last_chip = raise_chip
                    players_chip[j] = raise_chip
                    actions[i][action_idx][l] = 1
                elif raw_action == 'f':  # fold
                    actions[i][action_idx][-1] = 1
                elif raw_action == 'c':  # check或call
                    if players_chip[0] == players_chip[1] and m == 0:  # check
                        actions[i][action_idx][-2] = 1
                    else:  # call
                        actions[i][action_idx][-3] = 1
                    players_chip[j] = players_chip[(j + 1) % self.num_active_players]
                    m = 1
                else:
                    raise ValueError(f'未知的动作:{raw_action}')
                actions[i][action_idx + 2] = j + 1  # 更新位置信息，位置信息用[1, 2]表示大小盲注，而j的取值范围为[0,1]
                j = (j + 1) % self.num_active_players
                action_idx += 3
            players_chip[-1] += players_chip[0] + players_chip[1]
            players_chip[0] = 0  # 非flop阶段，两位玩家的初始筹码量都置为0
            players_chip[1] = 0
        return actions

    @staticmethod
    def convert_one_cards_history(one_cards_history: str, pos: int):
        """
        将pos位置的手牌和公共牌转为numpy形式
        :param one_cards_history: 手牌和公共牌
        :param pos: 准备提取哪个位置玩家的手牌，取值为0或1
        :return: numpy格式[5, 4, 13]
        """
        '''
        示例
        输入：one_cards_history='Td4h|6cAd/5c8hKh/Qh/5h',pos=0
        输出：格式为[5,4,13]，其中第一个[4,13]的矩阵是手牌（Td4h对应的矩阵形式），
        第二个[4,13]是flop阶段公共牌，以此类推，最后一个[4,13]是当前所有已发出的公共牌
        '''
        cards = np.zeros((5, 4, 13), dtype=int)
        one_cards_history_splits = one_cards_history.split('/')
        # 提取手牌
        hands_history = one_cards_history_splits[0]  # 两位玩家的手牌都在这里，根据pos的值确定提取哪个玩家的手牌
        pos = pos * 5
        hand1 = CARDS.index(hands_history[pos:pos + 2])
        suit, num = hand1 // 13, hand1 % 13
        cards[0][suit][num] = 1
        hand2 = CARDS.index(hands_history[pos + 2:pos + 4])
        suit, num = hand2 // 13, hand2 % 13
        cards[0][suit][num] = 1
        # 提取公共牌
        public_cards = ''
        for _ in one_cards_history_splits[1:]:
            public_cards += _
        # 判断是否有公共牌
        if public_cards:
            # 每两个字符表示一张牌，进行遍历
            for i in range(0, len(public_cards), 2):
                suit, num = CARDS.index(public_cards[i:i + 2]) // 13, CARDS.index(public_cards[i:i + 2]) % 13
                # i的可能取值为0，2，4，6，8
                if i <= 4:  # flop公共牌
                    cards[1][suit][num] = 1
                elif i == 6:  # turn公共牌
                    cards[2][suit][num] = 1
                elif i == 8:  # river公共牌
                    cards[3][suit][num] = 1
                cards[4][suit][num] = 1
        return cards

    def specific_to_abstract_chip(self, specific_chip, last_chip, ratio_flag=1):
        """
        将当前筹码量转为相对于上一个下注筹码量的倍率，即1bet, 1.25bet, 2bet等
        :param specific_chip: 当前下注筹码量（具体数值）
        :param last_chip: 最新一次下注筹码量，若是阶段内的第一次下注，则为底池大小
        :param ratio_flag: 值为1表示相对于最新一次下注（self.bet_ratio_seq）的倍率，
        值为0表示相对于底池（self.pot_ratio_seq）的倍率
        """
        if ratio_flag:
            ratio_seq = self.bet_ratio_seq
        else:
            ratio_seq = self.pot_ratio_seq
        if last_chip == 0:
            raise ValueError("分母last_chip不能为0")
        ratio = specific_chip / last_chip
        min_diff = float('inf')
        closest_index = -1
        for index, value in enumerate(ratio_seq):
            diff = abs(ratio - value)
            if diff < min_diff:
                min_diff = diff
                closest_index = index
        return closest_index

    @staticmethod
    def encode_pot_size_temperature(pot_size, min_pot_size=100, max_pot_size=40000, num_levels=13, temperature=0.5):
        """
        使用温度编码将底池筹码量编码成13维的向量。
        参数:
        pot_size: 当前的底池筹码量。
        min_pot_size: 底池的最小值。
        max_pot_size: 总底池的最大值。
        num_levels: 编码的维度，默认为13。
        temperature: 控制编码平滑度的温度参数。

        返回:
        一个13维的二进制向量，表示温度编码的底池筹码量。
        """
        # 创建一个13维的二进制向量
        pot_size_vector = [0] * num_levels

        # 计算每个级别的中心值和宽度
        step = (max_pot_size - min_pot_size) / num_levels
        centers = [min_pot_size + step * (i + 0.5) for i in range(num_levels)]

        # 根据温度参数进行编码
        for i, center in enumerate(centers):
            distance = abs(pot_size - center) / step  # 归一化的距离
            # 使用温度参数计算平滑的概率值
            value = 1 / (1 + math.exp(distance / temperature))
            # 二进制化，以0.5为阈值
            pot_size_vector[i] = 1 if value > 0.5 else 0

        return pot_size_vector


def convert_history():
    path1 = 'D:\\Desktop\\acpc_dataset'
    save_path1 = 'D:\\Desktop\\acpc_data_2'
    hc = HistoryConvertor(path1, save_path1)


if __name__ == '__main__':
    pass
