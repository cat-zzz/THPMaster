"""
@project: THPMaster
@File   : strategy_2.py
@Desc   :
@Author : gql
@Date   : 2024/7/24 20:05
"""
import math
import random

import numpy as np
import pandas as pd

from competition.core import constants
from competition.core.entity import Game
from competition.util.poker_util import deal_cards
from competition.util.poker_value import best_cards


class Strategy:
    def __init__(self):
        # oppo_hands阶段亮手牌，可以看出对手的手牌松紧度，但是没必要
        self.possible_oppo_hands = get_hands_with_win_rate_section()  # 基础胜率区间的手牌
        self.catzzz = None
        self.opponent = None
        self.game = None
        self.current_stage = constants.init_stage  # 每次比self.game.stage慢一步
        self.oppo_min_win_rate = 0.4  # 对手最低胜率手牌范围
        self.oppo_max_win_rate = 1.0  # 对手最高胜率手牌范围
        self.my_hands_win_rate = 0.0  # 在对方手牌入场范围的情况下，我方手牌胜率
        self.use_bluff = True  # 本局及之后对局是否会使用诈唬
        self.bluff_rate = 0.0  # 我方使用诈唬的概率
        self.certainly_win_prob = 0.0  # 必胜概率，表示离必胜还有多少筹码的差距。差距越小，下注应越谨慎
        # flop阶段及其之后阶段的胜率基本一致

    def start_game(self, game: Game):
        """
        每开始一局都要调用此函数
        """
        self.game = game
        self.catzzz = game.catzzz
        self.opponent = game.opponent
        self.current_stage = constants.init_stage

    def strategy(self, episode):
        """
        核心策略函数
        :param episode: 当前对局数
        :return: str->做出的决策，符合服务端指令格式要求
        """
        # todo 当我方是坚果牌时的策略
        min_win_chip = (70 - episode) / 2 * 150 + 100  # 后续对局直接弃牌最终也能获胜的最低筹码
        # 已赢得足够筹码，后续对局均弃牌
        if self.catzzz.total_earn_chip >= min_win_chip:
            print("筹码已足够，采取必胜策略: fold, 已赢得筹码:", self.catzzz.total_earn_chip,
                  "预计损失筹码：", (70 - episode) / 2 * 150 + 100)
            return "fold"
        self.my_hands_win_rate, win_rate, tie_rate, lose_rate = mc_my_hands_win_rate(self.catzzz.hand_cards,
                                                                                     self.game.public_cards,
                                                                                     self.possible_oppo_hands)
        # todo 我方是坚果牌时的策略
        if lose_rate == 0:
            # preflop阶段没有坚果牌
            if self.current_stage == constants.turn_stage:

            pass
        # 一般情况
        if self.game.stage == constants.preflop_stage:
            if self.current_stage == constants.init_stage:
                self.current_stage = constants.preflop_stage
                print("进入preflop阶段，my_hands_win_rate:", self.my_hands_win_rate)
        # preflop阶段下注筹码不超过1000，即手牌为AA时只下注1000

    def preflop_strategy(self):
        # todo 统计ACPC的比赛数据，统计全部智能体拥有某手牌时的下注筹码分布，preflop阶段就采用查表的方式下注
        # todo 如何让turn阶段、river阶段也能使用上ACPC数据，或者使用博弈树搜索
        oppo_stage_chip = self.opponent.stage_chip
        catzzz_stage_chip = self.catzzz.stage_chip
        if catzzz_stage_chip == oppo_stage_chip == 100:  # 我方作为大盲注第一次下注且可以check，此时check我们并不需要付出筹码
            return constants.player_check_action, 0
        raise_range = [0, 0.395, 0.505, 0.551, 0.6, 0.66, 0.7, 1]
        if raise_range[0] <= self.my_hands_win_rate < raise_range[1]:
            return constants.player_fold_action, -1
        elif raise_range[1] <= self.my_hands_win_rate < raise_range[2]:  # 大约是0.395-0.505
            if np.random.random() < 0.35:  # 0.5的概率会下注
                if oppo_stage_chip < 400:
                    catzzz_action = constants.player_raise_action
                    raise_chip = 2 * oppo_stage_chip + 3 + random.randint(1, 35)
                elif oppo_stage_chip > 800:
                    catzzz_action = constants.player_fold_action
                    raise_chip = -1
                else:
                    catzzz_action = constants.player_call_action
                    raise_chip = math.fabs(oppo_stage_chip - catzzz_stage_chip)
            else:
                catzzz_action = constants.player_fold_action
                raise_chip = -1
        elif raise_range[2] <= self.my_hands_win_rate < raise_range[3]:  # 0.505, 0.551
            if  np.random.random() < 0.55:
                if oppo_stage_chip < 600:
                    catzzz_action = constants.player_raise_action
                    raise_chip = 2.2 * oppo_stage_chip + random.randint(50, 100)
                elif oppo_stage_chip > 1100:
                    catzzz_action = constants.player_fold_action
                    raise_chip = -1
                else:
                    catzzz_action = constants.player_call_action
                    raise_chip = math.fabs(oppo_stage_chip - catzzz_stage_chip)
            else:
                catzzz_action = constants.player_fold_action
                raise_chip = -1
        elif raise_range[3] <= self.my_hands_win_rate < raise_range[4]:
            pass
        elif raise_range[4] <= self.my_hands_win_rate < raise_range[5]:
            pass
        elif raise_range[5] <= self.my_hands_win_rate < raise_range[6]:
            pass
        elif raise_range[6] <= self.my_hands_win_rate < raise_range[7]:
            pass
        else:
            return constants.player_fold_action, -1
        if self.my_hands_win_rate < 0.395:
            return constants.player_fold_action, -1
        return catzzz_action, raise_chip


def get_hands_with_win_rate_section(min_win_tate=0.4, max_win_rate=1.0, exclude=None):
    """
    选出基础胜率在[min_win_rate, max_win_rate]区间下的所有手牌(区分手牌花色)
    """
    # 读取文件需要用绝对路径
    # hands_prob_sum = pd.read_csv('../../tool/hands_prob_sum.csv', header=0,
    #                              usecols=range(1, 5)).values
    # 从main.py启动时，运行下面的代码
    # hands_prob_sum = pd.read_csv('tool/hands_prob_sum.csv', header=0, usecols=range(1, 5)).values
    hands_prob_sum = pd.read_csv('D:\\Development\\pythonProjects\\THPMaster\\competition\\tool\\hands_prob_sum.csv',
                                 header=0, usecols=range(1, 5)).values
    # high_win_hand_cards = hands_prob_sum[np.where(min_win_tate <= hands_prob_sum[:, -1] <= max_win_rate)][:, :-2]
    high_win_hand_cards = hands_prob_sum[np.where(min_win_tate <= hands_prob_sum[:, -1])]
    high_win_hand_cards = hands_prob_sum[np.where(high_win_hand_cards[:, -1] <= max_win_rate)][:, :-2]
    high_win_hand_cards = high_win_hand_cards.astype(np.int32)
    # 将手牌转成不同的花色的手牌，hands_prob_sum.csv文件只考虑同花和非同花，没有考虑具体的花色
    suit_hand_cards = np.zeros((0, 2, 2), dtype=int)
    for first, second in high_win_hand_cards:
        if first <= second:  # 对子(first==second)和非同花(first<second)
            for i in range(4):  # 确定第一张牌的花色
                for j in range(i + 1, 4):  # 确定第二张牌的花色
                    if i != j:
                        temp = np.array([[[i, first], [j, second]]], dtype=int)
                        if exclude is None or \
                                (exclude is not None and temp[0, 0] not in exclude and temp[0, 1] not in exclude):
                            suit_hand_cards = np.append(suit_hand_cards, temp, axis=0)
        else:  # 同花
            for i in range(4):  # 两张牌花色相同
                temp = np.array([[[i, first], [i, second]]], dtype=int)
                suit_hand_cards = np.append(suit_hand_cards, temp, axis=0)
    return suit_hand_cards


def mc_my_hands_win_rate(my_hands: np.ndarray, current_public_cards: np.ndarray,
                         oppo_hands=get_hands_with_win_rate_section(), total_count=3000):
    """
    蒙特卡洛模拟我方手牌胜率，其中对手手牌只取某一初始胜率区间的手牌
    :return: 四元组，第一个表示我方手牌胜率得分，第二个表示胜率，第三个表示平局率，第四个表示败率
    """
    per_lose_score = 0
    per_draw_score = 2
    per_win_score = 4

    win_count = 0
    tie_count = 0
    lose_count = 0
    score = 0
    current_public_cards_count = current_public_cards.shape[0]
    for _ in range(total_count):
        # 随机发出剩下的公共牌
        remain_public = deal_cards(5 - current_public_cards_count, np.vstack((current_public_cards, my_hands)))
        public_cards = np.vstack((current_public_cards, remain_public))
        oppo_hand = oppo_hands[np.random.choice(oppo_hands.shape[0])]  # 从可能的对手手牌中随机选择一副
        # 比较大小
        my_poker_value = best_cards(my_hands, public_cards)
        oppo_poker_value = best_cards(oppo_hand, public_cards)
        if my_poker_value > oppo_poker_value:
            score += per_win_score  # 我方牌型比对方大，加4分
            win_count += 1
        elif my_poker_value == oppo_poker_value:
            tie_count += 1
            score += per_draw_score  # 平局
        else:
            score += per_lose_score  # 我方牌型比对方小，加0分
            lose_count += 1
    print("func mc_my_hands_win_rate()->win:", win_count, "tie:", tie_count, "lose:", lose_count, "total:", total_count)
    return score / (
            total_count * per_win_score), win_count / total_count, tie_count / total_count, lose_count / total_count


if __name__ == '__main__':
    my = np.array([[1, 5], [1, 12]])
    # public = np.array([[1, 9], [2, 3], [1, 4], [2, 4], [1, 12]])
    public = np.array([[1, 0], [2, 8], [2, 11], [3, 7]])

    my = np.array([[0, 12], [0, 11]])
    public = np.array([[0, 10], [0, 9], [0, 8], [1, 3], [2, 0]])
    mc_my_hands_win_rate(my, public)
