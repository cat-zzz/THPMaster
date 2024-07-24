"""
@File   : strategy_by_win_rate.py
@Desc   : 基于实时胜率的策略
@Author : gql
@Date   : 2023/7/19 10:07
"""
import math
import random

import numpy as np
import pandas as pd

import competition.util.common_util as util
from competition.core import constants
from competition.core.entity import Game
from competition.tool.win_rate_evaluate import evaluate_hands_win_rate, evaluate_hands_win_rate_1
from competition.util.poker_util import deal_cards, per_win_score, per_draw_score, per_lose_score
from competition.util.poker_value import best_cards


class Strategy:
    def __init__(self):
        # oppo_hands阶段亮手牌，可以看出对手的手牌松紧度
        self.possible_oppo_hands = get_hands_with_win_rate_section()
        self.catzzz = None
        self.opponent = None
        self.game = None
        self.current_stage = constants.init_stage
        # self.oppo_hands_tight = 0.5  # 对手手牌松紧度(对方手牌入场范围)
        self.oppo_min_win_rate = 0.5
        self.oppo_max_win_rate = 1.0
        self.my_hands_win_rate = 0.0  # 在对方手牌入场范围的情况下，我方手牌胜率

    def start_game(self, game: Game):
        """
        每开始一局都要调用此函数
        """
        self.game = game
        self.catzzz = game.catzzz
        self.opponent = game.opponent
        self.current_stage = constants.init_stage

    def strategy(self):
        # todo 对手allin行为的策略
        # todo 当我方是坚果牌的策略
        # todo 考虑对手什么时候最容易诈唬
        if self.opponent.operation[-1, 0] == constants.player_allin_action:
            # 对手做出allin行为后，我方只能选择call或fold
            # todo 具体策略
            return "call"
        if self.game.stage == constants.preflop_stage:
            if self.current_stage == constants.init_stage:  # 每局第一次进入到preflop阶段
                self.current_stage = constants.preflop_stage
                self.possible_oppo_hands = get_hands_with_win_rate_section(
                    self.oppo_min_win_rate, self.oppo_max_win_rate, self.catzzz.hand_cards)
                # 胜率只需在发公共牌的时候计算一次即可，即每个阶段只需计算一次胜率
                self.my_hands_win_rate = my_hands_win_rate(
                    self.catzzz.hand_cards, np.zeros((0, 2)), self.possible_oppo_hands)
                print("进入preflop阶段，my_hands_win_rate:", self.my_hands_win_rate)
            action, raise_chip = self.preflop_strategy()
        elif self.game.stage == constants.flop_stage:
            if self.current_stage == constants.preflop_stage:  # 每局第一次进入到flop阶段
                self.current_stage = constants.flop_stage
                self.my_hands_win_rate = my_hands_win_rate(
                    self.catzzz.hand_cards, self.game.public_cards, self.possible_oppo_hands)
                print("进入flop阶段，my_hands_win_rate:", self.my_hands_win_rate)
            action, raise_chip = self.flop_strategy()
        elif self.game.stage == constants.turn_stage:
            if self.current_stage == constants.flop_stage:
                self.current_stage = constants.turn_stage
                self.my_hands_win_rate = my_hands_win_rate(
                    self.catzzz.hand_cards, self.game.public_cards, self.possible_oppo_hands)
                print("进入turn阶段，my_hands_win_rate:", self.my_hands_win_rate)
            action, raise_chip = self.turn_strategy()
        elif self.game.stage == constants.river_stage:
            if self.current_stage == constants.turn_stage:
                self.current_stage = constants.river_stage
                self.my_hands_win_rate = my_hands_win_rate(
                    self.catzzz.hand_cards, self.game.public_cards, self.possible_oppo_hands)
                print("进入river阶段，my_hands_win_rate:", self.my_hands_win_rate)
            action, raise_chip = self.river_strategy()
        else:
            util.print_exception(Strategy, "unknown stage")
            return "fold"
        # show_oppo_hands阶段不用返回动作
        if self.game.stage == constants.show_oppo_card:
            oppo_hands = self.opponent.hand_cards
            print("进入oppo_hands阶段，oppo_hands:", oppo_hands)
            self.show_oppo_cards_strategy(oppo_hands)
        return convert_client_cmd(action, raise_chip)

    def preflop_strategy(self):
        oppo_stage_chip = self.opponent.stage_chip
        catzzz_stage_chip = self.catzzz.stage_chip
        print("preflop_strategy->my_hands_win_rate:", self.my_hands_win_rate)
        if self.my_hands_win_rate < 0.4:
            return constants.player_fold_action, -1
        elif 0.4 <= self.my_hands_win_rate < 0.52:
            if oppo_stage_chip < 1000:
                catzzz_action = constants.player_raise_action
                raise_chip = 2 * oppo_stage_chip + random.randint(30, 50)
            elif oppo_stage_chip > 2000:  # 通常是对方第一次下注就超过2000
                catzzz_action = constants.player_fold_action
                raise_chip = -1
            else:
                catzzz_action = constants.player_call_action
                raise_chip = math.fabs(oppo_stage_chip - catzzz_stage_chip)
        elif 0.52 <= self.my_hands_win_rate < 0.6:
            if oppo_stage_chip < 1500:
                catzzz_action = constants.player_raise_action
                raise_chip = 2 * oppo_stage_chip + random.randint(30, 80)
            elif oppo_stage_chip > 3000:
                catzzz_action = constants.player_fold_action
                raise_chip = -1
            else:
                catzzz_action = constants.player_call_action
                raise_chip = math.fabs(oppo_stage_chip - catzzz_stage_chip)
        elif 0.6 <= self.my_hands_win_rate < 0.65:
            if oppo_stage_chip < 2000:
                catzzz_action = constants.player_raise_action
                raise_chip = 2.2 * oppo_stage_chip + random.randint(50, 100)
            elif oppo_stage_chip > 4000:
                catzzz_action = constants.player_fold_action
                raise_chip = -1
            else:
                catzzz_action = constants.player_call_action
                raise_chip = math.fabs(oppo_stage_chip - catzzz_stage_chip)
        elif 0.65 < self.my_hands_win_rate:
            if oppo_stage_chip < 3000:
                catzzz_action = constants.player_raise_action
                raise_chip = 2.5 * oppo_stage_chip + random.randint(50, 100)
            else:
                catzzz_action = constants.player_call_action
                raise_chip = math.fabs(oppo_stage_chip - catzzz_stage_chip)
        else:
            return constants.player_fold_action, -1
        return catzzz_action, raise_chip

    def flop_strategy(self):
        # 如果自己是第一个行动的，先check
        # 有些特殊（必赢）的手牌可以直接check
        # 按照概率选择check还是直接下注
        # 跟注的底池赔率
        call_odds = float((self.opponent.stage_chip - self.catzzz.stage_chip)) / self.game.total_chip
        # ev = self.game.total_chip * self.my_hands_win_rate - x * (1 - self.my_hands_win_rate)
        # 基于期望的最大下注
        ev_chip = self.game.total_chip * self.my_hands_win_rate / (1 - self.my_hands_win_rate)

        print("call_odds:", call_odds, "my_hands_win_rate:", self.my_hands_win_rate, "ev_chip:", ev_chip)
        print("oppo_stage:", self.opponent.stage_chip, "my_stage:", self.catzzz.stage_chip, "game_chip:",
              self.game.total_chip)
        print("my_hands_win_rate:", self.my_hands_win_rate, "call_odds:", call_odds)
        print("ev_chip(old):", ev_chip)
        # 默认对方只选择胜率大于0.5的手牌进行游戏
        catzzz_action = constants.player_fold_action
        catzzz_chip = -1
        # ev_chip_1 = int(self.game.total_chip * self.my_hands_win_rate / (1 - self.my_hands_win_rate))
        ev_chip = int(
            (self.oppo_min_win_rate + self.game.total_chip * self.my_hands_win_rate * (1 - self.oppo_min_win_rate)) /
            ((1 - self.oppo_min_win_rate) * (2 * self.my_hands_win_rate - 1)))
        print("新公式下的ev chip:", ev_chip)
        if self.opponent.stage_chip > 0:  # 对方本阶段已经下注
            # 跟注的底池赔率
            call_odds = float((self.opponent.stage_chip - self.catzzz.stage_chip)) / self.game.total_chip
            # 我方期望最大下注大于对方当前下注的2倍，则我方采取raise行为，而不是call行为
            if self.my_hands_win_rate >= call_odds and ev_chip >= self.opponent.stage_chip * 2:
                catzzz_action = constants.player_raise_action
                catzzz_chip = ev_chip
            elif self.my_hands_win_rate >= call_odds and \
                    self.opponent.stage_chip * 0.5 < ev_chip < self.opponent.stage_chip * 2:
                catzzz_action = constants.player_call_action
                catzzz_chip = self.opponent.stage_chip
            else:
                catzzz_action = constants.player_fold_action
                catzzz_chip = -1
        else:  # 本阶段第一次下注由我方先开始
            # 基于最大期望下注
            if ev_chip < 100:
                catzzz_action = constants.player_fold_action
                catzzz_chip = -1
            else:
                catzzz_action = constants.player_raise_action
                catzzz_chip = ev_chip
        print("策略函数返回值: ", catzzz_action, catzzz_chip)
        chip = catzzz_chip - self.catzzz.stage_chip + self.catzzz.game_chip
        if chip >= constants.total_chip:  # 本局总下注量大于20000，用allin指令
            return constants.player_allin_action, constants.total_chip
        return catzzz_action, catzzz_chip

    def turn_strategy(self):
        # pass
        return self.flop_strategy()

    def river_strategy(self):
        # 如果自己能组成坚果牌，可以逐次强势下注（多次下注，下注量很大）
        return self.flop_strategy()

    def show_oppo_cards_strategy(self, oppo_hands):
        # 随着对局的进行，可以动态调整对手手牌范围
        basic_win_rate = my_hands_win_rate(oppo_hands, np.zeros((0, 2), dtype=int))
        win_rate = my_hands_win_rate(oppo_hands, self.game.public_cards)
        if basic_win_rate < 0.4:
            if win_rate < 0.5:
                self.oppo_min_win_rate -= 0.025
            else:
                self.oppo_min_win_rate -= 0.01
        else:
            self.oppo_min_win_rate += 0.002
        if self.oppo_min_win_rate <= 0.3:
            self.oppo_min_win_rate = 0.3  # 最低不低于0.3
        elif self.oppo_min_win_rate >= 0.6:
            self.oppo_min_win_rate = 0.6  # 最高不高于0.6
        print("show oppo hands->oppo_min_win_rate:", self.oppo_min_win_rate)


def convert_client_cmd(action, raise_chip) -> str:
    """
    将[action, raise_chip]格式的指令转为字符串格式

    :param action:
    :param raise_chip:
    :return:
    """
    if action == constants.player_call_action:
        return 'call'
    elif action == constants.player_check_action:
        return 'check'
    elif action == constants.player_raise_action:
        return 'raise ' + str(int(raise_chip))
    elif action == constants.player_allin_action:
        return 'allin'
    elif action == constants.player_fold_action:
        return 'fold'
    else:
        util.print_exception(convert_client_cmd, '未知的客户端指令')
        return 'fold'


def get_hands_with_win_rate_section(min_win_tate=0.5, max_win_rate=1.0, exclude=None):
    """
    选出基础胜率在[min_win_rate, max_win_rate]区间下的所有手牌(区分手牌花色)
    """
    # 读取文件需要用绝对路径
    hands_prob_sum = pd.read_csv('../tool/hands_prob_sum.csv', header=0,
                                 usecols=range(1, 5)).values
    # 从main.py启动时，运行下面的代码
    # hands_prob_sum = pd.read_csv('tool/hands_prob_sum.csv', header=0,
    #                              usecols=range(1, 5)).values
    # high_win_hand_cards = hands_prob_sum[np.where(min_win_tate <= hands_prob_sum[:, -1] <= max_win_rate)][:, :-2]
    high_win_hand_cards = hands_prob_sum[np.where(min_win_tate <= hands_prob_sum[:, -1])]
    high_win_hand_cards = hands_prob_sum[np.where(high_win_hand_cards[:, -1] <= max_win_rate)][:, :-2]
    high_win_hand_cards = high_win_hand_cards.astype(np.int32)
    # 将手牌转成不同的花色的手牌，因为hands_prob_sum.csv文件只考虑同花和非同花，没有考虑具体的花色
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


def my_hands_win_rate(my_hands: np.ndarray, current_public_cards: np.ndarray,
                      oppo_hands=get_hands_with_win_rate_section(), total_count=1314):
    """
    随机模拟若干次后，我方手牌的胜率

    :param my_hands: 我方手牌（二维数组）
    :param current_public_cards: 当前公共牌（二维数组）
    :param oppo_hands: 对方可能的手牌集合（三维数组）
    :param total_count: 总模拟次数
    :return: 我方手牌胜率
    """
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
    print("win:", win_count, "tie:", tie_count, "lose:", lose_count, "total:", total_count)
    return score / (total_count * per_win_score)


if __name__ == '__main__':
    my = np.array([[1, 12], [1, 11]])
    # public = np.array([[1, 9], [2, 3], [1, 4], [2, 4], [1, 12]])
    # public = np.array([[1, 6], [2, 3], [0, 2]])
    # my = np.array([[1, 2], [3, 3]])
    public = np.array([[3, 12], [0, 5], [2, 3]])
    print(np.vstack((public, my)).shape)
    oppo = get_hands_with_win_rate_section(0.5, 1, np.vstack((public, my)))
    # oppo = np.array([[2, 12], [1, 10]], dtype=int)
    # oppo = np.array([[[3, 10], [3, 8]]])
    print("综合胜率:", my_hands_win_rate(my, public, oppo))
    print("未考虑任何公共牌和对方手牌的基础胜率:", evaluate_hands_win_rate(my, 1314))
    print("考虑公共牌的基础胜率:", evaluate_hands_win_rate_1(my, public, 1314))
    pass
