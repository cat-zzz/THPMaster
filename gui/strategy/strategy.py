"""
@project: THPMaster
@File   : strategy.py
@Desc   :
@Author : gql
@Date   : 2025/2/18 14:26
"""
import math
import random

import numpy as np
import pandas as pd

from gui.env import constants
from gui.env.rl_game import NoLimitHoldemGame
from gui.util.card_util import deal_cards
from gui.util.poker_value import best_cards


class Strategy:
    def __init__(self, game: NoLimitHoldemGame, player_idx, total_episode=70):
        self.total_episode = total_episode
        self.min_win_chip = total_episode / 2 * 150 + 1
        self.episode = 0
        self.possible_oppo_hands = get_hands_with_win_rate_section()  # 基础胜率区间的手牌
        # 只允许读取game中的属性，不允许修改game中的属性
        self.my_hands = game.get_hand_cards(player_idx)
        self.game = game
        self.catzzz = game.players[player_idx]
        self.oppo = game.players[(player_idx + 1) % len(game.players)]
        # self.opponent = None
        # self.current_stage = constants.init_stage  # 每次比self.game.stage慢一步
        self.oppo_min_win_rate = 0.4  # 对手最低胜率手牌范围
        self.oppo_max_win_rate = 1.0  # 对手最高胜率手牌范围
        self.my_hands_win_rate = 0.0  # 在对方手牌入场范围的情况下，我方手牌胜率
        self.use_bluff = True  # 本局及之后对局是否会使用诈唬
        self.bluff_rate = 0.0  # 我方使用诈唬的概率
        self.certainly_win_prob = 0.0  # 必胜概率，表示离必胜还有多少筹码的差距。差距越小，下注应越谨慎
        self.hands_raise_range_result = pd.read_csv(
            'D:\\Development\\pythonProjects\\THPMaster\\competition\\core\\strategy\\hands_raise_range_result_6.csv',
            header=None)

    def strategy_1(self, episode, stage_first_action_flag) -> np.ndarray:
        """
        核心策略函数（入口）
        :return: [ndarray] 动作指令，[动作类型, 筹码两]
        """
        """
        暂时不考虑“已赢得足够筹码，后续对局均弃牌”这一策略
        """
        self.min_win_chip = (self.total_episode - episode) / 2 * 150 + 100  # todo 为了游戏体验，可以调大min_win_chip的值
        self.my_hands_win_rate, win_rate, tie_rate, lose_rate = (
            mc_my_hands_win_rate(self.my_hands, self.game.get_public_cards(), self.possible_oppo_hands))
        # 我方为坚果牌时的策略
        if lose_rate < 1e-6:
            if self.catzzz.game_total_chip + self.catzzz.total_earn_chip > self.min_win_chip:
                if stage_first_action_flag:
                    action = constants.CHECK_ACTION
                    chips = 0
                else:
                    action = constants.CALL_ACTION
                    chips = int(math.fabs(self.oppo.stage_chip - self.catzzz.stage_chip))
            # 当前我方累计输了若干筹码
            elif self.catzzz.total_earn_chip < -5000:
                action = constants.RAISE_ACTION
                chips = 100 + math.fabs(self.catzzz.total_earn_chip) + random.randint(0, 20)  # 最低下注100
            else:  # 当前我方累计已赢得筹码
                if self.game.cur_stage == constants.flop_stage:
                    weight = 0.9
                elif self.game.cur_stage == constants.turn_stage:
                    weight = 0.95
                else:
                    weight = 1.0
                action = constants.RAISE_ACTION
                chips = 100 + int(weight * (self.min_win_chip - self.catzzz.game_chip))
            if action == constants.RAISE_ACTION \
                    and (chips + self.catzzz.game_chip - self.catzzz.stage_chip) >= constants.total_chip:
                action = constants.ALLIN_ACTION
                chips = constants.total_chip
            print(f'func strategy()->我方为坚果牌，做出动作: {action}, {chips}')
            return np.array([action, chips])
        # 每个阶段的动作分为首次行动和后续行动
        if stage_first_action_flag:  # 本阶段我方先行动
            if self.game.cur_stage == constants.preflop_stage:
                print(f'func strategy()->进入preflop阶段，我方手牌胜率: {self.my_hands_win_rate}')
                self.preflop_strategy_first()
            # elif
            # todo 首次行动和后续行动所采用的策略有区别，如何通过self.game区分是否为每阶段的首次行动？
            #  可以通过self.game.CheckStateFuncResult判断是否进入下一状态，CheckStateFuncResult.enter_next_state表示刚进入这一阶段
            #  需确认self.game.CheckStateFuncResult指的是当前状态(未执行当前动作时的状态)还是下一状态(step函数返回的状态)？
            #  如果是下一状态，或许不需要stage_first_action_flag参数
        else:  # 本阶段已有玩家行动
            pass

    def strategy_2(self):
        legal_actions = self.game.get_legal_actions()
        if legal_actions[0] == 1:
            return np.array([constants.CHECK_ACTION, 0])
        elif legal_actions[1] == 1:
            return np.array([constants.CALL_ACTION, 0])
        else:
            return np.array([constants.RAISE_ACTION, 300])
        # if self.game.check_game_state() == self.game.CheckStateFuncResult.enter_next_state:
        #     return np.array([constants.CHECK_ACTION, 0])
        # if self.game.check_game_state() == self.game.CheckStateFuncResult.stay_cur_stage:
        #     return np.array([constants.CALL_ACTION, 0])
        # return np.array([constants.FOLD_ACTION, 0])

    def strategy(self, episode, stage_first_action_flag):
        self.episode = episode
        self.min_win_chip = (self.total_episode - episode) / 2 * 150 + 100  # 后续对局直接弃牌最终也能获胜的最低筹码
        # 已赢得足够筹码，后续对局均弃牌
        if self.catzzz.total_earn_chip >= self.min_win_chip:  # 只会在整局游戏的第一个动作时才有可能满足此条件
            print(
                f'筹码已足够，采取弃牌必胜策略，已赢得筹码：{self.catzzz.total_earn_chip},预计损失筹码：{self.min_win_chip}')
            return 'fold'
        # 蒙特卡洛模拟胜率
        self.my_hands_win_rate, win_rate, tie_rate, lose_rate = (
            mc_my_hands_win_rate(self.catzzz.hand_cards, self.game.public_cards, self.possible_oppo_hands))
        # 我方是坚果牌时的策略
        if lose_rate < 1e-6:
            if self.catzzz.game_chip + self.catzzz.total_earn_chip > self.min_win_chip:
                # 本局已下注筹码+累计赢得筹码大于min_win_chip，无需再下注就能保证最后获胜，此时我方可选择check或call
                if stage_first_action_flag:
                    action = constants.CHECK_ACTION
                    raise_chip = 0
                else:
                    action = constants.CALL_ACTION
                    raise_chip = int(math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip))
            elif self.catzzz.total_earn_chip < -5000:  # 当前我方累计输了筹码
                action = constants.RAISE_ACTION
                raise_chip = 100 + math.fabs(self.catzzz.total_earn_chip) + random.randint(0, 20)  # 最低下注100
            else:  # 当前我方累计已赢得筹码
                if self.current_stage == constants.flop_stage:
                    weight = 0.90
                elif self.current_stage == constants.turn_stage:
                    weight = 0.95
                else:
                    weight = 1
                action = constants.RAISE_ACTION
                raise_chip = 100 + int(weight * (self.min_win_chip - self.catzzz.game_chip))  # 赢得最后胜利所需的筹码减去本局已下注的筹码
            # 判断raise动作是否合理，本局下注总筹码大于20000需采取allin动作
            if action == constants.RAISE_ACTION \
                    and (raise_chip + self.catzzz.game_chip - self.catzzz.stage_chip) >= constants.total_chip:
                action = constants.ALLIN_ACTION
                raise_chip = constants.total_chip
            print(f'func strategy()->我方为坚果牌，做出动作: {action}, {raise_chip}')
            return convert_client_cmd(action, raise_chip)

        if stage_first_action_flag:  # 本阶段内我方首先行动
            # 一般情况
            if self.game.stage == constants.preflop_stage:
                if self.current_stage == constants.init_stage:  # 每局第一次进入到preflop阶段
                    self.current_stage = constants.preflop_stage
                    print(f'func strategy()->进入preflop阶段，我方手牌胜率: {self.my_hands_win_rate}')
                action, raise_chip = self.preflop_strategy_first()
            elif self.game.stage == constants.flop_stage:
                if self.current_stage == constants.preflop_stage:  # 每局第一次进入到flop阶段
                    self.current_stage = constants.flop_stage
                print(f"func strategy()->进入flop阶段，我方手牌胜率: {self.my_hands_win_rate}")
                action, raise_chip = self.flop_strategy_first()
            elif self.game.stage == constants.turn_stage:
                if self.current_stage == constants.flop_stage:
                    self.current_stage = constants.turn_stage
                print(f"func strategy()->进入turn阶段，我方手牌胜率: {self.my_hands_win_rate}")
                action, raise_chip = self.turn_strategy_first()
            elif self.game.stage == constants.river_stage:
                if self.current_stage == constants.turn_stage:
                    self.current_stage = constants.river_stage
                print(f"func strategy()->进入river阶段，我方手牌胜率: {self.my_hands_win_rate}")
                action, raise_chip = self.river_strategy_first()
            else:
                action = constants.FOLD_ACTION
                raise_chip = -1
                print(f"----------------func strategy()->进入未知的阶段，我方选择{action}----------------")
        else:  # 本阶段已有玩家采取行动
            # 对手做出allin，我方只能call或fold
            if self.opponent.operation[-1, 0] == constants.ALLIN_ACTION \
                    or self.opponent.game_chip >= constants.total_chip:
                if lose_rate < 1e-6:  # 我方是必胜的
                    print('func strategy()->对方做出allin动作，我方有必胜概率，选择call')
                    return 'call'
                else:
                    print('func strategy()->对方做出allin动作，我方没有必胜概率，选择fold')
                    return 'fold'
            # 一般情况
            if self.game.stage == constants.preflop_stage:
                print(f'func strategy()->已处于preflop阶段，我方手牌胜率: {self.my_hands_win_rate}')
                action, raise_chip = self.preflop_strategy_other()
            elif self.game.stage == constants.flop_stage:
                print(f"func strategy()->已处于flop阶段，我方手牌胜率: {self.my_hands_win_rate}")
                action, raise_chip = self.flop_strategy_other()
            elif self.game.stage == constants.turn_stage:
                print(f"func strategy()->已处于turn阶段，我方手牌胜率: {self.my_hands_win_rate}")
                action, raise_chip = self.turn_strategy_other()
            elif self.game.stage == constants.river_stage:
                print(f"func strategy()->已处于river阶段，我方手牌胜率: {self.my_hands_win_rate}")
                action, raise_chip = self.river_strategy_other()
            else:
                action = constants.FOLD_ACTION
                raise_chip = -1
                print(f"----------------func strategy()->处于未知的阶段，我方选择{action}----------------")
            # todo 判断raise动作是否合理
        return convert_client_cmd(action, raise_chip)


def convert_client_cmd(action, raise_chip) -> np.ndarray:
    pass


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
    print("func mc_my_hands_win_rate()->win:", win_count, "tie:", tie_count, "lose:", lose_count,
          "total:", total_count, "win_rate", score / (total_count * per_win_score))
    return score / (
            total_count * per_win_score), win_count / total_count, tie_count / total_count, lose_count / total_count


if __name__ == '__main__':
    pass
