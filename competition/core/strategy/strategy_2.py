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
from competition.core.constants import CARDS
from competition.core.entity import Game
from competition.util import common_util
from competition.util.poker_util import deal_cards
from competition.util.poker_value import best_cards


class Strategy:
    def __init__(self):
        # oppo_hands阶段亮手牌，可以看出对手的手牌松紧度，但是没必要
        self.min_win_chip = 70 / 2 * 150 + 1
        self.episode = 0
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
        self.hands_raise_range_result = pd.read_csv(
            'D:\\Development\\pythonProjects\\THPMaster\\competition\\core\\strategy\\hands_raise_range_result_6.csv',
            header=None)

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
        :return: str->做出的决策，需符合服务端指令格式要求
        """
        # action = constants.player_fold_action
        # raise_chip = -1
        self.min_win_chip = (70 - episode) / 2 * 150 + 100  # 后续对局直接弃牌最终也能获胜的最低筹码
        # 已赢得足够筹码，后续对局均弃牌
        if self.catzzz.total_earn_chip >= self.min_win_chip:
            print(
                f'筹码已足够，采取弃牌必胜策略，已赢得筹码：{self.catzzz.total_earn_chip},预计损失筹码：{self.min_win_chip}')
            return 'fold'
        self.my_hands_win_rate, win_rate, tie_rate, lose_rate = mc_my_hands_win_rate(self.catzzz.hand_cards,
                                                                                     self.game.public_cards,
                                                                                     self.possible_oppo_hands)

        if self.opponent.operation[-1, 0] == constants.player_allin_action:
            # 对手做出allin动作，我方只能选择fold或call
            if lose_rate < 1e-6:  # 我方是必胜的
                print('对方做出allin动作，我方有必胜概率，选择call')
                return 'call'
            else:
                print('对方做出allin动作，我方没有必胜概率，选择fold')
                return 'fold'

        if lose_rate < 1e-6:  # 我方是坚果牌时的策略
            # 标识当我方手牌为坚果牌时，下注筹码使我方暂时不输还是使我方弃牌也必胜
            # todo 这里可以更细化一些，例如比赛初期此处值可以更小，后期更大，跟episode有关的函数
            if self.catzzz.total_earn_chip < -5000:  # 当前我方累计输了筹码
                # 使我方暂时不输
                action = constants.player_raise_action
                raise_chip = math.fabs(self.catzzz.total_earn_chip)
                if raise_chip < self.opponent.stage_chip * 2:
                    action = constants.player_call_action
                    raise_chip = int(math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip))
                elif raise_chip + self.catzzz.game_chip - self.catzzz.stage_chip >= constants.total_chip:
                    action = constants.player_allin_action
                    raise_chip = constants.total_chip
            else:  # 当前我方累计已赢得筹码
                if self.current_stage == constants.flop_stage:
                    weight = 0.90
                elif self.current_stage == constants.turn_stage:
                    weight = 0.95
                else:
                    weight = 1
                action = constants.player_raise_action
                raise_chip = int(weight * (self.min_win_chip - self.catzzz.game_chip))  # 赢得最后胜利所需的筹码减去本局已下注的筹码
                if raise_chip <= 0:
                    action = constants.player_call_action
                    raise_chip = int(math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip))
                elif raise_chip <= self.opponent.stage_chip * 7 / 4 and self.opponent >= 100:  # 小于对方下注筹码3/2时，采取call
                    action = constants.player_call_action
                    raise_chip = int(math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip))
                elif self.opponent.stage_chip * 3 / 2 < raise_chip < self.opponent.stage_chip * 2:
                    action = constants.player_raise_action
                    raise_chip = int(2 * self.opponent.stage_chip)
            return convert_client_cmd(action, raise_chip)

        # 一般情况
        # preflop阶段下注筹码不超过1000，即手牌为AA时也只下注1000
        if self.game.stage == constants.preflop_stage:
            if self.current_stage == constants.init_stage:  # 每局第一次进入到preflop阶段
                self.current_stage = constants.preflop_stage
                print(f'func strategy()->进入preflop阶段，我方手牌胜率: {self.my_hands_win_rate}')
            action, raise_chip = self.preflop_strategy()
        elif self.game.stage == constants.flop_stage:
            if self.current_stage == constants.preflop_stage:  # 每局第一次进入到flop阶段
                self.current_stage = constants.flop_stage
                print(f"func strategy()->进入flop阶段，我方手牌胜率: {self.my_hands_win_rate}")
        return convert_client_cmd(action, raise_chip)

    def strategy_1(self, episode, catzzz_first_action_flag):
        self.episode = episode
        self.min_win_chip = (70 - episode) / 2 * 150 + 100  # 后续对局直接弃牌最终也能获胜的最低筹码
        # 已赢得足够筹码，后续对局均弃牌
        if self.catzzz.total_earn_chip >= self.min_win_chip:  # 只会在整局游戏的第一个动作时才有可能满足此条件
            print(
                f'筹码已足够，采取弃牌必胜策略，已赢得筹码：{self.catzzz.total_earn_chip},预计损失筹码：{self.min_win_chip}')
            return 'fold'
        # MC模拟胜率
        self.my_hands_win_rate, win_rate, tie_rate, lose_rate = (
            mc_my_hands_win_rate(self.catzzz.hand_cards, self.game.public_cards, self.possible_oppo_hands))
        # 我方是坚果牌时的策略
        if lose_rate < 1e-6:
            if self.catzzz.game_chip + self.catzzz.total_earn_chip > self.min_win_chip:
                # 本局已下注筹码+累计赢得筹码大于min_win_chip，无需再下注就能保证最后获胜，此时我方可选择check或call
                if catzzz_first_action_flag:
                    action = constants.player_check_action
                    raise_chip = 0
                else:
                    action = constants.player_call_action
                    raise_chip = int(math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip))
            elif self.catzzz.total_earn_chip < -5000:  # 当前我方累计输了筹码
                action = constants.player_raise_action
                raise_chip = 100 + math.fabs(self.catzzz.total_earn_chip) + random.randint(0, 20)  # 最低下注100
            else:  # 当前我方累计已赢得筹码
                if self.current_stage == constants.flop_stage:
                    weight = 0.90
                elif self.current_stage == constants.turn_stage:
                    weight = 0.95
                else:
                    weight = 1
                action = constants.player_raise_action
                raise_chip = 100 + int(weight * (self.min_win_chip - self.catzzz.game_chip))  # 赢得最后胜利所需的筹码减去本局已下注的筹码
            # 判断raise动作是否合理，本局下注总筹码大于20000需采取allin动作
            if action == constants.player_raise_action \
                    and (raise_chip + self.catzzz.game_chip - self.catzzz.stage_chip) >= constants.total_chip:
                action = constants.player_allin_action
                raise_chip = constants.total_chip
            print(f'func strategy()->我方为坚果牌，做出动作: {action}, {raise_chip}')
            return convert_client_cmd(action, raise_chip)

        if catzzz_first_action_flag:  # 本阶段内我方首先行动
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
                action = constants.player_fold_action
                raise_chip = -1
                print(f"----------------func strategy()->进入未知的阶段，我方选择{action}----------------")
        else:  # 本阶段已有玩家采取行动
            # 对手做出allin，我方只能call或fold
            if self.opponent.operation[-1, 0] == constants.player_allin_action \
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
                action = constants.player_fold_action
                raise_chip = -1
                print(f"----------------func strategy()->处于未知的阶段，我方选择{action}----------------")
            # todo 判断raise动作是否合理
        return convert_client_cmd(action, raise_chip)

    def preflop_strategy_first(self):
        oppo_stage_chip = self.opponent.stage_chip
        catzzz_stage_chip = self.catzzz.stage_chip
        card1_idx = CARDS.index(self.catzzz.hand_cards_raw[0:2])
        card2_idx = CARDS.index(self.catzzz.hand_cards_raw[2:4])
        hands_raise_chip = self.hands_raise_range_result.iloc[card1_idx, card2_idx]
        print(
            f'func preflop_strategy_first()->手牌对应字符串: {self.catzzz.hand_cards_raw}, 手牌对应下注筹码: {hands_raise_chip}')
        if oppo_stage_chip > hands_raise_chip:
            action = constants.player_fold_action
            raise_chip = -1
        elif hands_raise_chip > oppo_stage_chip > (hands_raise_chip * 3 / 5):
            action = constants.player_call_action
            raise_chip = int(math.fabs(oppo_stage_chip - catzzz_stage_chip) + random.randint(0, 20))
        else:
            action = constants.player_raise_action
            raise_chip = int(max(hands_raise_chip, 2 * oppo_stage_chip) + random.randint(0, 20))
        # 动作合法性检查
        if action == constants.player_raise_action \
                and (raise_chip + self.catzzz.game_chip - self.catzzz.stage_chip) >= constants.total_chip:
            action = constants.player_allin_action
            raise_chip = constants.total_chip
        elif action == constants.player_raise_action and raise_chip < 100:
            action = constants.player_check_action
            raise_chip = 0
        print(f'func preflop_strategy_first()->输出策略: {action}, {raise_chip}')
        return action, raise_chip

    def preflop_strategy_other(self):
        # todo 需要特殊判断小盲注call的情况
        oppo_stage_chip = self.opponent.stage_chip
        catzzz_stage_chip = self.catzzz.stage_chip
        card1_idx = CARDS.index(self.catzzz.hand_cards_raw[0:2])
        card2_idx = CARDS.index(self.catzzz.hand_cards_raw[2:4])
        hands_raise_chip = self.hands_raise_range_result.iloc[card1_idx, card2_idx]
        if hands_raise_chip <= (oppo_stage_chip * 2 / 3):
            if self.opponent.operation[-1, 0] == constants.player_check_action \
                    and self.catzzz.stage_chip == self.opponent.stage_chip == 100:
                print('对方check，我方不弃牌，选择call（理论上对方check之后会直接进入flop阶段，而不再进行我方的行动）')
                action = constants.player_call_action
                raise_chip = 0
            else:
                if hands_raise_chip >= 900 and self.my_hands_win_rate > 0.66:
                    print(f'hands_rise_chip={hands_raise_chip}>900,my_hands_win_rate={self.my_hands_win_rate}，我方不弃牌')
                    action = constants.player_call_action
                    raise_chip = 0
                else:
                    print(
                        f'我方hands_raise_chip={hands_raise_chip}(大于900)，我方不弃牌，选择call（理论上对方check之后会直接进入flop阶段，而不再进行我方的行动）')
                    action = constants.player_fold_action
                    raise_chip = -1
            # action = constants.player_fold_action
            # raise_chip = -1
        elif (oppo_stage_chip * 2 / 3) <= hands_raise_chip < (oppo_stage_chip * 4 / 3):
            if (self.catzzz.stage_chip == self.opponent.stage_chip == 100
                    and self.catzzz.identity == constants.BIGBLIND):
                action = constants.player_check_action
                raise_chip = 0
            else:
                action = constants.player_call_action
                raise_chip = int(math.fabs(oppo_stage_chip - catzzz_stage_chip) + random.randint(0, 20))
        elif (oppo_stage_chip * 4 / 3) <= hands_raise_chip < (oppo_stage_chip * 2):
            action = constants.player_raise_action
            raise_chip = int(oppo_stage_chip * 2 + random.randint(0, 20))
        else:
            action = constants.player_raise_action
            raise_chip = int(hands_raise_chip + random.randint(0, 20))
        # if oppo_stage_chip > (hands_raise_chip * 2 / 3):
        #     action = constants.player_fold_action
        #     raise_chip = -1
        # elif (hands_raise_chip * 2 / 3) <= oppo_stage_chip < (hands_raise_chip * 4 / 3):
        #     action = constants.player_call_action
        #     raise_chip = int(math.fabs(oppo_stage_chip - catzzz_stage_chip) + random.randint(0, 20))
        # else:
        #     action = constants.player_raise_action
        #     raise_chip = int(hands_raise_chip + random.randint(0, 20))
        print(f"func preflop_strategy_other()->hand_raise_chip={hands_raise_chip},"
              f"raise_chip={raise_chip},action={action}")
        return action, raise_chip

    def flop_strategy_first(self):
        # todo 主动下注也应考虑防止斩杀？
        # 我方手牌胜率不高时，先check（首次下注不用fold，即使我方胜率不高）
        if self.my_hands_win_rate <= 0.4:
            return constants.player_check_action, 0
        elif self.my_hands_win_rate < 0.5:
            return constants.player_raise_action, 101 + random.randint(0, 10)
        # 由于是本阶段内第一次下注，无法计算底池赔率，使用下注比率代替
        raise_rate = 2 * self.my_hands_win_rate - 1  # 下注比率，只要胜率大于0.5，下注比率就是正的
        # 小于0.5，再次降低下注比率；大于0.7，再次增加下注比率
        if raise_rate < 0.4:
            raise_rate = max(0.1, raise_rate * 0.85)
        elif raise_rate > 0.6:
            raise_rate = min(1.0, raise_rate * 1.2)
        action = constants.player_raise_action
        raise_chip = int(1000 * raise_rate + random.randint(0, 10))  # 主动下注最高为1000
        # 检查raise动作合法性
        if action == constants.player_raise_action and raise_chip < 100:
            action = constants.player_check_action
            raise_chip = 0
        elif action == constants.player_raise_action \
                and (raise_chip + self.catzzz.game_chip - self.catzzz.stage_chip) >= constants.total_chip:
            action = constants.player_allin_action
            raise_chip = constants.total_chip
        print(f'func flop_strategy_first()->输出策略: {action}, {raise_chip}')
        return action, raise_chip

    def flop_strategy_other(self):
        # 被动下注策略，此时本阶段内已有玩家下注
        # todo 防斩杀，大于30局才考虑
        # 如果我们跟注但输掉这局，会被对方斩杀
        if self.opponent.game_chip - self.catzzz.total_earn_chip > self.min_win_chip:  # 对方下注过大，可能会被斩杀
            if self.episode > 15:  # 大于15局就采取防斩杀策略
                print('flop_strategy_other()->大于15局就采取防斩杀策略，弃牌')
                if self.opponent.operation[-1, 0] == constants.player_check_action:
                    print('对方check，我方不弃牌，选择call')
                    return constants.player_call_action, 0
                else:
                    return constants.player_fold_action, -1
                # if self.my_hands_win_rate < 0.8:  # 当手牌胜率过低时，下注只会损失更多筹码
                #     if self.episode < 60:   # 如果只剩最后几局
                #         return constants.player_fold_action, -1
                #     else:
                #         action = constants.player_call_action
                #         raise_chip = int(
                #             math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                #         return action, raise_chip
                # else:
                #     action = constants.player_call_action
                #     raise_chip = int(
                #         math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                #     return action, raise_chip
                # if self.my_hands_win_rate < 0.95 and self.catzzz.total_earn_chip > -self.min_win_chip:
                #     return constants.player_fold_action, -1
                # elif self.my_hands_win_rate < 0.95 and self.catzzz.total_earn_chip <= -self.min_win_chip:
                #     action = constants.player_call_action
                #     raise_chip = int(
                #         math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                #     return action, raise_chip
                # else:
                #     action = constants.player_call_action
                #     raise_chip = int(
                #         math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                #     return action, raise_chip
            else:  # 小于15局
                if self.my_hands_win_rate < 0.6:
                    print('flop_strategy_other()->小于15局但胜率低于0.6，弃牌')
                    # todo 如果对方是check，即使我们手牌不好，我们也可以call而不是fold（我方后手时都适用）
                    if self.opponent.operation[-1, 0] == constants.player_check_action:
                        print('对方check，我方不弃牌，选择call')
                        return constants.player_call_action, 0
                    else:
                        return constants.player_fold_action, -1
                else:
                    print('flop_strategy_other()->小于15局但胜率高于0.6（不采取防斩杀），跟注')
                    action = constants.player_call_action
                    raise_chip = int(
                        math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                    return action, raise_chip
        else:  # 对方下注筹码没有达到斩杀线
            if self.my_hands_win_rate <= 0.4:
                print('flop_strategy_other()->对方下注未达到斩杀线，我方胜率低于0.4，弃牌')
                if self.opponent.operation[-1, 0] == constants.player_check_action:
                    print('对方check，我方不弃牌，而选择call')
                    action = constants.player_call_action
                    raise_chip = 0
                else:
                    print('对方未check，我方弃牌')
                    action = constants.player_fold_action
                    raise_chip = -1
            elif 0.4 < self.my_hands_win_rate <= 0.6:
                if self.opponent.stage_chip < 2000:  # 下注筹码小于2000，选择跟注
                    print('flop_strategy_other()->对方下注未达到斩杀线，我方胜率在0.4到0.6之间，下注筹码小于2000，跟注')
                    action = constants.player_call_action
                    raise_chip = int(
                        math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                else:
                    print('flop_strategy_other()->对方下注未达到斩杀线，我方胜率在0.4到0.6之间，下注筹码大于2000，弃牌')
                    action = constants.player_fold_action
                    raise_chip = -1
                # action = constants.player_call_action
                # raise_chip = int(
                #     math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
            elif 0.6 < self.my_hands_win_rate <= 0.8:
                if self.opponent.stage_chip > 1500:  # 下注筹码大于2000，选择跟注
                    action = constants.player_call_action
                    raise_chip = int(
                        math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                else:  # 对方下注筹码小于2000
                    call_odds = min(self.game.total_chip / (self.opponent.stage_chip + 1),
                                    5.0)  # stage_chip+1防止对方check时分母为0
                    f_kelly = (call_odds * 0.9 * self.my_hands_win_rate + 0.9 * self.my_hands_win_rate - 1) / call_odds
                    raise_chip = f_kelly * 2000
                    action = constants.player_raise_action
                    # 判断动作合法性
                    if self.opponent.stage_chip < raise_chip <= self.opponent.stage_chip * 3 / 2:
                        # 下注筹码大于1倍但小于3/2倍，选择call
                        action = constants.player_call_action
                        raise_chip = int(
                            math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                    elif self.opponent.stage_chip * 3 / 2 < raise_chip <= self.opponent.stage_chip * 2:
                        # 下注筹码大于3/2但小于2倍，按照2倍下注
                        action = constants.player_raise_action
                        raise_chip = int(2 * self.opponent.stage_chip + random.randint(0, 20))
                    elif self.opponent.stage_chip * 2 < raise_chip:
                        action = constants.player_raise_action
                        raise_chip = int(f_kelly * 2000)
                print(
                    f'flop_strategy_other()->对方下注未达到斩杀线，我方胜率在0.6到0.8之间，选择动作: {action},{raise_chip}')
            else:  # 手牌胜率大于0.8
                if self.opponent.stage_chip > 4000:
                    action = constants.player_call_action
                    raise_chip = int(
                        math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                else:
                    call_odds = max(self.game.total_chip / (self.opponent.stage_chip + 1), 5.0)
                    f_kelly = (call_odds * 0.9 * self.my_hands_win_rate + 0.9 * self.my_hands_win_rate - 1) / call_odds
                    raise_chip = f_kelly * 4000
                    action = constants.player_raise_action
                    # 判断动作合法性
                    if self.opponent.stage_chip < raise_chip <= self.opponent.stage_chip * 3 / 2:
                        # 下注筹码大于1倍但小于3/2倍
                        action = constants.player_call_action
                        raise_chip = int(
                            math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                    elif self.opponent.stage_chip * 3 / 2 < raise_chip <= self.opponent.stage_chip * 2:
                        # 下注筹码大于3/2但小于2倍，按照2倍下注
                        action = constants.player_raise_action
                        raise_chip = int(2 * self.opponent.stage_chip + random.randint(0, 20))
                    elif self.opponent.stage_chip * 2 < raise_chip:
                        action = constants.player_raise_action
                        raise_chip = f_kelly * 4000
                print(
                    f'flop_strategy_other()->对方下注未达到斩杀线，我方胜率在大于0.8，选择动作: {action},{raise_chip}')

            # 判断raise动作是否合法
            if action == constants.player_raise_action:
                if raise_chip < 2 * self.opponent.stage_chip:
                    action = constants.player_call_action
                    raise_chip = int(
                        math.fabs(self.opponent.stage_chip - self.catzzz.stage_chip) + random.randint(0, 20))
                elif (raise_chip + self.catzzz.game_chip - self.catzzz.stage_chip) >= constants.total_chip:
                    action = constants.player_allin_action
                    raise_chip = constants.total_chip
            return action, raise_chip

    def turn_strategy_first(self):
        return self.flop_strategy_first()

    def turn_strategy_other(self):
        return self.flop_strategy_other()

    def river_strategy_first(self):
        return self.flop_strategy_first()

    def river_strategy_other(self):
        return self.flop_strategy_other()

    def preflop_strategy(self):
        # 已废弃
        # todo 如何让flop,turn,river阶段也能使用上ACPC数据，或者使用博弈树搜索
        oppo_stage_chip = self.opponent.stage_chip
        catzzz_stage_chip = self.catzzz.stage_chip
        # if catzzz_stage_chip == oppo_stage_chip == 100:  # 我方作为大盲注第一次下注且可以check，此时check我们并不需要付出筹码
        #     print('func preflop_strategy()->我方作为大盲注第一次下注且可以check')
        #     return constants.player_check_action, 0
        card1_idx = CARDS.index(self.catzzz.hand_cards_raw[0:2])
        card2_idx = CARDS.index(self.catzzz.hand_cards_raw[2:4])
        hands_raise_chip = self.hands_raise_range_result.iloc[card1_idx, card2_idx]
        print(f'手牌对应字符串: {self.catzzz.hand_cards_raw}, 手牌对应下注筹码: {hands_raise_chip}')
        if oppo_stage_chip > hands_raise_chip:
            action = constants.player_fold_action
            raise_chip = -1
        elif hands_raise_chip > oppo_stage_chip > (hands_raise_chip * 3 / 5):
            action = constants.player_call_action
            raise_chip = int(math.fabs(oppo_stage_chip - catzzz_stage_chip))
        else:
            action = constants.player_raise_action
            raise_chip = int(max(hands_raise_chip, 2 * oppo_stage_chip))
        print(f'func preflop_strategy()->输出策略: {action}, {raise_chip}')
        return action, raise_chip


def convert_client_cmd(action, raise_chip) -> str:
    """
    将[action, raise_chip]格式的指令转为字符串格式
    """
    if action == constants.player_call_action:
        return 'call'
    elif action == constants.player_check_action:
        return 'check'
    elif action == constants.player_raise_action:
        if raise_chip >= constants.total_chip:
            return 'allin'
        else:
            return 'raise ' + str(int(raise_chip))
    elif action == constants.player_allin_action:
        return 'allin'
    elif action == constants.player_fold_action:
        return 'fold'
    else:
        common_util.print_exception(convert_client_cmd, '未知的客户端指令')
        return 'fold'


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
    my = np.array([[1, 5], [1, 12]])
    # public = np.array([[1, 9], [2, 3], [1, 4], [2, 4], [1, 12]])
    public = np.array([[1, 0], [2, 8], [2, 11], [3, 7]])

    my1 = np.array([[0, 12], [0, 11]])
    public1 = np.array([[0, 10], [0, 9], [0, 8], [1, 3], [2, 0]])
    mc_my_hands_win_rate(my1, public1)
