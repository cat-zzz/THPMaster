"""
@project: machine_game
@File   : game5.py.py
@Desc   : 德州扑克游戏逻辑（两人，一局一复位）
@Author : gql
@Date   : 2024/3/22 13:15
"""
from enum import Enum

import numpy as np

from src.env import card_util, constants, poker_value

"""
参数
seed：随机数种子
show all hands：默认若进行比牌，则向所有玩家都展示手牌
reset_chip_by_one：筹码默认为一局一复位
store_all_game_history：全部对局的历史序列，还是只保存一局序列
"""
default_game_config = {
    'seed': 0,  # 随机数种子，为0表示不设置随机数种子
    'show_all_hands': True,  # 默认一旦进行比牌，则向所有玩家都展示手牌
    'one_game_reset': True,  # todo 筹码默认为一局一复位（未实现）
    'store_all_game_history': False,  # game_history是否保存全部对局的历史序列，还是只保存一局的序列
}

"""
系统动作
[-1, 各玩家身份、手牌]
[-1, 公共牌]

游戏阶段
prepare_round()
    游戏初始准备阶段，所做的工作有确定牌面（各玩家、公共牌）、下大小盲注、分发手牌
    下小盲注是整局游戏的第一个动作，下大盲注是第二个动作，所以prev_player_idx为大盲注玩家索引

其他
牌面信息均使用numpy表示，格式[花色, 点数]
动作信息均使用numpy表示，格式[动作类型, 具体筹码量]，有些动作并不需要筹码量，其对应的具体筹码量是无意义的
"""


class NoLimitHoldemGame:
    def __init__(self, game_config=None):
        if game_config is None:
            game_config = default_game_config
        self.config = game_config
        if game_config['seed'] != 0:
            np.random.seed(game_config['seed'])
        self.name = 'No Limit Holdem Game'
        self.num_players = 2
        self.players = [Player(_) for _ in range(self.num_players)]  # 所有玩家
        # 每局游戏开始需要更新的变量
        self.all_cards = None
        self.small_idx = 0
        self.big_idx = 0
        self.payoffs = [0] * self.num_players
        # 每个阶段需要更新的变量
        self.cur_stage = None
        # 每次行动需要更新的变量
        self.pot_chip = 0  # 本局游戏总下注量
        self.folded_player_count = 0  # 弃牌玩家数
        self.allin_player_count = 0  # Allin玩家数
        # 每次行动或每个阶段都需要更新的变量
        self.cur_player_idx = 0  # 当前轮到哪位玩家行动
        self.prev_player_idx = 0  # 上一次行动的玩家
        self.cur_stage_chip = 0  # 当前阶段已下注的筹码量之和
        # valid_bet_chip：如果call动作，则下注量与此值相等，如果是raise动作，则下注量是此值的两倍及以上
        self.valid_bet_chip = 0  # 最新一次有效下注量
        self.same_bet_player_count = 1  # 做出一次raise或allin的玩家及紧接着连续call或check的玩家总数
        self.game_history = []
        # self.last_action = []

    def reset(self):
        self.prepare_round()

    def prepare_round(self):
        """
        准备阶段
        """
        if self.small_idx == 0 and self.big_idx == 0:  # 第一局开始
            self.small_idx = 0
            self.big_idx = 1
        else:  # 不是第一局
            self.small_idx = (self.small_idx + 1) % self.num_players
            self.big_idx = (self.big_idx + 1) % self.num_players
        # 无论是两人德扑还是多人德扑，首次行动的玩家都是大盲位的下一位
        self.cur_player_idx = (self.big_idx + 1) % self.num_players
        # 下小盲注是整局游戏的第一个动作，下大盲注是第二个动作
        self.prev_player_idx = self.big_idx  # 游戏开始时的前一位玩家为大盲注玩家
        # 确定所有牌面
        self.all_cards = card_util.deal_cards(self.num_players * 2 + 5)
        # 确定各玩家身份
        for i, p in enumerate(self.players):
            if i == self.small_idx:
                p.init_player(constants.SMALLBLIND)
            elif i == self.big_idx:
                p.init_player(constants.BIGBLIND)
        # 下盲注
        self.pot_chip = self.cur_stage_chip = constants.SMALLBLIND_CHIP + constants.BIGBLIND_CHIP
        # 重置其他变量
        self.cur_stage = constants.preflop_stage
        self.folded_player_count = 0
        self.valid_bet_chip = constants.BIGBLIND_CHIP
        self.allin_player_count = 0
        self.same_bet_player_count = 1
        self.payoffs = [0] * self.num_players

    def step(self, action: np.ndarray):
        """
        交互一步
        仅raise动作需要具体筹码量
        :param action: 所做的动作，ndarray [动作类型, 具体筹码]
        :return: info: 如果是在进入下一阶段时，info保存了隐藏的一次call行动；如果是在earn chip阶段，info保存了各玩家输赢筹码量
        """
        info = {}
        down = False
        if self.check_action(action):
            is_legal = True
            self.take_action(action)
            game_state_flag = self._check_game_state()
            if game_state_flag == self.CheckStateFuncResult.enter_next_state:
                # 隐藏的每个阶段最后一次call动作，包括river阶段最后的call，
                # 所以在earn_chip阶段判断输赢时，info中既包括payoffs，又包括hide_action
                info['hide_action'] = [self.cur_player_idx, action]
                self._enter_next_stage()
                # 正常比牌进入到earn chip阶段
                if self.cur_stage == constants.earn_chip_stage:
                    payoffs = self._enter_earn_chip(game_state_flag)
                    info['payoffs'] = payoffs
                    down = True
            elif (game_state_flag == self.CheckStateFuncResult.folded_enter_earn_chip_stage
                  or game_state_flag == self.CheckStateFuncResult.allin_enter_earn_chip_stage):
                # fold或allin进入earn chip阶段
                payoffs = self._enter_earn_chip(game_state_flag)
                info['payoffs'] = payoffs
                down = True
            else:
                # 游戏仍在当前阶段内
                self._next_player_idx()
        else:  # 动作不合法
            is_legal = False
            game_state_flag = self.CheckStateFuncResult.stay_cur_stage
        state = self.get_state()
        return state, is_legal, game_state_flag, down, info

    def take_action(self, action):
        """
        执行动作
        仅raise动作需要给出具体筹码量，其他动作均不需要；
        此函数不进行动作合法性检查；
        带有'每次行动需要更新的变量'标签的成员变量均在此函数中更新；
        带有'每次行动或每个阶段都需要更新的变量'标签的成员变量也会在此函数（但不限于此函数）中更新。
        :param action: ndarray [动作类型, 具体筹码量]
        """
        cur_p = self.players[self.cur_player_idx]
        prev_p = self.players[self.prev_player_idx]
        cur_player_chip = cur_p.cur_stage_chip
        if action[0] == constants.CHECK_ACTION:
            # todo 只有每个阶段开始的第一次动作可以是check，即一个阶段中check只会出现一次（包括preflop阶段的大盲注）
            self.same_bet_player_count = 1
            action1 = np.array([constants.CHECK_ACTION, 0], dtype=int)
            cur_p.take_action(action1)
        elif action[0] == constants.CALL_ACTION:
            # 与当前阶段最新一次的有效下注量相同，不能与前一位玩家的下注量相同，因为前一位玩家可能会fold
            call_chip = self.valid_bet_chip
            action1 = np.array([constants.CALL_ACTION, call_chip], dtype=int)
            cur_p.take_action(action1)
            self.pot_chip += (action1[1] - cur_player_chip)
            self.cur_stage_chip += (action1[1] - cur_player_chip)
            # 特殊情况：preflop阶段，小盲注call，大盲注check，小盲注再call
            if self.cur_stage != constants.preflop_stage:
                self.same_bet_player_count += 1
            elif prev_p.cur_stage_chip > constants.BIGBLIND_CHIP:
                self.same_bet_player_count += 1
        elif action[0] == constants.ALLIN_ACTION:
            action1 = np.array([constants.ALLIN_ACTION, constants.total_chip], dtype=int)
            cur_p.take_action(action1)
            self.pot_chip += (action1[1] - cur_player_chip)
            self.cur_stage_chip += (action1[1] - cur_player_chip)
            self.valid_bet_chip = action[1]
            self.allin_player_count += 1
            self.same_bet_player_count += 1
        elif action[0] == constants.FOLD_ACTION:
            action1 = np.array([constants.FOLD_ACTION, 0], dtype=int)
            cur_p.take_action(action1)
            self.folded_player_count += 1
        elif action[0] == constants.RAISE_ACTION:
            action1 = action
            cur_p.take_action(action1)
            self.pot_chip += (action[1] - cur_player_chip)  # 本次raise新增的筹码量
            self.cur_stage_chip += (action[1] - cur_player_chip)
            self.valid_bet_chip = action[1]
            self.same_bet_player_count = 1
        else:
            action1 = np.array([0, 0], dtype=int)
            card_util.print_exception(self.take_action, '未知的动作类型 ' + action)
        # 更新历史序列
        self.game_history.append([self.cur_player_idx, action1])
        # self._next_player_idx()

    class CheckStateFuncResult(Enum):
        enter_next_state = constants.func_enter_next_stage  # 游戏进入下一阶段（包括earn chip阶段，此时属于正常比牌）
        stay_cur_stage = constants.func_not_enter_next_stage  # 游戏仍在当前阶段
        folded_enter_earn_chip_stage = constants.func_direct_enter_earn_chip_stage  # 由于其他玩家弃牌而直接进入到earn_chip阶段
        allin_enter_earn_chip_stage = constants.func_allin_enter_earn_chip_stage  # 由于所有玩家均为Allin而直接进入到earn_chip阶段

    def check_action(self, action):
        """
        判断当前action是否合法
        :return: 合法返回True，非法返回False
        """
        legal_actions = self.get_legal_actions()
        # 下注最多为20000，且下注为20000时应使用Allin动作
        if action[0] != constants.ALLIN_ACTION and action[1] > constants.TOTAL_CHIP:
            return False
        if action[0] == constants.CHECK_ACTION and legal_actions[0] == 1:
            return True
        elif action[0] == constants.CALL_ACTION and legal_actions[1] == 1:
            return True
        elif action[0] == constants.FOLD_ACTION:
            return True
        elif action[0] == constants.ALLIN_ACTION and legal_actions[3] == 1:
            return True
        elif action[0] == constants.RAISE_ACTION and action[1] >= legal_actions[4]:
            return True
        else:
            return False

    def get_legal_actions(self):
        """
        得到当前状态下合法的动作
        动作类型
        check，call，fold，allin，raise
        check：每个阶段开始的第一个动作
        call：每个阶段开始的第一个动作不能是call，且有玩家allin时也不能是call。call适用于check、allin或raise
        fold：任何时候都可以
        allin：有玩家allin，其余玩家只能fold或call
        :return: [check, call, fold, allin, raise_chip]，前四个取值为0或1；raise_chip取值为具体的筹码量，若为0表示不可raise
        """
        if self.allin_player_count >= 1:  # 有玩家allin
            # 只能allin或fold
            legal_actions = [0, 1, 1, 0, 0]
            return legal_actions
        i = 0
        # 此处只判断有多少玩家跟注，还需确保游戏此时在preflop阶段
        for p in self.players:
            if p.game_total_chip == constants.BIGBLIND_CHIP:
                i += 1
        if self.valid_bet_chip == 0:
            if self.same_bet_player_count == 1:
                # 不能仅根据valid_bet_chip判断是不是第一个动作（每个阶段的第一个动作是check）
                legal_actions = [0, 1, 1, 1, constants.min_chip]
            else:
                # 每个阶段的第一次动作（preflop阶段除外）
                legal_actions = [1, 0, 1, 1, constants.min_chip]
            return legal_actions
        elif (self.cur_stage == constants.preflop_stage and i == self.num_players
              and self.players[self.cur_player_idx].identity == constants.BIGBLIND):
            # preflop阶段其他玩家call，大盲注此时可以check
            legal_actions = [1, 0, 1, 1, 2 * constants.BIGBLIND_CHIP]
            return legal_actions
        else:
            # 一般下注情况（一个阶段内第二次及以上的下注）
            legal_actions = [0, 1, 1, 1, 2 * self.valid_bet_chip]
            return legal_actions

    def get_hand_cards(self, player_idx):
        """
        获取玩家手牌
        :param player_idx: 玩家索引
        :return: 玩家手牌
        """
        return self.all_cards[player_idx * 2:player_idx * 2 + 2]

    def get_public_cards(self):
        # 获取公共牌
        if self.cur_stage == constants.flop_stage:
            public_cards = self.all_cards[-5:-2]
        elif self.cur_stage == constants.turn_stage:
            public_cards = self.all_cards[-5:-1]
        elif self.cur_stage == constants.river_stage or self.cur_stage == constants.earn_chip_stage:
            public_cards = self.all_cards[-5:]
        else:
            # preflop
            public_cards = np.zeros((0, 2), dtype=int)
        return public_cards

    def get_state(self):
        # 获取公共牌
        # if self.cur_stage == constants.flop_stage:
        #     public_cards = self.all_cards[-5:-2]
        # elif self.cur_stage == constants.turn_stage:
        #     public_cards = self.all_cards[-5:-1]
        # elif self.cur_stage == constants.river_stage or self.cur_stage == constants.earn_chip_stage:
        #     public_cards = self.all_cards[-5:]
        # else:
        #     # preflop
        #     public_cards = np.zeros((0, 2), dtype=int)
        public_cards = self.get_public_cards()
        state = {'cur_stage': self.cur_stage, 'cur_stage_chip': self.cur_stage_chip,
                 'pot_chip': self.pot_chip, 'public_cards': public_cards,
                 'cur_player_idx': self.cur_player_idx, 'players:': self.players}
        return state

    def _check_game_state(self):
        """
        判断游戏处于何种状态，是否进入下一阶段，是否进入earn chip阶段
        只进行判断，不修改成员变量
        :return:
        """
        # 1 判断所有未弃牌玩家是否均为Allin
        i = 0  # 记录allin的玩家数
        j = 0  # 记录active状态的玩家数
        for player in self.players:
            if player.player_state == constants.player_allin:
                i += 1
            elif player.player_state == constants.player_active:
                j += 1
        # 无论是两人德扑还是多人德扑，只要有2人及以上玩家Allin，且其余玩家都是fold（即active状态的玩家数为0），则翻出后续所有公共牌，直接比牌
        if i >= 2 and j == 0:
            state_flag = self.CheckStateFuncResult.folded_enter_earn_chip_stage
            return state_flag
        # 2 判断游戏是否进入下一阶段
        if self.folded_player_count == self.num_players - 1:
            # 只有一位玩家未弃牌，说明游戏直接进入到earn chip阶段
            state_flag = self.CheckStateFuncResult.folded_enter_earn_chip_stage
        elif self.same_bet_player_count == self.num_players - self.folded_player_count:
            # preflop阶段，其他玩家call，此时可从大盲注开始再进行一轮下注
            if self.cur_stage == constants.preflop_stage and self.valid_bet_chip == constants.BIGBLIND_CHIP:
                state_flag = self.CheckStateFuncResult.stay_cur_stage
            else:
                state_flag = self.CheckStateFuncResult.enter_next_state
        else:
            # 以上情况都不是，说明游戏还在当前阶段内
            state_flag = self.CheckStateFuncResult.stay_cur_stage
        return state_flag

    def _enter_next_stage(self):
        """
        游戏正常进入下一个下注轮(不包括earn chip阶段)要做的操作
        发公共牌、更新游戏阶段、确定当前玩家索引
        """
        if self.cur_stage == constants.preflop_stage:
            self.cur_stage = constants.flop_stage
            self.game_history.append([-1, self.all_cards[-5:-2]])
        elif self.cur_stage == constants.flop_stage:
            self.cur_stage = constants.turn_stage
            self.game_history.append([-1, self.all_cards[-2]])
        elif self.cur_stage == constants.turn_stage:
            self.cur_stage = constants.river_stage
            self.game_history.append([-1, self.all_cards[-1]])
        elif self.cur_stage == constants.river_stage:
            self.cur_stage = constants.earn_chip_stage
        else:
            card_util.print_exception(self._enter_next_stage, '未知的游戏阶段 ' + self.cur_stage)
        # 更新相关变量
        self.cur_stage_chip = 0
        self.valid_bet_chip = 0
        self.same_bet_player_count = 0
        # 更新当前行动玩家索引
        if self.cur_stage == constants.preflop_stage:
            self.cur_player_idx = self.small_idx
            self.prev_player_idx = self.small_idx
        else:
            self.cur_player_idx = self.big_idx
            self.prev_player_idx = self.big_idx
        # 更新每个玩家的cur_stage_chip
        for p in self.players:
            if p.player_state == constants.player_active:
                p.enter_next_stage()

    def _enter_earn_chip(self, state_flag):
        """
        进入earn chip阶段意味着游戏结束，判断输赢筹码
        :param state_flag: 表示以什么状态进入earn chip阶段（正常比牌、所有Allin、其余玩家Fold）
        :return: 每位玩家输赢的筹码
        """
        payoffs = [0] * self.num_players
        if (state_flag == self.CheckStateFuncResult.enter_next_state
                or state_flag == self.CheckStateFuncResult.allin_enter_earn_chip_stage):
            # 正常比牌进入earn chip阶段
            # 计算所有玩家的最大牌力值，若玩家已弃牌，则牌力值为0
            # poker_values格式[[玩家1的索引，玩家1的最大牌力值], [玩家2的索引，玩家2的最大牌力值]...]
            poker_values = np.zeros((0, 2), dtype=int)
            poker_values = np.vstack((poker_values, [
                [i, poker_value.best_cards(self.all_cards[2 * i:2 * i + 2], self.all_cards[-5:])[0]]
                if self.players[i] != constants.player_folded else [i, 0]
                for i in range(self.num_players)]))
            # poker_values = np.vstack((poker_values, [[i, poker_value.best_cards(p.hands, self.public_cards)[0]]
            #                                          if p.player_state != constants.player_folded else [i, 0]
            #                                          for i, p in enumerate(self.players)]))
            # poker_values = poker_values[np.argsort(poker_values, axis=0)[:, 1], :]  # 按照第2列（索引为1）排序
            # 找出牌力值最大的玩家（可能不止一位），即找出若poker_values第二列元素的最大值所在的整个一行的数据
            # 示例: 若poker_values的值为[[0, 0], [1, 2865830], [2, 2017222], [3, 2865830]]，
            # 则max_poker_values的值为[[1, 2865830], [3, 2865830]]
            max_poker_values = poker_values[np.where(poker_values == np.max(poker_values[:, 1], axis=0))[0], :]
            # 计算赢家获得筹码，未考虑赢得的筹码可能为小数的情况
            player_earn_chip = int(self.pot_chip / max_poker_values.shape[0])  # shape[0]表示行数
            # 更新每位玩家的输赢筹码量
            j = 0
            for i, p in enumerate(self.players):
                # j < max_poker_values.shape[0]为了防止越界
                if j < max_poker_values.shape[0] and i == max_poker_values[j, 0]:  # 赢家
                    p.game_over(player_earn_chip - p.game_total_chip)
                    payoffs[i] = player_earn_chip - p.game_total_chip
                    j += 1
                else:  # 输家
                    p.game_over(-p.game_total_chip)
                    payoffs[i] = -p.game_total_chip
        elif state_flag == self.CheckStateFuncResult.folded_enter_earn_chip_stage:
            # 场上只剩一位玩家未弃牌，底池全归那位未弃牌的玩家所有
            for i, p in enumerate(self.players):
                if p.player_state == constants.player_active:
                    p.game_over(self.pot_chip - p.game_total_chip)
                    payoffs[i] = self.pot_chip - p.game_total_chip
                elif p.player_state == constants.player_folded:
                    p.game_over(p.game_total_chip)
                    payoffs[i] = -p.game_total_chip
        else:
            card_util.print_exception(self._enter_earn_chip, '进入earn chip阶段时出现未知的state_flag ' + state_flag)
        return payoffs

    def _next_player_idx(self):
        """
        更新下一个行动玩家的索引
        需保证此时场上仍有未弃牌、未Allin的玩家，不然会导致此函数死循环
        """
        # 更新prev_player_idx
        self.prev_player_idx = self.cur_player_idx
        # 更新cur_player_idx，需要跳过已经弃牌和Allin的玩家
        self.cur_player_idx = (self.cur_player_idx + 1) % self.num_players
        while (self.players[self.cur_player_idx].player_state == constants.player_folded
               or self.players[self.cur_player_idx].player_state == constants.player_allin):
            self.cur_player_idx = (self.cur_player_idx + 1) % self.num_players

    def step_back(self):
        pass

    def print_info(self):
        print('-----------------游戏信息----------------------')
        print('all_cards:', str(self.all_cards).replace('\n', ' '))
        print(f'stage: {self.cur_stage}, small_idx: {self.small_idx}, big_idx: {self.big_idx}, '
              f'player_num: {self.num_players}')
        print(f'pot: {self.pot_chip}, same bet player count: {self.same_bet_player_count}, '
              f'fold player count: {self.folded_player_count}')
        print(f'cur_player_idx: {self.cur_player_idx}, valid_bet_chip:{self.valid_bet_chip}')
        print('-----------------玩家个人信息------------------')
        print(str(self.players).replace('[', '').replace('\n, ', '\n').replace('\n]', ''))
        print('-----------------合法的动作--------------------')
        print(f'| check | call | fold | allin |  raise chip |')
        legal_actions = self.get_legal_actions()
        print(f'|   {legal_actions[0]}   |  {legal_actions[1]}   |  {legal_actions[2]}   '
              f'|   {legal_actions[3]}   |     {legal_actions[4]:5}   |')
        print('---------------------------------------------', end='\n')


class Player:
    """
    Player属于被动方，其成员变量的修改是由Game类导致的，且不进行动作合法性、游戏状态的判断
    在遍历所有玩家时，首先需要判断玩家是否为active状态
    只有玩家是active状态时，对玩家成员变量的修改才有效
    均通过函数实现对成员变量的修改，不可直接修改成员变量的值
    """

    def __init__(self, player_idx):
        self.player_idx = player_idx  # 玩家id，从0开始，对应players中的索引
        self.identity = constants.ORDINARY
        self.player_state = constants.player_active  # 玩家当前的状态（活跃、弃牌、全押）
        self.game_total_chip = 0  # 本局累计下注量
        self.cur_stage_chip = 0  # 当前阶段下注量
        self.total_earn_chip = 0  # 累计赢得的筹码，每局结束时更新

    def init_player(self, identity):
        """
        每局游戏开始需要调用此函数
        """
        self.identity = identity
        self.player_state = constants.player_active  # 玩家当前的状态（活跃、弃牌、全押）
        if identity == constants.SMALLBLIND:
            self.game_total_chip = self.cur_stage_chip = constants.SMALLBLIND_CHIP
        elif identity == constants.BIGBLIND:
            self.game_total_chip = self.cur_stage_chip = constants.BIGBLIND_CHIP
        else:
            self.game_total_chip = self.cur_stage_chip = 0

    def enter_next_stage(self):
        self.cur_stage_chip = 0

    def take_action(self, action):
        """
        玩家采取的动作，只涉及对玩家自己筹码量的修改，call指令需要给出具体筹码量
        """
        if action[0] == constants.CHECK_ACTION:
            self.check()
        elif action[0] == constants.CALL_ACTION:
            self.call(action[1])
        elif action[0] == constants.ALLIN_ACTION:
            self.allin()
        elif action[0] == constants.FOLD_ACTION:
            self.fold()
        elif action[0] == constants.RAISE_ACTION:
            self.raise_bet(action[1])
        else:
            card_util.print_exception(self.take_action, '未知的动作类型(unknown action)')

    def check(self):
        # check动作没有需要改变的量
        # self.cur_stage_chip = 0
        pass

    def call(self, call_chip):
        """
        call动作
        :param call_chip: call的筹码量
        """
        last_chip = self.cur_stage_chip
        self.cur_stage_chip = call_chip
        self.game_total_chip += (call_chip - last_chip)

    def fold(self):
        self.player_state = constants.player_folded

    def raise_bet(self, raise_chip):
        """
        下注动作（raise chip）
        :return:
        """
        last_chip = self.cur_stage_chip
        self.cur_stage_chip = raise_chip
        self.game_total_chip += (raise_chip - last_chip)

    def allin(self):
        # todo 下注量是否需要改变
        self.player_state = constants.player_folded

    def game_over(self, earn_chip):
        """
        统计输赢筹码
        :param earn_chip: 输（负）或赢（正）的筹码量
        :return:
        """
        self.total_earn_chip += earn_chip

    def __str__(self):
        return f'player_idx: {self.player_idx}, identity: {self.identity}, cur_player_state: ' \
               f'{self.player_state}, cur_stage_chip: {self.cur_stage_chip:5}, game_total_chip: ' \
               f'{self.game_total_chip:5}, total_earn_chip: {self.total_earn_chip: }\n'

    def __repr__(self):
        return self.__str__()


def test_act(game: NoLimitHoldemGame, act):
    print('做出的动作', act)
    state, is_legal, game_state_flag, down, info = game.step(act)
    print('下一状态')
    # game.print_info()
    print(state)
    print('is legal:', is_legal, 'game_state_flag:', game_state_flag, 'info:', info)
    print('\n\n')


# noinspection DuplicatedCode
def test1(game=NoLimitHoldemGame()):
    game.prepare_round()
    game.print_info()
    print(game.get_legal_actions())
    # preflop
    chips = np.random.randint(200, 2000, size=5)
    print('preflop')
    act = np.array([constants.CALL_ACTION, 100])
    test_act(game, act)
    act = np.array([constants.RAISE_ACTION, chips[0]])
    test_act(game, act)
    act = np.array([constants.CALL_ACTION, 100])
    test_act(game, act)
    print('flop')
    act = np.array([constants.RAISE_ACTION, chips[1]])
    test_act(game, act)
    act = np.array([constants.CALL_ACTION, 100])
    test_act(game, act)
    # flop
    print('turn')
    act = np.array([constants.RAISE_ACTION, chips[2]])
    test_act(game, act)
    act = np.array([constants.CALL_ACTION, 100])
    test_act(game, act)
    # turn
    print('river')
    act = np.array([constants.RAISE_ACTION, chips[3]])
    test_act(game, act)
    act = np.array([constants.CALL_ACTION, 100])
    test_act(game, act)


if __name__ == '__main__':
    game1 = NoLimitHoldemGame()
    for _ in range(300):
        test1(game1)
