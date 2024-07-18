"""
@File   : constant.py
@Desc   : 存放常量
@Author : gql
@Date   : 2023/8/21 22:05
"""

SMALLBLIND_CHIP = 50
BIGBLIND_CHIP = 100
TOTAL_CHIP = 20000
smallblind_chip = 50
bigblind_chip = 100
min_chip = 100  # 在flop、river、turn阶段，最小加注量为100，此处处为1/2
total_chip = 20000
# CHECK_ACTION=101
# 动作空间
# N_ACTIONS = 14
CHECK_ACTION = 101
CALL_ACTION = 102
FOLD_ACTION = 103
ALLIN_ACTION = 104
RAISE_ACTION = 105
# 从106到120都可以用于表示raise行为
# RAISE_ONE_THIRD_POT_ACTION = 106  # 1/3底池的加注
# RAISE_HALF_POT_ACTION = 107  # 1/2底池的加注
# RAISE_TWO_THIRDS_POT_ACTION = 108  # 2/3底池的加注
# RAISE_POT_ACTION = 109
# RAISE_ONE_AND_HALF_ACTION = 110
# RAISE_2POT_ACTION = 111
# RAISE_3POT_ACTION = 112
# RAISE_4POT_ACTION = 113
# RAISE_5POT_ACTION = 114
# RAISE_6POT_ACTION = 115
# EARN_CHIP_ACTION = 121

# 动作空间与底池大小比率的映射关系
# action_pot_rate_table = {RAISE_ONE_THIRD_POT_ACTION: 1 / 3,
#                          RAISE_HALF_POT_ACTION: 1 / 2,
#                          RAISE_TWO_THIRDS_POT_ACTION: 2 / 3,
#                          RAISE_POT_ACTION: 1,
#                          RAISE_2POT_ACTION: 2,
#                          RAISE_3POT_ACTION: 3
#                          }

# 身份信息
SMALLBLIND = 201
BIGBLIND = 202
ORDINARY = 203  # 除了大小盲注之外的身份
BUTTON = 204  # 庄家位

# 阶段信息
preflop_stage = 301
flop_stage = 302
turn_stage = 303
river_stage = 304
earn_chip_stage = 305

# 函数NoLimitHoldemGame.take_player_action()的返回值
func_enter_next_stage = 401
func_not_enter_next_stage = 402
func_direct_enter_earn_chip_stage = 403
func_allin_enter_earn_chip_stage = 404
func_game_over = 405
func_game_going = 406

# 玩家当前状态（是否弃牌、Allin），主要用于多人德扑
player_active = 501
player_folded = 502
player_allin = 503
# 以下状态暂未使用
player_waiting = 504  # 等待状态，未轮到该玩家行动
player_showdown = 505  # 亮手牌
player_finished = 506  # 游戏已结束
