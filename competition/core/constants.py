"""
@File   : constants
@Desc   : 定义常量文件
@Author : gql
@Date   : 2023/2/2 21:36
"""

# 单局总筹码量
total_chip = 20000
# 玩家行为
player_raise_action = 201
player_call_action = 202
player_check_action = 203
player_allin_action = 204
player_fold_action = 205
player_unknown_action = 206  # 未知行为（错误行为）
player_no_required_action = 207     # 无需采取行动

# 游戏阶段
init_stage = "init"  # 游戏初始阶段，还没有进入到preflop阶段
preflop_stage = "preflop"
flop_stage = "flop"
turn_stage = "turn"
river_stage = "river"  # 河牌阶段
earn_chip_stage = "earnChips"  # 赢得筹码量
show_oppo_card = "show_oppo_hands"  # 展示对手的手牌的阶段

# 玩家身份
SMALLBLIND = "SMALLBLIND"
BIGBLIND = "BIGBLIND"

# parse_server_cmd()函数的返回值表示
take_action = 301
no_take_action = 302
game_over = 303
game_start = 304  # 一局游戏开始
game_keeping = 305  # 一局游戏正在进行中

# 牌型大小（用于编码）
royal = 10

# 玩家状态（是否进行本局对弈）
PLAY_THIS = 401
NOT_PLAY_THIS = 402
UNCERTAIN_PLAY_THIS = 403  # 需要判断是否进行对弈，当我方第一次行动时，需要判断是否继续对弈
