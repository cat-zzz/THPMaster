"""
@File   : server.py
@Desc   : 程序入口
@Author : gql
@Date   : 2023/4/2 18:57
"""
import queue
import socket
import time

from competition.core import constants
from competition.core.entity import Player, Game
from competition.core.strategy.strategy_2 import Strategy
from competition.network.receive_forward_msg import ReceiveMsgThread, connect_server


# def play(s: socket.socket):
#     catzzz = Player("catzzz")
#     opponent = Player("opponent")
#     strategy = Strategy()
#     msg_queue = queue.Queue(0)
#     msg_thread = ReceiveMsgThread(s, msg_queue)
#     msg_thread.start()
#     for i in range(70):
#         print("---第{0}局开始---".format(i + 1))
#         game = Game(catzzz, opponent)
#         strategy.start_game(game)
#         # 单局游戏的循环
#         while True:
#             """
#             1 接受服务端指令
#             2 解析服务端指令并更新信息
#             3.1 如果需要我方做出动作：
#                 3.1.1 调用策略函数，生成客户端指令
#                 3.1.2 更新我方信息
#                 3.1.3 发送指令到服务端
#             3.2 否则：（表示此时我方不需要行动）
#                 3.2.1 当前对局在preflop阶段且我方是大盲注，则我方不需要先行动
#                 3.2.2 earnChips和oppo_hands指令表示一局游戏的结束
#             4 跳转到1
#             """
#             # 使用多线程+消息队列的形式，防止一次性接收到多条指令（一次只能解析一条指令）
#             # reply = s.recv(1024).decode()  # 第1步
#             reply = msg_queue.get(block=True)  # 第1步
#             print("handling reply: ", reply)
#             game_flag = game.parse_server_cmd(reply)  # 第2步
#             if game_flag == constants.take_action:  # 第3.1步
#                 # client_cmd = strategy.strategy(catzzz, opponent, game.stage, game.public_cards)  # 第3.1.1步 生成客户端指令
#                 # 第3.1.1步 生成客户端指令
#                 # client_cmd = strategy_by_hands_win_rate.strategy(catzzz, opponent, game.stage, game.public_cards)
#                 # client_cmd = strategy_by_win_rate.strategy(catzzz, opponent, game.stage, game.public_cards,
#                 #                                            game.total_chip)
#                 client_cmd = strategy.strategy()
#                 # 3.1.2 更新己方信息
#                 game.parse_client_cmd(client_cmd, catzzz, opponent)
#                 print("client send: ", client_cmd)
#                 time.sleep(1)
#                 s.sendall(client_cmd.encode("ASCII"))  # 3.1.3 发送指令
#             elif game.stage == constants.show_oppo_card:  # 亮手牌，根据对手手牌分析对手特征
#                 strategy.show_oppo_cards_strategy()
#             game.print_info()
#             # 3.2.2 earnChips指令和oppo_hands指令
#             print("game_flag:", game_flag)
#             if game_flag == constants.game_over:
#                 catzzz.reset()
#                 opponent.reset()
#                 break


def play_1(s: socket.socket):
    catzzz = Player("catzzz")
    opponent = Player("opponent")
    strategy = Strategy()
    msg_queue = queue.Queue(0)
    msg_thread = ReceiveMsgThread(s, msg_queue)
    msg_thread.start()
    i = 0
    # for i in range(70):
    game = Game(catzzz, opponent)
    strategy.start_game(game)
    while True:
        reply = msg_queue.get(block=True)  # 第1步，接收服务端指令
        if "preflop" in reply:  # 一局刚开始（即收到preflop指令）
            print("---第{0}局开始---".format(i + 1))
            strategy.start_game(game)
            i += 1
        print("main线程-->正在处理指令: ", reply)
        game_flag, catzzz_action_flag, catzzz_first_action_flag = game.parse_server_cmd_1(reply)  # 解析服务端指令并更新实体类信息
        # game.print_info()
        if catzzz_action_flag == constants.take_action:  # 需要我方做出动作
            client_cmd = strategy.strategy_1(i, catzzz_first_action_flag)
            game.parse_client_cmd(client_cmd, catzzz, opponent)
            print("\n我方发送指令: ", client_cmd)
            # time.sleep(1)
            s.sendall(client_cmd.encode("ASCII"))  # 发送指令
            # game.print_info()


if __name__ == '__main__':
    # client_socket = connect_server("Thinking in THP", ip_port=("124.70.157.73", 10001))
    # client_socket = connect_server("Thinking in THP", ip_port=("123.60.107.245", 10001))    # 弈思德扑常驻
    # client_socket = connect_server("Thinking in THP", ip_port=("60.204.144.239", 10001))
    # client_socket = connect_server("Thinking in THP", ip_port=("60.204.184.192", 10001))
    # client_socket = connect_server("Thinking in THP", ip_port=("124.70.157.73", 10001))
    # play(client_socket)
    # client_socket = connect_server("Thinking in THP", ip_port=("119.45.103.129", 10001))  # 腾讯会议号405-7261-4519，常驻骑士德州扑克 1 队
    # client_socket = connect_server("Thinking in THP", ip_port=("119.45.237.113", 10001))  # 494-9758-7129，弈翔_德州扑克一队
    client_socket = connect_server("Thinking in THP", ip_port=("1.13.175.56", 10001))  # 315-7857-6471，星光德州扑克

    # client_socket = connect_server("Thinking in THP", ip_port=("127.0.0.1", 10001))
    # client_socket = connect_server("Thinking in THP2", ip_port=("127.0.0.1", 10001))
    play_1(client_socket)
    print("运行结束")
