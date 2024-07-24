"""
@File   : common_util.py
@Desc   : 
@Author : gql
@Date   : 2023/7/20 17:18
"""


def print_exception(sender, msg):
    print("\033[0;31merror: \033[0m" + "func " + sender.__name__ + "() -> " + msg)
