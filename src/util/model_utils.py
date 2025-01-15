"""
@project: THPMaster
@File   : model_utils.py
@Desc   :
@Author : gql
@Date   : 2024/9/3 11:44
"""
import os
from datetime import datetime

import torch
from torch import nn

from src.util.logger import logger


def print_model_info(model):
    total_params = 0
    s = ''
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            s += f'层:{name}  形状:{param.shape}  参数量:{param_count}  位于设备:{param.device}\n'
    logger.info(f'模型结构信息如下:\n{s}')
    logger.info(f"模型的总参数量: {total_params}")


def save_model(model: nn.Module, filepath):
    logger.info(f'保存模型({model.__class__.__name__})权重至{filepath}')
    torch.save(model.state_dict(), filepath)


def save_model_with_timestamp(model, dir_path, prefix='thpmodel'):
    """
    带时间戳保存模型
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(dir_path, f"{prefix}_{timestamp}.pth")
    torch.save(model.state_dict(), filepath)
    logger.info(f"{model.__class__}模型已保存: {filepath}")


# model = ThpModel(default_hyper_args['card_in_channels'], default_hyper_args['env_features_dim'],
#                  default_hyper_args['action_input_dim'], default_hyper_args['action_lstm_hidden_dim'],
#                  default_hyper_args['actor_output_dim'])

# save_model_with_timestamp(model, './model/')


def load_model(model: nn.Module, filepath):
    logger.info(f'从{filepath}加载模型({model.__class__.__name__})权重')
    model.load_state_dict(torch.load(filepath))


if __name__ == '__main__':
    pass
