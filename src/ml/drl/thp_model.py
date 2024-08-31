"""
@project: THPMaster
@File   : thp_model.py
@Desc   :
@Author : gql
@Date   : 2024/7/13 13:21
"""
import torch
from torch import nn

from src.ml.nnblock.residual_conv import ResidualConvBlock


class ThpModel(nn.Module):
    def __init__(self, card_in_channels, env_features_dim, action_input_dim, action_lstm_hidden_dim,
                 card_env_flatten_dim, model_config=None):
        super().__init__()
        if model_config is None:
            self.model_config = default_model_config
        else:
            self.model_config = model_config
        # 牌面信息--> 残差卷积层
        self.card_conv_layer = nn.Sequential(
            ResidualConvBlock(card_in_channels, self.model_config['card_conv1_channels']),
            ResidualConvBlock(self.model_config['card_conv1_channels'], self.model_config['card_conv2_channels']),
            # nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # 动作信息--> LSTM->FC
        self.action_lstm_layer = nn.Sequential(
            # todo 是否需要嵌入层，以及需要确定输入维度的大小
            # nn.Embedding(action_input_dim, embedding_dim),
            nn.LSTM(input_size=action_input_dim, hidden_size=action_lstm_hidden_dim,
                    num_layers=self.model_config['action_lstm_layers'],
                    batch_first=True, dropout=0, bidirectional=False),
            # todo 是否在LSTM之后添加全连接层
        )
        # 环境状态特征--> 全连接层
        self.env_fc_layer = nn.Sequential(
            nn.Linear(env_features_dim, self.model_config['env_fc1_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_config['env_fc1_dim'], self.model_config['env_fc2_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_config['env_fc2_dim'], self.model_config['env_fc3_dim']),
            nn.ReLU(inplace=True),
        )
        # 常用的设置：attention_input_dim=512, heads=8
        self.card_env_attention_layer = nn.Sequential(
            nn.Linear(card_env_flatten_dim, self.model_config['action_attention_embed_dim']),
            nn.MultiheadAttention(embed_dim=self.model_config['card_env_attention_embed_dim'],
                                  num_heads=self.model_config['card_env_attention_num_heads']),
            # todo 是否在Attention之后添加一层全连接层
        )
        self.action_attention_layer = nn.Sequential(
            nn.MultiheadAttention(embed_dim=self.model_config['action_attention_embed_dim'],
                                  num_heads=self.model_config['action_attention_num_heads']),
            # todo 是否在Attention之后添加一层全连接层
        )
        # todo 通过MultiheadAttention实现CrossAttention

        # 最后的全连接层
        self.overall_fc_layer = nn.Sequential(
            nn.Linear(100, self.model_config['overall_fc1_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(env_features_dim, self.model_config['overall_fc2_dim']),
            nn.ReLU(inplace=True),
        )
        # todo value和actor都单独需要全连接层

    def forward(self, **kwargs):
        """
        :param kwargs: 用字典存储模型的多个输入
        :return: 模型输出
        """
        '''
        card_features --> 2~ResidualConv()
                                            (concat)--> MultiheadAttention()
        env_features  -->      2~FC()
        
        action_seq    -->      LSTM()      -->     MultiheadAttention()
        '''
        card_input = kwargs['card_input']
        env_input = kwargs['env_input']
        action_input = kwargs['action_input']
        card_output = self.card_conv_layer(card_input)
        print('card output shape:', card_output.shape)
        env_output = self.env_fc_layer(env_input)
        print(env_output.shape)
        action_output, (hn, cn) = self.action_lstm_layer(action_input)
        print(action_output.shape)
        action_last_output = action_output[:, -1, :]
        print(action_last_output.shape)
        card_output = torch.flatten(card_output, start_dim=1)
        print(card_output.shape)
        card_env_input = torch.cat((card_output, env_output), dim=1)
        print(card_env_input.shape)
        card_env_output = self.card_env_attention_layer(card_env_input, card_env_input, card_env_input)
        print('card_env_output shape:', card_env_output.shape)
        pass


# 仅定义网络内部结构的维度，不包括输入输出维度
default_model_config = {
    # 牌面信息的残差卷积层的参数
    'card_conv1_channels': 128,
    'card_conv2_channels': 64,
    # 环境状态信息的全连接层的参数
    'env_fc1_dim': 4096,
    'env_fc2_dim': 2048,
    'env_fc3_dim': 256,
    # 动作信息的LSTM层的参数
    'action_lstm_layers': 3,
    'action_fc1_dim': 2048,
    'action_fc2_dim': 2048,
    # 牌面和环境特征的attention层的参数
    'card_env_attention_embed_dim': 1024,
    'card_env_attention_num_heads': 4,
    # 动作序列的attention层的参数
    'action_attention_embed_dim': 1024,
    'action_attention_num_heads': 4,
    # 最后全连接层的参数
    'overall_fc1_dim': 1024,
    'overall_fc2_dim': 1024,
}
default_hyper_args = {
    'batch_size': 16,
    'card_in_channels': 5,
    'env_features_dim': 13,
    'action_input_dim': 18,
    'action_input_seq': 24,
    'action_lstm_hidden_dim': 1024,
    'card_matrix_rows': 4,
    'card_matrix_cols': 13,
    'card_env_flatten_dim': default_model_config['card_conv2_channels'] * 4 * 13 + default_model_config['env_fc3_dim'],
}
if __name__ == '__main__':
    card_input1 = torch.randn(default_hyper_args['batch_size'], default_hyper_args['card_in_channels'],
                              default_hyper_args['card_matrix_rows'], default_hyper_args['card_matrix_cols'])
    env_input1 = torch.randn(default_hyper_args['batch_size'], default_hyper_args['env_features_dim'])
    # LSTM input: (batch_size, sequence_length, action_input_dim)
    action_input1 = torch.randn(default_hyper_args['batch_size'], default_hyper_args['action_input_seq'],
                                default_hyper_args['action_input_dim'])
    model = ThpModel(default_hyper_args['card_in_channels'], default_hyper_args['env_features_dim'],
                     default_hyper_args['action_input_dim'],
                     default_hyper_args['action_lstm_hidden_dim'], default_hyper_args['card_env_flatten_dim'])
    model(card_input=card_input1, env_input=env_input1, action_input=action_input1)
    pass
