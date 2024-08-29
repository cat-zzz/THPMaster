"""
@project: THPMaster
@File   : thp_model.py
@Desc   :
@Author : gql
@Date   : 2024/7/13 13:21
"""
from torch import nn

from src.ml.nnblock.residual_conv import ResidualConvBlock

# 仅定义网络内部结构的维度，不包括输入输出维度
default_model_config = {
    # 牌面信息的残差卷积层的参数
    'card_conv1_channels': 128,
    'card_conv2_channels': 128,
    # 环境状态信息的全连接层的参数
    'env_fc1_dim': 1024,
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


class ThpModel(nn.Module):
    def __init__(self, card_in_channels, env_features_dim, action_input_dim, action_lstm_hidden_dim, model_config=None):
        super().__init__()
        if model_config is None:
            self.model_config = default_model_config
        else:
            self.model_config = model_config
        # 牌面信息--> 残差卷积层
        self.card_conv_layer = nn.Sequential(
            ResidualConvBlock(card_in_channels, self.model_config['card_conv1_channels']),
            ResidualConvBlock(self.model_config['card_conv1_channels'], self.model_config['card_conv2_channels']),
        )
        # 动作信息--> LSTM->FC
        self.action_lstm_layer = nn.Sequential(
            # todo 是否需要嵌入层，以及需要确定输入维度的大小
            # nn.Embedding(action_input_dim, embedding_dim),
            nn.LSTM(action_input_dim, action_lstm_hidden_dim, num_layers=self.model_config['action_lstm_layers'],
                    batch_first=True, dropout=0, bidirectional=False),
            # todo 是否在LSTM之后添加全连接层
        )
        # 环境状态特征--> 全连接层
        self.env_fc_layer = nn.Sequential(
            nn.Linear(env_features_dim, self.model_config['env_fc1_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_config['env_fc1_dim'], self.model_config['env_fc2_dim']),
            nn.ReLU(inplace=True),
        )
        # 常见的设置attention_input_dim=512, heads=8
        self.card_env_attention_layer = nn.Sequential(
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
            nn.Linear(根据前面的参数确定, self.model_config['overall_fc1_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(env_features_dim, self.model_config['overall_fc2_dim']),
            nn.ReLU(inplace=True),
        )
        # todo value和actor都单独需要全连接层

    def forward(self):
        pass


if __name__ == '__main__':
    pass
