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
from src.util.model_utils import print_model_info


class ThpModel(nn.Module):
    def __init__(self, card_in_channels, env_features_dim, action_input_dim, action_lstm_hidden_dim, actor_output_dim,
                 model_config=None):
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
        # todo 是否在Attention之后添加一层全连接层
        # 常用的设置：attention_input_dim=512, heads=8
        self.card_env_attn_layer = nn.MultiheadAttention(
            embed_dim=self.model_config['card_env_attention_embed_dim'],
            num_heads=self.model_config['card_env_attention_num_heads'])
        self.action_attn_layer = nn.MultiheadAttention(
            embed_dim=self.model_config['action_attention_embed_dim'],
            num_heads=self.model_config['action_attention_num_heads'])
        # QKV的seq length和embed dim可以均不相等，其中K和V的embed dim可以通过kdim和vdim指定；seq length本身就可以不相等
        # CrossAttention的输出形状为[129, 16, 52]，其中129为card_conv2_channels+一个环境特征，52是将牌面信息展平为一维
        self.cross_attn_layer = nn.MultiheadAttention(embed_dim=self.model_config['card_env_attention_embed_dim'],
                                                      num_heads=self.model_config['cross_attention_num_heads'],
                                                      kdim=self.model_config['action_attention_embed_dim'],
                                                      vdim=self.model_config['action_attention_embed_dim'])
        # 最后的全连接层
        self.final_fc_layer = nn.Sequential(
            nn.Linear(
                (self.model_config['card_conv2_channels'] + 1) * self.model_config['card_env_attention_embed_dim'],
                self.model_config['final_fc1_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(self.model_config['final_fc1_dim'], self.model_config['final_fc2_dim']),
            nn.ReLU(inplace=True),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(self.model_config['final_fc2_dim'], self.model_config['value_fc1_dim']),
            nn.Linear(self.model_config['value_fc1_dim'], self.model_config['value_output_dim'])
        )
        self.actor_layer = nn.Sequential(
            nn.Linear(self.model_config['final_fc2_dim'], self.model_config['actor_fc1_dim']),
            nn.Linear(self.model_config['actor_fc1_dim'], actor_output_dim)
        )

    def forward(self, **kwargs):
        """
        :param kwargs: 用字典存储模型的多个输入
        :return: 模型输出
        """
        '''
        原始输入
        card input: [16, 5, 4, 13], 16表示batch size, 5表示通道数
        env input: [16, 13]
        action_input: [16, 24, 18], 24表示序列长度，18表示动作特征维度

        中间网络结构的输出形状
        card_conv_layer: [16, 128, 4, 13]-->(将[4,13]部分展平为1维)-->[16, 128, 52]
        env_fc_layer: [16, 52]-->(拓展成3维)-->[16, 1, 52]
        拼接card和env: [129, 16, 52], 将batch size调换到第2维了
        card_env_attn_layer: [129, 16, 52], 注意力层的输出形状与Query的形状一致
        action_lstm_layer: [16, 24, 1024], 24为序列长度，16为batch size，1024为隐藏层维度
        action维度转换: [24, 16, 1024]
        action_attn_layer: [24, 16, 1024]
        cross_attn_layer: [129, 16, 52], Q是card_env_attn_layer的输出，K和V是action_attn_layer的输出
        
        actor和value部分
        value: 两个全连接层，输入为[16, 2048]，输出为[16, 1]
        actor: 两个全连接层，输入为[16, 2048]，输出为[16, 18]，18是actor_output_dim的值
        '''
        card_input = kwargs['card_input']
        env_input = kwargs['env_input']
        action_input = kwargs['action_input']
        card_output = self.card_conv_layer(card_input)
        env_output = self.env_fc_layer(env_input)
        card_output = card_output.view(card_output.shape[0], card_output.shape[1], -1)  # 将牌面信息展平为一维
        env_output = env_output.unsqueeze(1)
        card_env_combined = torch.cat([card_output, env_output], dim=1)
        card_env_combined = card_env_combined.permute(1, 0, 2)
        card_env_attn_output, _ = self.card_env_attn_layer(query=card_env_combined,
                                                           key=card_env_combined, value=card_env_combined)
        action_lstm_output, (_, cn) = self.action_lstm_layer(action_input)
        action_lstm_output = action_lstm_output.permute(1, 0, 2)
        action_attn_output, _ = self.action_attn_layer(query=action_lstm_output,
                                                       key=action_lstm_output, value=action_lstm_output)
        cross_attn_output, _ = self.cross_attn_layer(query=card_env_attn_output,
                                                     key=action_attn_output, value=action_attn_output)
        cross_attn_output = (cross_attn_output.permute(1, 0, 2)
                             .reshape(card_env_attn_output.shape[1],
                                      card_env_attn_output.shape[0] * card_env_attn_output.shape[2]))
        final_fc_output = self.final_fc_layer(cross_attn_output)
        value = self.value_layer(final_fc_output)
        actor = self.actor_layer(final_fc_output)
        print('value shape:', value.shape)
        print('actor shape:', actor.shape)


# 仅定义网络内部结构的维度，不包括输入输出维度
default_model_config = {
    # 牌面信息的残差卷积层的参数
    'card_conv1_channels': 128,
    'card_conv2_channels': 128,
    # 环境状态信息的全连接层的参数
    'env_fc1_dim': 4096,
    'env_fc2_dim': 2048,
    'env_fc3_dim': 52,
    # 动作信息的LSTM层的参数
    'action_lstm_layers': 3,
    'action_fc1_dim': 2048,
    'action_fc2_dim': 2048,
    # 牌面和环境特征的attention层的参数
    'card_env_attention_embed_dim': 52,
    'card_env_attention_num_heads': 4,
    # 动作序列的attention层的参数
    'action_attention_embed_dim': 1024,  # 应与cross_attention_embed_dim一致
    'action_attention_num_heads': 4,
    # 'cross_attention_embed_dim': 1024,  # 应与action_attention_embed_dim一致
    'cross_attention_num_heads': 4,
    # 最后全连接层的参数
    'final_fc1_dim': 4096,
    'final_fc2_dim': 2048,
    'value_fc1_dim': 2048,
    'value_output_dim': 1,
    'actor_fc1_dim': 2048,
    # 'action_output_dim': 18
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
    'actor_output_dim': 18,
    'card_env_flatten_dim': default_model_config['card_conv2_channels'] * 4 * 13 + default_model_config['env_fc3_dim'],
}
if __name__ == '__main__':
    card_input1 = torch.randn(default_hyper_args['batch_size'], default_hyper_args['card_in_channels'],
                              default_hyper_args['card_matrix_rows'], default_hyper_args['card_matrix_cols'])
    env_input1 = torch.randn(default_hyper_args['batch_size'], default_hyper_args['env_features_dim'])
    # LSTM input: (batch_size, sequence_length, action_input_dim)
    action_input1 = torch.randn(default_hyper_args['batch_size'], default_hyper_args['action_input_seq'],
                                default_hyper_args['action_input_dim'])
    model1 = ThpModel(default_hyper_args['card_in_channels'], default_hyper_args['env_features_dim'],
                      default_hyper_args['action_input_dim'], default_hyper_args['action_lstm_hidden_dim'],
                      default_hyper_args['actor_output_dim'])
    model1(card_input=card_input1, env_input=env_input1, action_input=action_input1)
    print_model_info(model1)
