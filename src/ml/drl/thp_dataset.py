"""
@project: THPMaster
@File   : thp_dataset.py
@Desc   :
@Author : gql
@Date   : 2024/7/12 20:54
"""
from torch.utils.data import Dataset

'''
状态分为三个部分：牌面信息，动作序列和环境状态信息
'''


class ThpDataset(Dataset):
    def __init__(self, directory):
        pass

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        pass


if __name__ == '__main__':
    pass
