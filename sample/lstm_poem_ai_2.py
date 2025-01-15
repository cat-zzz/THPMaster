"""
@project: THPMaster
@File   : lstm_poem_ai_2.py.py
@Desc   :
@Author : gql
@Date   : 2024/8/28 17:32
"""
import torch
import os
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from tqdm.notebook import tqdm
from tqdm import tqdm


class DictObj(object):
    # 私有变量是map
    # 设置变量的时候 初始化设置map
    def __init__(self, mp):
        self.map = mp
        # print(mp)

    # set 可以省略 如果直接初始化设置
    def __setattr__(self, name, value):
        if name == 'map':  # 初始化的设置 走默认的方法
            # print("init set attr", name ,"value:", value)
            object.__setattr__(self, name, value)
            return
        # print('set attr called ', name, value)
        self.map[name] = value

    # 之所以自己新建一个类就是为了能够实现直接调用名字的功能。
    def __getattr__(self, name):
        # print('get attr called ', name)
        return self.map[name]


Config = DictObj({
    'poem_path': "./tang.npz",
    'tensorboard_path': './tensorboard/',
    'model_save_path': './modelDict2/',
    'embedding_dim': 128,
    'hidden_dim': 1024,
    'lr': 0.001,
    'LSTM_layers': 3
})


def view_data(poem_path):
    datas = np.load(poem_path, allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    word_data = np.zeros((1, data.shape[1]), dtype=str)  # 这样初始化后值会保留第一一个字符，所以输出中'<START>' 变成了'<'
    row = np.random.randint(data.shape[0])
    for col in range(data.shape[1]):
        word_data[0, col] = ix2word[data[row, col]]
    print(data.shape)  # (57580, 125)
    print(word_data)  # 随机查看


class PoemDataSet(Dataset):
    def __init__(self, poem_path, seq_len):
        self.seq_len = seq_len
        self.poem_path = poem_path
        self.poem_data, self.ix2word, self.word2ix = self.get_raw_data()
        self.no_space_data = self.filter_space()

    def __getitem__(self, idx: int):
        txt = self.no_space_data[idx * self.seq_len: (idx + 1) * self.seq_len]
        label = self.no_space_data[idx * self.seq_len + 1: (idx + 1) * self.seq_len + 1]  # 将窗口向后移动一个字符就是标签
        txt = torch.from_numpy(np.array(txt)).long()
        label = torch.from_numpy(np.array(label)).long()
        return txt, label

    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)

    def filter_space(self):  # 将空格的数据给过滤掉，并将原始数据平整到一维
        t_data = torch.from_numpy(self.poem_data).view(-1)
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if i != 8292:
                no_space_data.append(i)
        return no_space_data

    def get_raw_data(self):
        datas = np.load(self.poem_path, allow_pickle=True)  # numpy 1.16.2  以上引入了allow_pickle
        # datas = np.load(self.poem_path)
        data = datas['data']
        ix2word = datas['ix2word'].item()
        word2ix = datas['word2ix'].item()
        return data, ix2word, word2ix


poem_ds = PoemDataSet(Config.poem_path, 48)
ix2word = poem_ds.ix2word
word2ix = poem_ds.word2ix
poem_loader = DataLoader(poem_ds, batch_size=16, shuffle=True, num_workers=0)


class MyPoetryModel_tanh(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MyPoetryModel_tanh, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # vocab_size:就是ix2word这个字典的长度。
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                            batch_first=True, dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, vocab_size)

    #         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，

    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(Config.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))  # hidden 是h,和c 这两个隐状态
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# %%
# top k的准确率计算
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True)  # 使用topk来获得前k个的索引
    pred = pred.t()  # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred))  # 与正确标签序列形成的矩阵相比，生成True/False矩阵
    #     print(correct)

    rtn = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        correct_k = correct[:k].reshape(-1).float().sum(0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size))  # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn


def train(epochs, train_loader, device, model, criterion, optimizer, scheduler, tensorboard_path):
    model.train()
    top1 = AvgrageMeter()
    model = model.to(device)
    for epoch in range(epochs):
        print(f'第{epoch + 1}/{epochs}轮开始')
        train_loss = 0.0
        train_loader = tqdm(train_loader)
        train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
            #             print(' '.join(ix2word[inputs.view(-1)[k] for k in inputs.view(-1).shape.item()]))
            labels = labels.view(-1)  # 因为outputs经过平整，所以labels也要平整来对齐
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()
            outputs, hidden = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, pred = outputs.topk(1)
            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)

            #             break
            # ternsorboard 曲线绘制
            if os.path.exists(Config.tensorboard_path) == False:
                os.mkdir(Config.tensorboard_path)
            # writer = SummaryWriter(tensorboard_path)
            # writer.add_scalar('Train/Loss', loss.item(), epoch)
            # writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            # writer.flush()
        scheduler.step()
        # 模型保存
        if not os.path.exists(Config.model_save_path):
            os.mkdir(Config.model_save_path)
        if (epoch + 1) % 5 == 0:
            print(f'保存第{epoch + 1}轮模型')
            torch.save(model.state_dict(), Config.model_save_path + 'poem' + str(epoch + 1) + '.pth')
    print('训练完成')


def print_model_info(model):
    total_params = 0
    print("模型结构及参数量：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"层: {name}\t 形状: {param.shape}\t 参数量: {param_count}")

    print(f"\n模型的总参数量: {total_params}")


def func():
    model = MyPoetryModel_tanh(len(word2ix), embedding_dim=Config.embedding_dim, hidden_dim=Config.hidden_dim)
    print_model_info(model)


def run():
    print('加载数据集')
    # 上述参数的配置网络训练显存消耗为2395M，超过显存的话，重新调整下网络配置
    model = MyPoetryModel_tanh(len(word2ix),
                               embedding_dim=Config.embedding_dim,
                               hidden_dim=Config.hidden_dim)
    print_model_info(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 学习率调整
    criterion = nn.CrossEntropyLoss()

    # 因为使用tensorboard画图会产生很多日志文件，这里进行清空操作
    import shutil
    if os.path.exists(Config.tensorboard_path):
        shutil.rmtree(Config.tensorboard_path)
        os.mkdir(Config.tensorboard_path)
    train(epochs, poem_loader, device, model, criterion, optimizer, scheduler, Config.tensorboard_path)


if __name__ == '__main__':
    # run()
    func()
    pass
