"""
@project: THPMaster
@File   : lstm_poem_ai.py
@Desc   :
@Author : gql
@Date   : 2024/8/27 17:11
"""
import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from tqdm.notebook import tqdm
from tqdm import tqdm

from sample.lstm_poem_ai_2 import MyPoetryModel_tanh


class MyDict(object):
    def __init__(self, map):
        self.map = map

    # set 可以省略 如果直接初始化设置
    def __setattr__(self, name, value):
        if name == 'map':  # 初始化的设置 走默认的方法
            # print("init set attr", name ,"value:", value)
            object.__setattr__(self, name, value)
            return
        # print('set attr called ', name, value)
        self.map[name] = value

    def __getattr__(self, name):
        # 之所以自己新建一个类就是为了能够实现直接调用名字的功能
        # print('get attr called ', name)
        return self.map[name]


Config = MyDict({
    'poem_path': "./tang.npz",
    'tensorboard_path': './tensorboard/',
    'model_save_path': './modelDict/',
    # 'poem_path': "/mnt/data/tang.npz",
    # 'tensorboard_path': '/mnt/data/tensorboard/',
    # 'model_save_path': '/mnt/data/modelDict/',
    'embedding_dim': 128,
    'hidden_dim': 1024,
    'lr': 0.001,
    'LSTM_layers': 3
})


def view_data(poem_path):
    """

    :param poem_path:
    :return:
    """
    '''
    唐诗数据文件分为三部分，data部分是唐诗数据的总共包含57580首唐诗数据，其中每一首都被格式化成125个字符，
    唐诗开始用'<START\>'标志，结束用'<EOP\>'标志,空余的用'<space\>'标志，
    ix2word和word2ix是汉字的字典索引。因此可以不用自己去构建这个字典了。

    数据集中包含很多空格，如果不去除空格数据的话，模型的虽然最开始训练的时候就有60多的准确率，
    但是这些准确率是因为预测空格来造成的，所以需要将空格数据给去掉。
    '''
    datas = np.load(poem_path, allow_pickle=True)
    print(datas.files)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    word_data = np.zeros((1, data.shape[1]), dtype=str)  # 这样初始化后值会保留第一一个字符，所以输出中'<START>' 变成了'<'
    row = np.random.randint(data.shape[0])
    for col in range(data.shape[1]):
        word_data[0, col] = ix2word[data[row, col]]
    print(data.shape)  # 结果为(57580, 125)
    print(word_data)  # 随机查看
    print(data)


class PoemDataset(Dataset):
    def __init__(self, poem_path, seq_len):
        self.poem_path = poem_path
        self.seq_len = seq_len
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
        datas = np.load(self.poem_path, allow_pickle=True)
        data = datas['data']
        ix2word = datas['ix2word'].item()
        word2ix = datas['word2ix'].item()
        return data, ix2word, word2ix


class MyPoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MyPoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # vocab_size:就是ix2word这个字典的长度。
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                            batch_first=True, dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, vocab_size)

    def forward(self, input_data, hidden=None):
        embeds = self.embeddings(input_data)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = input_data.size()
        if hidden is None:
            h_0 = input_data.data.new(Config.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input_data.data.new(Config.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
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
        # self.reset()
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# top k的准确率计算
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True)  # 使用topk来获得前k个的索引
    pred = pred.t()  # 进行转置
    # eq按照对应元素进行比较view(1,-1) 自动转换到行为1,的形状，expand_as(pred)扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred))  # 与正确标签序列形成的矩阵相比，生成True/False矩阵
    rtn = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size))  # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn


def train(epochs, train_loader, device, model, criterion, optimizer, scheduler, tensorboard_path):
    model.train()
    top1 = AvgrageMeter()
    model = model.to(device)
    for epoch in range(epochs):
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
            #             print(get_word(pred))
            #             print(get_word(labels))
            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)
            #             break
            # TensorBoard 曲线绘制
            if not os.path.exists(Config.tensorboard_path):
                os.mkdir(Config.tensorboard_path)
            # writer = SummaryWriter(tensorboard_path)
            # writer.add_scalar('Train/Loss', loss.item(), epoch)
            # writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            # writer.flush()
        scheduler.step()
        if not os.path.exists(Config.model_save_path):
            os.mkdir(Config.model_save_path)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), Config.model_save_path + 'poem' + str(epoch) + '.pth')

    print('Finished Training')


def run(load_model_flag=False, load_model_path='/mnt/data/modelDict/poem0.pth'):
    print('加载数据集')
    poem_ds = PoemDataset(Config.poem_path, 48)
    ix2word = poem_ds.ix2word
    word2ix = poem_ds.word2ix
    print('加载模型')
    # 上述参数的配置网络训练显存消耗为2395M，超过显存的话，重新调整下网络配置
    model = MyPoetryModel(len(word2ix), embedding_dim=Config.embedding_dim, hidden_dim=Config.hidden_dim)
    if load_model_flag:
        model.load_state_dict(torch.load(load_model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调整
    criterion = nn.CrossEntropyLoss()

    # 使用tensorboard画图会产生很多日志文件，这里进行清空操作
    if os.path.exists(Config.tensorboard_path):
        shutil.rmtree(Config.tensorboard_path)
        os.mkdir(Config.tensorboard_path)
    poem_loader = DataLoader(poem_ds, batch_size=16, shuffle=True, num_workers=0)
    print('开始训练，调用train()函数')
    train(epochs, poem_loader, device, model, criterion, optimizer, scheduler, Config.tensorboard_path)


def generate(model, start_words, ix2word, word2ix, device):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input_tensor = torch.Tensor([word2ix['<START>']]).view(1, 1).long()

    # 最开始的隐状态初始为0矩阵
    hidden = torch.zeros((2, Config.LSTM_layers * 1, 1, Config.hidden_dim), dtype=torch.float)
    input_tensor = input_tensor.to(device)
    hidden = hidden.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(48):  # 诗的长度
            output, hidden = model(input_tensor, hidden)
            # 如果在给定的句首中，input为句首中的下一个字
            if i < start_words_len:
                w = results[i]
                input_tensor = input_tensor.data.new([word2ix[w]]).view(1, 1)
            # 否则将output作为下一个input进行
            else:
                top_index = output.data[0].topk(1)[1][0].item()  # 输出的预测的字
                w = ix2word[top_index]
                results.append(w)
                input_tensor = input_tensor.data.new([top_index]).view(1, 1)
            if w == '<EOP>':  # 输出了结束标志就退出
                del results[-1]
                break
    return results


def run_generate():
    poem_ds = PoemDataset(Config.poem_path, 48)
    ix2word = poem_ds.ix2word
    word2ix = poem_ds.word2ix
    model = MyPoetryModel_tanh(len(word2ix), embedding_dim=Config.embedding_dim, hidden_dim=Config.hidden_dim)
    model.load_state_dict(torch.load(Config.model_save_path+'poem100.pth'))  # 模型加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = generate(model, '雨', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '湖光秋月两相得，', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '人生得意须尽欢，', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '万里悲秋常作客，我言', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '风急天高猿啸哀，', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '千山鸟飞绝，', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '床前明月光，疑是地上霜。', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '天生我材必有用，', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '戍鼓断人行，', ix2word, word2ix, device)
    print(''.join(i for i in results))
    results = generate(model, '早知留酒待，', ix2word, word2ix, device)
    print(''.join(i for i in results))
    """
    引电随龙密又轻， '酒' '桮' '闲' '噀''得' '嘉' '名' '。'
    '千' '山' '草' '木' '如' '云' '暗' '，' '陆' '地' '波' '澜' '接' '海' '平' '。' 
    '洒' '竹' '几' '添' '春' '睡' '重' '，' '滴' '簷' '偏' '遣' '夜' '愁' '生' '。'
    '阴' '妖' '冷' '孽' '成' '何' '怪' '，' '敢' '蔽' '高' '天' '日' '月' '明' '。'
    """


if __name__ == '__main__':
    view_data(Config.poem_path)
    # run()
    run_generate()
    pass
