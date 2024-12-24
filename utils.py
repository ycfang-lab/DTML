import os
import torch
import random
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


def time_counter(begin_time, end_time):
    run_time = round(end_time - begin_time)
    # 计算时分秒
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    # 输出
    return f'{hour:02d}:{minute:02d}:{second:02d}'

    
class InfiniteDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=False, collate_fn=None)

    def __iter__(self):
        return self.iter_function()

    def iter_function(self):
        while True:
            for batch in super().__iter__():
                if batch[-1].shape[0] != self.batch_size:
                    continue
                yield batch


def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def weights_init_normal(model:nn.Module):
    """
    args:
        model: model
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight, 0.0, 0.02)
        nn.init.constant_(model.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.constant_(model.bias, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.constant_(model.bias, 0.0)


class LamdaLR:
    """
    lr scheduler
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        """
        args:
            n_epochs: num epochs
            offset: start epoch
            decay_start_epoch: delayed start epoch
        """
        assert (n_epochs - decay_start_epoch) > 0,"Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class ReplayBuffer:
    """
    样本收集再采样
    """
    def __init__(self, max_size=50):
        """
        args:
            max_size: 最大容量
        """
        assert max_size > 0, "Empty buffer or or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        """
        args:
            data: 数据集
        """
        to_return=[]
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad