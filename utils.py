import  torch
from    torch import nn
from    torch.nn import functional as F

import warnings
import os
import numpy as np
import csv
from matplotlib import pyplot as plt

def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc



def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res


class CountRecorder():
    def __init__(self):
        self.now_count=0
        
    def do_count(self):
        self.now_count+=1
    
    def get_count(self):
        return self.now_count


class MaxCountRecorder(CountRecorder):

    def __init__(self,number_of_maximal_count):
        if number_of_maximal_count:
            self.maximal=number_of_maximal_count
        else:
            self.maximal=np.float('inf')
        self.now_count=0


    def check_then_count(self):
        if self.now_count>self.maximal:
            return False
        self.now_count+=1
        return True


class simple_ploter():

    def __init__(self,save_name='simple_plot'):
        self.countRecorder=CountRecorder
        self.save_name=save_name
        self.x_list=[]
        self.y_list=[]

    def record_data(self,y,x=None):
        if x==None:
            self.countRecorder.do_count()  # add 1 to the count recorder
            x=self.countRecorder.get_count()
        self.x_list.append(x)
        self.y_list.append(y)

    def show_the_plot_and_save(self):
        
        import os
        path=os.getcwd()
        path=os.path.abspath(path)
        os.chdir(path)
        plt.rcParams['savefig.dpi'] = 500
        plt.title(self.save_name)
        plt.plot(self.x_list,self.y_list)
        #plt.show()
        plt.savefig('./{}.jpg'.format(self.save_name))
        plt.cla()





def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    #raise Exception
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item(),out


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc
