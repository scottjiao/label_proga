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





def train(model, data, train_idx, optimizer,train_mask,val_mask,test_mask,device,args):
    model.train()

    optimizer.zero_grad()
    log_softmax_all_out,softmax_all_out=model(data.x, data.adj_t)
    out_of_training_samples = log_softmax_all_out[train_idx]
    #raise Exception
    loss = F.nll_loss(out_of_training_samples, data.y.squeeze(1)[train_idx])

    
    if args.with_psuedo_loss:
        '''psuedo label loss'''
        train_label=softmax_all_out.argmax(dim=1)
        softmaxed_out=softmax_all_out
        with torch.no_grad():
            #get psuedo labes
            psuedo_labels= softmaxed_out.argmax(dim=1)
            #true label filtering
            psuedo_labels=psuedo_labels*(1-train_mask.int())+train_label*train_mask.int()
            #get confidence
            placeholder_1=psuedo_labels.unsqueeze(-1).to(device)
            one_hot_pred_labels=torch.zeros(softmax_all_out.shape).to(device).scatter_(1,placeholder_1,1)
            confidence=torch.max( torch.mul( softmax_all_out,one_hot_pred_labels) ,dim=1 )[0] 
            #threshold confidence
            confidence=torch.relu(confidence-args.confidence_threshold)
            #class eq
            class_eqer_counter=torch.sum(one_hot_pred_labels,dim=0)
            class_eqer_executor=torch.div(one_hot_pred_labels,(class_eqer_counter+1))
            confidence*=torch.max(class_eqer_executor,dim=1)[0]
            #raise Exception
        #label propagation
        #confidence=torch.sparse.mm(support,confidence.unsqueeze(-1)).reshape(-1)
        #add the self-training loss
        #loss+= masked_loss(all_out, psuedo_labels, confidence)  
        mask=confidence

        temp_loss=-torch.log(torch.max(softmax_all_out*one_hot_pred_labels,dim=1)[0])

        mask = mask.float()
        mask = mask / mask.mean()
        temp_loss *= mask
        temp_loss = temp_loss.mean()
        loss+=temp_loss





    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out,_ = model(data.x, data.adj_t)
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
