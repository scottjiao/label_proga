import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F

import  numpy as np
from    data import few_labels, load_data, preprocess_features, preprocess_adj, sparse_to_tuple, json_data_io, load_ogb_data, sample_mask
from    model import GCN
from    config import  args
from    utils import masked_loss, masked_acc, simple_ploter ,adjust_learning_rate, test, train
import  warnings
import os
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import  scipy.sparse as sp


warnings.filterwarnings("ignore", category=UserWarning)

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        #assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')









if __name__=='__main__':
        #ploter initialization
    args.with_psuedo_loss  =   eval(args.with_psuedo_loss)
    args.standard_split  =   eval(args.standard_split)
    args.feature_normalize  =   eval(args.feature_normalize)
    args.hidden_list  =   eval(args.hidden_list)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    #args.device=device
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    test_idx = split_idx['test'].to(device)
    valid_idx = split_idx['valid'].to(device)
    number_samples,number_class=(dataset.data.num_nodes[0] ,dataset.num_classes)

    train_mask=torch.from_numpy( sample_mask(train_idx.cpu().numpy(), number_samples)).to(device)
    val_mask = torch.from_numpy(sample_mask(valid_idx.cpu().numpy(), number_samples)).to(device)
    test_mask = torch.from_numpy(sample_mask(test_idx.cpu().numpy(), number_samples)).to(device)



    num_layers=len(args.hidden_list)+2
    hidden_channels=args.hidden_list[0]
    model = GCN(data.num_features, hidden_channels,
                     dataset.num_classes, num_layers,
                     args.dropout).to(device)
    evaluator = Evaluator(name='ogbn-arxiv')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logger = Logger(10, args)

    
    test_ploter=simple_ploter(save_name='test_of_{}'.format(args.exp_id))
    train_ploter=simple_ploter(save_name='train_of_{}'.format(args.exp_id))
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, data, train_idx, optimizer,train_mask,val_mask,test_mask,device,args)
        result = test(model, data, split_idx, evaluator)
        logger.add_result(int(args.exp_times  ), result)

        if epoch % 1 == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {int(args.exp_times) + 1}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')


        test_ploter.record_data(test_acc,x=epoch)
        train_ploter.record_data(train_acc,x=epoch)
logger.print_statistics(int(args.exp_times  ))
meta_results_path=os.path.join('.','results',args.exp_id)   #./results/exp_id

results_path=os.path.join(meta_results_path,str(args.exp_times))    #./results/exp_id/exp_times

for path in [meta_results_path,results_path]:
    if not os.path.exists(path):
        os.mkdir(path)




results_log_path= os.path.join(results_path,'log.txt')
results_data_path=os.path.join(results_path,'data')
json_io=json_data_io(file_name=results_data_path)
json_io.save({'test_x_list':test_ploter.x_list,'test_y_list':test_ploter.y_list,'args':vars(args)})


test_ploter.save_name=os.path.join('results',args.exp_id,str(args.exp_times),'test_data')
test_ploter.show_the_plot_and_save()

train_ploter.save_name=os.path.join('results',args.exp_id,str(args.exp_times),'train_data')
train_ploter.show_the_plot_and_save()
        