import os

import  numpy as np
from    model import GCN
from data import json_data_io
import  warnings
import sys
from subprocess import run
import argparse
from matplotlib import pyplot as plt

args = argparse.ArgumentParser()
#args.add_argument('--dataset', default='cora')
#args.add_argument('--dataset', default='citeseer')
#args.add_argument('--dataset', default='pubmed')
args.add_argument('--dataset', default='ogbn-arxiv')
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', type=float, default=1e-3)    # 1e-2 for coras
args.add_argument('--epochs', type=int, default=1000)
#args.add_argument('--hidden', type=int, default=128)
args.add_argument('--dropout', type=float, default=0.5)     #0.8 for coras
args.add_argument('--weight_decay', type=float, default=0)   # 1e-3 for coras
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--max_degree', type=int, default=3)
args.add_argument('--few_label_seed', type=int, default=1)
args.add_argument('--few_label_number', type=int, default=10)
args.add_argument('--confidence_threshold', type=float, default=0.75)
args.add_argument('--feature_normalize', default='True')
args.add_argument('--with_psuedo_loss', default='False')
args.add_argument('--standard_split', default='True')
args.add_argument('--hidden_list', type=str, default='[256,256,256]')


args.add_argument('--exp_id', default='arxiv_std_gcn_0')
args.add_argument('--repeat_times', type=int, default=10)

args = args.parse_args()

print(vars(args))

y_list=None
x_list=None

total_results_path=os.path.join('.','results')
if not os.path.exists(total_results_path):
    os.mkdir(total_results_path)

meta_results_path=os.path.join('.','results',args.exp_id)   #./results/exp_id
args_ioer=json_data_io(file_name=os.path.join(meta_results_path,'data'))


for i in range(args.repeat_times):
    
    run('python train.py --exp_id {} --exp_times {} --few_label_seed {} --dataset {} --model {} --learning_rate {} --epochs {} --dropout {} --weight_decay {} --early_stopping {} --max_degree {}  --few_label_number {} --confidence_threshold {} --feature_normalize {} --with_psuedo_loss {} --standard_split {} --hidden_list {}'.format(args.exp_id,i,i, args.dataset, args.model, args.learning_rate, args.epochs, args.dropout, args.weight_decay, args.early_stopping, args.max_degree, args.few_label_number, args.confidence_threshold, args.feature_normalize,args.with_psuedo_loss,args.standard_split, args.hidden_list),shell=True)

    # collect data
    results_path=os.path.join(meta_results_path,str(i))    #./results/exp_id/exp_times
    temp_ioer=json_data_io(file_name=os.path.join(results_path,'data'))
    temp_data=temp_ioer.load()
    if y_list==None:
        y_list=[[j] for j in temp_data['y_list']]
        x_list=temp_data['x_list']
    for j in range(len(y_list)):
        y_list[j].append(temp_data['y_list'][j])


# get statistics

statistics_mean=[]
statistics_var=[]
statistics_std=[]
error_limit=[]

plt.title('{} of label {}'.format(args.dataset,args.few_label_number))
for i in range(len(y_list)):
    statistics_mean.append(np.mean(y_list[i]))
    statistics_var.append(np.var(y_list[i]))
    statistics_std.append(np.std(y_list[i]))
    error_limit.append([statistics_std[-1] ])
for exp_cut in range(len(y_list[0])):
    plt.scatter(x_list,[epoch_cut[exp_cut] for epoch_cut in  y_list])
plt.errorbar(x_list, statistics_mean, yerr=error_limit, fmt=":o", ecolor="y", elinewidth=4,
             ms=5, mfc="c", mec="r", capsize=7, capthick=8)
plt.savefig(os.path.join(meta_results_path,'error_plot.jpg'))
plt.cla()


#record statistics

args_ioer.save({'args':vars(args),'mean':statistics_mean[-1],'std_error':statistics_std[-1]})














