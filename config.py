import  argparse

args = argparse.ArgumentParser()
#args.add_argument('--dataset', default='cora')
#args.add_argument('--dataset', default='citeseer')
#args.add_argument('--dataset', default='pubmed')
args.add_argument('--dataset', default='ogbn-arxiv')
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', type=float, default=1e-3)    # 1e-2 for cora
args.add_argument('--epochs', type=int, default=500)
args.add_argument('--hidden', type=int, default=64)       #64 for cora
args.add_argument('--dropout', type=float, default=0.8)
args.add_argument('--weight_decay', type=float, default=1e-3)   # 1e-3 for cora
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--max_degree', type=int, default=3)
args.add_argument('--few_label_seed', type=int, default=1)
args.add_argument('--few_label_number', type=int, default=10)
args.add_argument('--confidence_threshold', type=float, default=0.75)
args.add_argument('--feature_normalize', type=bool, default=True)
args.add_argument('--with_psuedo_loss', type=bool, default=False)
args.add_argument('--standard_split', type=bool, default=False)


args.add_argument('--exp_id', type=str, default='arxiv100')
args.add_argument('--exp_times', type=str, default='0')


args = args.parse_args()
print(args)