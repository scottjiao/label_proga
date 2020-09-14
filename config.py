import  argparse

args = argparse.ArgumentParser()

#args.add_argument('--dataset', default='cora')
#args.add_argument('--dataset', default='citeseer')
#args.add_argument('--dataset', default='pubmed')
args.add_argument('--dataset', default='ogbn-arxiv')
args.add_argument('--model', default='gcn')
args.add_argument('--device', default='0')
args.add_argument('--epochs', type=int, default=1000)
args.add_argument('--early_stopping', type=int, default=10)
args.add_argument('--max_degree', type=int, default=3)
args.add_argument('--few_label_seed', type=int, default=1)
args.add_argument('--few_label_number', type=int, default=10)
args.add_argument('--confidence_threshold', type=float, default=0.9)
args.add_argument('--with_psuedo_loss', default='True')
args.add_argument('--standard_split', default='True')

args.add_argument('--feature_normalize', default='False')     #for arxiv
#args.add_argument('--feature_normalize', default='True')       #for coras

args.add_argument('--learning_rate', type=float, default=0.003)                  # 1e-2 for arxiv
#args.add_argument('--learning_rate', type=float, default=1e-2)                 # 1e-2 for coras
args.add_argument('--dropout', type=float, default=0.5)                         # 0.5  for arxiv
#args.add_argument('--dropout', type=float, default=0.8)                        # 0.8  for coras
args.add_argument('--weight_decay', type=float, default=0)                      # 0    for arxiv
#args.add_argument('--weight_decay', type=float, default=1e-3)                  # 1e-3 for coras
args.add_argument('--hidden_list', type=str, default='[256]')                   #      for arxiv
#args.add_argument('--hidden_list', type=str, default='[64]')                   #      for coras
args.add_argument('--bias', default='True')                                     #      for arxiv
#args.add_argument('--bias', default='False')                                    #      for coras


args.add_argument('--exp_id', type=str, default='arxiv_std_dsgcn_0')
args.add_argument('--exp_times', type=str, default='0')


args = args.parse_args()
print(args)