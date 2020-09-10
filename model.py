import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution
from collections import OrderedDict

#from    config import args

class GCN(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero,input_sparse,hidden_list,args):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)


        dim_list=hidden_list
        dim_list.append(output_dim)
        dim_list.insert(0,input_dim)



        sequence_list=[]
        sequence_list.append(('conv0', GraphConvolution(dim_list[0], dim_list[1],           num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=input_sparse) ))
        for i in range(len(dim_list)-2):
            sequence_list.append(('conv{}'.format(i+1), GraphConvolution(dim_list[i+1], dim_list[i+2], num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False) ))
        
        self.layers = nn.Sequential(OrderedDict(sequence_list))


        '''self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=input_sparse),

                                    GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    )'''

    def forward(self, inputs):
        x, support = inputs

        x = self.layers((x, support))

        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss
