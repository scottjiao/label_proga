import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution
from collections import OrderedDict

#from    config import args

class GCN(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero,input_sparse,hidden_list,args):
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)


        dim_list=hidden_list
        dim_list.append(output_dim)
        dim_list.insert(0,input_dim)
        self.dim_list=dim_list

        self.convs=nn.ModuleList()
        self.bns=nn.ModuleList()#trick from dgl

        self.convs.append(GraphConvolution(dim_list[0], dim_list[1],           num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=min(0.1,args.dropout),     #trick from dgl
                                                     is_sparse_inputs=input_sparse,
                                                     bias=args.bias) )
        self.bns.append(nn.BatchNorm1d(dim_list[1]))

        for i in range(len(dim_list)-2):
            if i<len(dim_list)-2:
                bn=nn.BatchNorm1d(dim_list[i+2])
                activation=F.relu
                dropout=args.dropout
            else:
                bn=nn.Identity() #trick from dgl
                activation=lambda x:x #trick from dgl
                dropout=0 #trick from dgl
            conv=GraphConvolution(dim_list[i+1], dim_list[i+2], num_features_nonzero,
                                                     activation=activation,
                                                     dropout=dropout,
                                                     is_sparse_inputs=False,
                                                     bias=args.bias) 
            self.convs.append(conv)
            self.bns.append(bn)
            
            


        


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

        #x = self.layers((x, support))

        for i in range(len(self.dim_list)-1):   
            #print('forward {} layer'.format(i))
            x,support=self.convs[i]((x,support))
            x=self.bns[i](x)



        return x,support

    '''def l2_loss(self):


        for p in self.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss'''
