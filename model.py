import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution
from torch_geometric.nn import GCNConv
from collections import OrderedDict

#from    config import args

class GCN(nn.Module):


    def __init__(self, input_dim, output_dim,hidden_list,args):
        super(GCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)


        dim_list=hidden_list
        dim_list.append(output_dim)
        dim_list.insert(0,input_dim)
        self.dim_list=dim_list

        self.conv_layers=nn.ModuleList()
        self.bn_layers=nn.ModuleList()#trick from dgl
        self.activation_funcs=[]
        self.dropout_values=[]

        self.conv_layers.append(GCNConv(dim_list[0],dim_list[1], cached=True) )
        self.bn_layers.append(nn.BatchNorm1d(dim_list[1]))
        self.activation_funcs.append(F.relu)
        self.dropout_values.append(min(0.1,args.dropout))  # trick


        for i in range(len(dim_list)-2):
            if i<len(dim_list)-2:
                bn=nn.BatchNorm1d(dim_list[i+2])
                activation=F.relu
                dropout=args.dropout
            else:
                bn=nn.Identity() #trick from dgl
                activation=lambda x:x #trick from dgl
                dropout=0 #trick from dgl
            conv=GCNConv(dim_list[i+1], dim_list[i+2], cached=True) 
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.activation_funcs.append(activation)
            self.dropout_values.append(dropout)
            
            


    def forward(self, inputs):
        x, support = inputs

        #x = self.layers((x, support))

        for i in range(len(self.dim_list)-1):   
            #print('forward {} layer'.format(i))
            x=self.conv_layers[i](x,support)
            x=  self.bn_layers[i](x)
            x=  self.activation_funcs(x)
            x= F.dropout(x, self.dropout_values[i], training=self.training)



        return x,support

    '''def l2_loss(self):


        for p in self.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss'''
