import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F

import  numpy as np
from    data import few_labels, load_data, preprocess_features, preprocess_adj, sparse_to_tuple, json_data_io, load_ogb_data
from    model import GCN
from    config import  args
from    utils import masked_loss, masked_acc, simple_ploter ,adjust_learning_rate,debugPrint
import  warnings
import os

import  scipy.sparse as sp


warnings.filterwarnings("ignore", category=UserWarning)

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)

#bool the str
args.standard_split=eval(args.standard_split)
args.with_psuedo_loss=eval(args.with_psuedo_loss)
args.feature_normalize=eval(args.feature_normalize)
args.bias=eval(args.bias)
args.debug=eval(args.debug)

DebugPrint=debugPrint(args.debug)


if __name__=='__main__':
    # load data
    if 'ogb' in args.dataset:
        load_data=load_ogb_data
    #labels=y_train+ y_val y_test
    adj, features,labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
    # make data few-labels
    if not args.standard_split:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask=few_labels(adj, features, labels,args)
    DebugPrint('adj:', adj.shape)
    DebugPrint('features:', features.shape)
    DebugPrint('y:', y_train.shape, y_val.shape, y_test.shape)
    DebugPrint('mask:', train_mask.shape, val_mask.shape, test_mask.shape)

    # D^-1@X
    
    feature_sparsity=sp.issparse(features)
    if args.feature_normalize:
        features = preprocess_features(features,sparsity=feature_sparsity) # [49216, 2], [49216], [2708, 1433]
    elif feature_sparsity:
        features = sparse_to_tuple(features)


    supports = preprocess_adj(adj)

    device = torch.device('cuda')

    total_label = torch.from_numpy(labels).long().to(device)
    total_label = total_label.argmax(dim=1)
    #raise Exception

    train_label = torch.from_numpy(y_train).long().to(device)
    num_classes = train_label.shape[1]
    train_label = train_label.argmax(dim=1)
    train_mask = torch.from_numpy(train_mask.astype(np.int)).to(device).bool()
    val_label = torch.from_numpy(y_val).long().to(device)
    val_label = val_label.argmax(dim=1)
    val_mask = torch.from_numpy(val_mask.astype(np.int)).to(device).bool()
    test_label = torch.from_numpy(y_test).long().to(device)
    test_label = test_label.argmax(dim=1)
    test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device).bool()

    if feature_sparsity:
        i = torch.from_numpy(features[0]).long().to(device)
        v = torch.from_numpy(features[1]).to(device)
        feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)
    else:
        feature = torch.from_numpy(features).to(device)

    i = torch.from_numpy(supports[0]).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

    DebugPrint('x :', feature)
    DebugPrint('sp:', support)
    try:
        num_features_nonzero = feature._nnz()
    except:
        num_features_nonzero=0
    feat_dim = feature.shape[1]

    hidden_list=eval(args.hidden_list)

    net = GCN(feat_dim, num_classes, num_features_nonzero, feature_sparsity,hidden_list,args)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(                   # dgi trick
    #    optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    #)

    #ploter initialization
    test_ploter=simple_ploter(save_name='test_of_{}'.format(args.exp_id))
    for epoch in range(args.epochs):
        #train
        net.train()
        #adjust_learning_rate(optimizer, args.learning_rate, epoch)   # dgi trick
        out = net((feature, support))
        out = out[0]
        


        loss = masked_loss(out, train_label, train_mask)
        #DebugPrint(train_label)
        #loss += args.weight_decay * net.l2_loss()
        if args.with_psuedo_loss:
            DebugPrint('!!!!!!!!!!!!!!!!!!computing psurdo loss!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            '''psuedo label loss'''
            #softmax the outputs
            softmaxed_out=F.softmax(out,dim=1)
            with torch.no_grad():
                #get psuedo labes
                psuedo_labels= softmaxed_out.argmax(dim=1)
                #true label filtering
                psuedo_labels=psuedo_labels*(1-train_mask.int())+train_label*train_mask.int()
                #get confidence
                placeholder_1=psuedo_labels.unsqueeze(-1).to(device)
                one_hot_pred_labels=torch.zeros(softmaxed_out.shape).to(device).scatter_(1,placeholder_1,1)
                confidence=torch.max( torch.mul( softmaxed_out,one_hot_pred_labels) ,dim=1 )[0] 
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
            loss+= masked_loss(out, psuedo_labels, confidence)


        acc = masked_acc(out, train_label, train_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #lr_scheduler.step(loss)


        if epoch % 10 == 0:
            #DebugPrint('out[0] is {}'.format( out[0]))
            #DebugPrint('train_label[0] is {}'.format( train_label[0]))
            #DebugPrint('train_mask[0] is {}'.format( train_mask[0]))
            #DebugPrint('average(train_mask) is {}'.format( torch.mean( train_mask.float())))
            #DebugPrint('out.shape is {}'.format( out.shape))
            #DebugPrint('support.shape is {}'.format( support.shape))

            DebugPrint('Training epoch={}'.format(epoch), 'train loss={}'.format(loss.item()), 'train acc={}'.format(train_acc.item()))

            #test
            net.eval()
            out = net((feature, support))
            out = out[0]
            acc = masked_acc(out, test_label, test_mask)
            DebugPrint('test acc:', acc.item())
            test_ploter.record_data(acc.item(),x=epoch)
    
    meta_results_path=os.path.join('.','results',args.exp_id)   #./results/exp_id
    
    results_path=os.path.join(meta_results_path,str(args.exp_times))    #./results/exp_id/exp_times

    for path in [meta_results_path,results_path]:
        if not os.path.exists(path):
            os.mkdir(path)




    results_log_path= os.path.join(results_path,'log.txt')
    results_data_path=os.path.join(results_path,'data')
    json_io=json_data_io(file_name=results_data_path)
    json_io.save({'x_list':test_ploter.x_list,'y_list':test_ploter.y_list,'args':vars(args)})

    
    test_ploter.save_name=os.path.join('results',args.exp_id,str(args.exp_times),'test_data')
    test_ploter.show_the_plot_and_save()