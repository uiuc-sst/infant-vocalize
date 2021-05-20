from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model_py.cnn_multi_filters_multitask import MultiTaskNet, selfAtt
import util
import numpy as np
from collections import Counter,OrderedDict
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,cohen_kappa_score
import os

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target_sp, target_chn, target_fan, target_man, target_cxn) in enumerate(train_loader):
        data = data.to(device)
        target_sp, target_chn, target_fan, target_man,target_cxn = target_sp.to(device), target_chn.to(device), target_fan.to(device), target_man.to(device),target_cxn.to(device)
        optimizer.zero_grad()
        out_sp,out_chn,out_fan,out_man,out_cxn = model(data)
        loss_sp=0
        loss_chn=0
        loss_fan=0    
        loss_man=0
        loss_cxn=0
        for j in range(out_sp.size(1)):
            loss_sp+=-torch.sum(target_sp*F.log_softmax(out_sp[:,j,:],-1))
            loss_chn+=-torch.sum(target_chn*F.log_softmax(out_chn[:,j,:],-1))
            loss_fan+=-torch.sum(target_fan*F.log_softmax(out_fan[:,j,:],-1))                    
            loss_man+=-torch.sum(target_man*F.log_softmax(out_man[:,j,:],-1))                    
            loss_cxn+=-torch.sum(target_cxn*F.log_softmax(out_cxn[:,j,:],-1))                    

            loss_sp/= out_sp.size(1)
            loss_chn/=out_chn.size(1)
            loss_fan/=out_fan.size(1)
            loss_man/=out_man.size(1)
            loss_cxn/=out_cxn.size(1)

        loss_sp.backward(retain_graph=True)
        loss_chn.backward(retain_graph=True)
        loss_fan.backward(retain_graph=True)
        loss_man.backward(retain_graph=True)
        loss_cxn.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tSpeaker Loss: {:.6f}\tCHN loss: {:.6f}\tFAN loss: {:.6f}\tMAN loss: {:.6f}\tCXN loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_sp.item(), loss_chn.item(),loss_fan.item(),loss_man.item(),loss_cxn.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    ypred_sp=torch.tensor([],dtype=torch.long)
    ytrue_sp=torch.tensor([],dtype=torch.long)
    ypred_type_chn=torch.tensor([],dtype=torch.long)
    ytrue_type_chn=torch.tensor([],dtype=torch.long)
    ypred_type_fan=torch.tensor([],dtype=torch.long)
    ytrue_type_fan=torch.tensor([],dtype=torch.long)   
    ypred_type_man=torch.tensor([],dtype=torch.long)
    ytrue_type_man=torch.tensor([],dtype=torch.long)   
    ypred_type_cxn=torch.tensor([],dtype=torch.long)
    ytrue_type_cxn=torch.tensor([],dtype=torch.long)           

    total=0
    with torch.no_grad():
        for data, target_sp, target_chn, target_fan, target_man, target_cxn in test_loader:
            data = data.to(device)
            out_sp,out_chn,out_fan,out_man,out_cxn = model(data)

            # for j in range(out_sp.size(1)):
            #     test_loss+=-torch.sum(target_sp*F.log_softmax(out_sp[:,j,:],-1))\
            #         -torch.sum(target_chn*F.log_softmax(out_chn[:,j,:],-1))\
            #         -torch.sum(target_fan*F.log_softmax(out_fan[:,j,:],-1))                    
            # test_loss/= out_sp.size(1)

            _ypred_sp, _ypred_type_chn,_ypred_type_fan,_ypred_type_man,_ypred_type_cxn = out_sp.argmax(dim=-1), out_chn.argmax(dim=-1),out_fan.argmax(dim=-1),out_man.argmax(dim=-1),out_cxn.argmax(dim=-1)

            _ypred_sp, _ = torch.mode(_ypred_sp,dim=-1)
            _ypred_type_chn, _ = torch.mode(_ypred_type_chn,dim=-1)
            _ypred_type_fan, _ = torch.mode(_ypred_type_fan,dim=-1)
            _ypred_type_man, _ = torch.mode(_ypred_type_man,dim=-1)
            _ypred_type_cxn, _ = torch.mode(_ypred_type_cxn,dim=-1)

            _ypred_sp,_ypred_type_chn,_ypred_type_fan,_ypred_type_man,_ypred_type_cxn = _ypred_sp.cpu(),_ypred_type_chn.cpu(),_ypred_type_fan.cpu(),_ypred_type_man.cpu(),_ypred_type_cxn.cpu()
            
            ypred_sp=torch.cat((ypred_sp,_ypred_sp))
            ytrue_sp=torch.cat((ytrue_sp,target_sp))
            
            #chn_index=torch.cat(((target_sp==util.CHN_idx).nonzero(),(target_sp==util.SIL_idx).nonzero(),(target_sp==3).nonzero()))
            #fan_index=torch.cat(((target_sp==util.FAN_idx).nonzero(),(target_sp==util.SIL_idx).nonzero(),(target_sp==3).nonzero()))
            chn_index=(target_chn!=0).nonzero()
            fan_index=(target_fan!=0).nonzero()
            man_index=(target_man!=0).nonzero()
            cxn_index=(target_cxn!=0).nonzero()

            ypred_type_chn=torch.cat((ypred_type_chn,_ypred_type_chn[chn_index]))
            ypred_type_fan=torch.cat((ypred_type_fan,_ypred_type_fan[fan_index]))
            ypred_type_man=torch.cat((ypred_type_man,_ypred_type_man[man_index]))
            ypred_type_cxn=torch.cat((ypred_type_cxn,_ypred_type_cxn[cxn_index]))

            ytrue_type_chn=torch.cat((ytrue_type_chn,target_chn[chn_index]-1))
            ytrue_type_fan=torch.cat((ytrue_type_fan,target_fan[fan_index]-1))
            ytrue_type_man=torch.cat((ytrue_type_man,target_man[man_index]-1))
            ytrue_type_cxn=torch.cat((ytrue_type_cxn,target_cxn[cxn_index]-1))

            total+=target_sp.size()[0]
            print(total)
    test_loss /=  len(test_loader.dataset)
    ypred_sp, ypred_type_chn,ypred_type_fan,ypred_type_man,ypred_type_cxn = ypred_sp.data.numpy(), ypred_type_chn.data.numpy(), ypred_type_fan.data.numpy(),ypred_type_man.data.numpy(),ypred_type_cxn.data.numpy()
    ytrue_sp, ytrue_type_chn,ytrue_type_fan,ytrue_type_man,ytrue_type_cxn = ytrue_sp.data.numpy(), ytrue_type_chn.data.numpy(), ytrue_type_fan.data.numpy(),ytrue_type_man.data.numpy(),ytrue_type_cxn.numpy()
    # np.save(os.path.join(prefix,pred_folder,"pred_sp.npy"),ypred_sp)
    # np.save(os.path.join(prefix,pred_folder,"pred_chn.npy"),ypred_type_chn)
    # np.save(os.path.join(prefix,pred_folder,"pred_fan.npy"),ypred_type_fan)

    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    try:
        print("Speaker diarization")
        f1_sp,f1_mac_sp, acc_sp,kappa_sp, cm_sp=compute_metrics(ytrue_sp,ypred_sp)
    except:
        f1_sp,f1_mac_sp, acc_sp,kappa_sp, cm_sp=None,None,None,None,None
    try:
        print("Vocal type CHN")
        f1_type_chn,f1_mac_type_chn, acc_type_chn, kappa_chn, cm_type_chn=compute_metrics(ytrue_type_chn,ypred_type_chn)    
    except:
        f1_type_chn,f1_mac_type_chn, acc_type_chn, kappa_chn, cm_type_chn=None,None,None,None,None
    try:
        print("Vocal type FAN")
        f1_type_fan,f1_mac_type_fan, acc_type_fan,kappa_fan,cm_type_fan=compute_metrics(ytrue_type_fan,ypred_type_fan)       
    except:
        f1_type_fan,f1_mac_type_fan, acc_type_fan,kappa_fan,cm_type_fan=None,None,None,None,None
    try:
        print("Vocal type MAN")    
        f1_type_man,f1_mac_type_man, acc_type_man,kappa_man,cm_type_man=compute_metrics(ytrue_type_man,ypred_type_man)       
    except:
        f1_type_man,f1_mac_type_man, acc_type_man,kappa_man,cm_type_man=None,None,None,None,None
    try:
        print("Vocal type CXN")
        f1_type_cxn,f1_mac_type_cxn, acc_type_cxn,kappa_cxn,cm_type_cxn=compute_metrics(ytrue_type_cxn,ypred_type_cxn)       
    except:
        f1_type_cxn,f1_mac_type_cxn, acc_type_cxn,kappa_cxn,cm_type_cxn=None,None,None,None,None

    f1=[f1_sp,f1_type_chn,f1_type_fan,f1_type_man,f1_type_cxn]
    f1_mac=[f1_mac_sp,f1_mac_type_chn,f1_mac_type_fan,f1_mac_type_man,f1_mac_type_cxn]
    acc=[acc_sp,acc_type_chn,acc_type_fan,acc_type_man,acc_type_cxn]
    kappa=[kappa_sp,kappa_chn,kappa_fan,kappa_man,kappa_cxn]    
    cm=[cm_sp,cm_type_chn,cm_type_fan,cm_type_man,cm_type_cxn]
    return f1,f1_mac,acc,kappa,cm

def compute_metrics(ytrue,ypred):
    print("Accuracy")
    acc=accuracy_score(ytrue,ypred)
    print(acc)
    print("weighted f1 score")
    f1=f1_score(ytrue,ypred,average="weighted")
    print(f1)
    print("macro f1 score")
    f1_mac=f1_score(ytrue,ypred,average="macro")
    print(f1_mac)
    print("kappa scores")
    kappa=cohen_kappa_score(ytrue,ypred)
    print(kappa)
    unique_labels=np.unique(ytrue)
    for l in unique_labels:
        curr_ytrue=np.where(ytrue==l,1,0)
        curr_ypred=np.where(ypred==l,1,0)
        print(cohen_kappa_score(curr_ytrue,curr_ypred))
    print("Confusion matrix")
    cm=confusion_matrix(ytrue,ypred)
    print(cm)
    return f1,f1_mac,acc,kappa,cm    

def str2bool(v):
    #print("v",v)
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_model(model,save):
    if save:
        if save[-3:]!='.pt':
            save+='.pt'
        torch.save(model.state_dict(), save)

def load_pretrain_model(model,load,device):
    if load:
        pretrain=torch.load(load,map_location=device)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict and v.shape==model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--intensity', type=str2bool, default=False, metavar='I',
                        help='whether multilabel problem is treated as intensity or label')
    parser.add_argument('--eval_mode', type=str2bool, default=False, metavar='E',
                        help='whether evaluate the model only without training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--feature-train-path', type=str,
                        default=os.path.join(prefix,folder,"total_{}_norm.h5".format(feature_name)))
                        #prefix+"/bergelson/train_combined_multiple1_norm.h5")
    parser.add_argument('--emo-train-path', type=str,
                        default=os.path.join(prefix,folder,"total_label{}.h5".format(fold)))
                        #default=prefix+"/bergelson/train_label_multitask_1.h5")
    parser.add_argument('--feature-test-path',  type=str,
                        default=os.path.join(prefix,folder,"total_{}_norm.h5".format(feature_name)))                        
                        #default=prefix+"/bergelson/test_combined_multiple1_norm.h5")
    parser.add_argument('--emo-test-path', type=str,
                        default=os.path.join(prefix,folder,"total_label{}.h5".format(fold)))
                        #default=prefix+"/bergelson/test_label_multitask_1.h5")

    parser.add_argument('--load', type = str, default=None,
                        help = "Specify if want to load emotion model")
    parser.add_argument('--save', type=str, default=None,
                        help='For Saving the emotion Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    shuffle=True
    train_loader = torch.utils.data.DataLoader(
        dataset=util.dataset_lena_multitask_h5(args.feature_train_path, args.emo_train_path),
        collate_fn=util.collate_lena_multitask_train,batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=util.dataset_lena_multitask_h5(args.feature_test_path, args.emo_test_path),
        collate_fn=util.collate_lena_multitask_test,batch_size=args.batch_size, shuffle=False, **kwargs)
    model = MultiTaskNet(selfAtt).to(device)
    model = load_pretrain_model(model, args.load,device)
    #train emotion classifier
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    #scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    # if args.fine_tune:
    #     print("freezing the entire model except the last layer")
    #     for param in model.parameters():
    #         param.requires_grad=False
    #     model.fc4.weight.requires_grad=True
    #     model.fc4.bias.requires_grad=True

    if args.eval_mode:
        test(args, model, device, test_loader)
    else:
        best_f1_mac = -1
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch)
            curr_f1,curr_f1_mac, curr_acc, curr_kappa, curr_cm =test(args, model, device, test_loader)
            curr_f1=[x for x in curr_f1 if x!=None]
            curr_f1_mac=[x for x in curr_f1_mac if x!=None]
            curr_acc=[x for x in curr_acc if x!=None]
            curr_kappa=[x for x in curr_kappa if x!=None]

            avg_f1=sum(curr_f1)/len(curr_f1)
            avg_f1_mac=sum(curr_f1_mac)/len(curr_f1_mac)
            avg_acc=sum(curr_acc)/len(curr_acc)
            avg_kappa=sum(curr_kappa)/len(curr_kappa)
            if avg_f1_mac>best_f1_mac:
                best_f1_mac=avg_f1_mac
                f1 = curr_f1
                f1_mac=curr_f1_mac
                acc = curr_acc
                cm = curr_cm
                kappa=curr_kappa
                best_epoch=epoch
                save_model(model,args.save)                    
        print("Best epoch",best_epoch)
        print("Best accuracy are ",acc)
        print("Best F1 weighted scores are ",f1)
        print("Best F1 macro scores are ",f1_mac)
        print("Best kappa scores are ",kappa)
        print("Average of Best Acc, weighted F1, macro F1",sum(acc)/len(acc),sum(f1)/len(f1),sum(f1_mac)/len(f1_mac))
        print("Confusion matrices",cm)  


if __name__ == '__main__':
    fold=""
    prefix="/media/jialu/Elements/LENA_110/"
    feature_name="embo{}".format(fold)
    folder="features/no_sil"
    pred_folder="pred/no_sil"
    main()
