from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model_py.cnn_multi_filters_v2 import Net
import util
import os
import numpy as np
from collections import Counter,OrderedDict
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,recall_score
from focalloss import FocalLoss

def train(args, model, device, train_loader, optimizer, epoch, joint=True):

    model.train()
    focal_loss=FocalLoss(gamma=0.2)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        print(data.size())
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if len(output.size())>2:
            loss = 0
            for j in range(output.size(1)):
                if joint:
                    loss += torch.sum(-target*F.log_softmax(output[:,j,:],-1),-1)#target has to be multihot encoding scheme
                else:
                    loss += F.cross_entropy(output[:,j,:], target)
                    #loss += focal_loss(output[:,j,:],target)
            loss/= output.size(1)
        else:
            loss = F.cross_entropy(output,target)

        if joint: loss=loss.mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    ypred=torch.tensor([],dtype=torch.long,device=device)
    ytrue=torch.tensor([],dtype=torch.long,device=device)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if len(output.size())>2:
                loss = 0
                for j in range(output.size(1)):
                    loss += F.cross_entropy(output[:,j,:], target)                
                loss/= output.size(1)
            else:
                loss = F.cross_entropy(output,target)

            test_loss+=loss

            pred = output.argmax(dim=-1)
            pred,_ = torch.mode(pred,dim=-1)
            correct +=pred.eq(target).sum().item()
            ytrue=torch.cat((ytrue,target),dim=0)
            ypred=torch.cat((ypred,pred.view_as(target)),dim=0)

    test_loss /=  len(test_loader.dataset)

    ytrue=ytrue.cpu().data.numpy()
    ypred=ypred.cpu().data.numpy()
    #np.save("/home/jialu/disk1/infant-vocalize/full_mode/{}/cnn/test_{}_norm_pred.npy".format(folder,feature_name),ypred)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("Confusion matrix")
    cm=confusion_matrix(ytrue,ypred)
    print(cm)
    print("Accuracy",correct/len(test_loader.dataset))
    print("weighted f1 score")
    f1=f1_score(ytrue,ypred,average="weighted")
    print(f1)
    print("macro f1 score")
    f1_mac=f1_score(ytrue,ypred,average="macro")
    print(f1_mac)
    print("unweighted average recall")
    uar = recall_score(ytrue,ypred,average="macro")
    print(uar)    
    return f1,f1_mac,correct / len(test_loader.dataset),uar,cm

def compute_metrics(ytrue,ypred):
    print("Confusion matrix")
    cm=confusion_matrix(ytrue,ypred)
    print(cm)
    print("Accuracy")
    acc=accuracy_score(ytrue,ypred)
    print(acc)
    print("weighted f1 score")
    f1=f1_score(ytrue,ypred,average="weighted")
    print(f1)
    print("macro f1 score")
    f1_mac=f1_score(ytrue,ypred,average="macro")
    print(f1_mac)
    return f1,f1_mac,acc,cm    

def test_joint_tier(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    built_in_error=0
    total_labels=0

    ypred_sp=[]
    ytrue_sp=[]
    ypred_type_chn=[]
    ytrue_type_chn=[]
    ypred_type_fan=[]
    ytrue_type_fan=[]    
    ypred_emo=[]
    ytrue_emo=[]        
    #joint_label_dict={'113': 1, '123': 2, '132': 3, '142': 4, '150': 5, '100': 6, '141': 7, '200': 8, '210': 9, '220': 10, '232': 11, '241': 12, '251': 13, '260': 14, '300': 15, '310': 16, '320': 17, '332': 18, '341': 19, '351': 20, '360': 21, '400': 22, '410': 23, '413': 24, '423': 25, '432': 26, '441': 27, '451': 28, '000': 0}
    #joint_label_dict = {v: k for k, v in joint_label_dict.items()}
    joint_label_dict = util.joint_label_code_to_str_dict
    CHN_IDX=1
    total=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if len(output.size())>2:
                loss = 0
                for j in range(output.size(1)):
                    loss += torch.sum(-target*F.log_softmax(output[:,j,:],-1))#target has to be multihot encoding scheme                
                loss/= output.size(1)
                loss=loss.mean()
            else:
                loss = F.cross_entropy(output,target)

            test_loss+=loss

            pred = output.argmax(dim=-1)
            pred,_ = torch.mode(pred,dim=-1)
            for i in range(target.size(0)):
                nonzero=torch.nonzero(target[i,:])
                built_in_error+=nonzero.size(0)-1
                total_labels+=nonzero.size(0)

                ypred_sp.append(int(joint_label_dict[pred[i].item()][0]))
                if ypred_sp[-1]<=CHN_IDX:
                    ypred_type_chn.append(int(joint_label_dict[pred[i].item()][1]))
                else:
                    ypred_type_fan.append(int(joint_label_dict[pred[i].item()][1]))
                ypred_emo.append(int(joint_label_dict[pred[i].item()][2]))

                if pred[i].item() in nonzero:
                    ytrue_sp.append(int(joint_label_dict[pred[i].item()][0]))
                    if ytrue_sp[-1]<=CHN_IDX:
                        ytrue_type_chn.append(int(joint_label_dict[pred[i].item()][1]))
                    else:
                        ytrue_type_fan.append(int(joint_label_dict[pred[i].item()][1]))                    
                    ytrue_emo.append(int(joint_label_dict[pred[i].item()][2]))
                else:        
                    ### if classification is not correct, try to find the label correcponding to the same speaker
                    ### Then calculate the error rate for type and emo separately
                    sp_correct=False    
                    for value in nonzero:
                        sp=int(joint_label_dict[value.item()][0])
                        if sp==ypred_sp[-1]:
                            sp_correct=True
                            ytrue_sp.append(sp)
                            if ytrue_sp[-1]<=CHN_IDX:
                                ytrue_type_chn.append(int(joint_label_dict[value.item()][1]))
                            else:
                                ytrue_type_fan.append(int(joint_label_dict[value.item()][1]))
                            ytrue_emo.append(int(joint_label_dict[value.item()][2]))
                    if not sp_correct:
                        ytrue_sp.append(int(joint_label_dict[nonzero[0].item()][0]))
                        if ypred_sp[-1]<=CHN_IDX:
                            ytrue_type_chn.append(-1)
                        else:
                            ytrue_type_fan.append(-1)
                        ytrue_emo.append(int(joint_label_dict[nonzero[0].item()][2]))

            total+=target.size(0)
            print(total)
    test_loss /=  len(test_loader.dataset)
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    print("Built-in Error")
    print("{}/{},{}".format(built_in_error,total_labels,built_in_error/total_labels))
    print("Speaker diarization")
    f1_sp,f1_mac_sp, acc_sp,cm_sp=compute_metrics(ytrue_sp,ypred_sp)
    print("Vocal type CHN")
    f1_type_chn,f1_mac_type_chn, acc_type_chn,cm_type_chn=compute_metrics(ytrue_type_chn,ypred_type_chn)    
    print("Vocal type FAN")
    f1_type_fan,f1_mac_type_fan, acc_type_fan,cm_type_fan=compute_metrics(ytrue_type_fan,ypred_type_fan)    
    print("Emo type")
    f1_emo,f1_mac_emo, acc_emo,cm_emo=compute_metrics(ytrue_emo,ypred_emo)    
    f1=[f1_sp,f1_type_chn,f1_type_fan,f1_emo]
    f1_mac=[f1_mac_sp,f1_mac_type_chn,f1_mac_type_fan,f1_mac_emo]
    acc=[acc_sp,acc_type_chn,acc_type_fan,acc_emo]
    cm=[cm_sp,cm_type_chn,cm_type_fan,cm_emo]
    return f1,f1_mac,acc,cm,built_in_error/total_labels


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
    parser.add_argument('--attention', type=str2bool, default=True,
                        help='use attention module or not')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--intensity', type=str2bool, default=False, metavar='I',
                        help='whether multilabel problem is treated as intensity or label')
    parser.add_argument('--eval_mode', type=str2bool, default=False, metavar='E',
                        help='whether evaluate the model only without training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--feature-train-path', type=str,
                        help='the file of the fbank of training data',\
                        default=os.path.join(prefix,folder,"train_{}.h5".format(feature_name)))
                        #default=os.path.join(prefix,folder,"train_{}.h5".format(feature_name)))
    parser.add_argument('--emo-train-path', type=str,
                        help='the file of the target of training data',\
                        #default=os.path.join(prefix,"full_mode/merged/merged_idp_lena_google_freesound/","train_selected_cnn_combined{}_multiple_label.h5".format(fold)))
                        default=os.path.join(prefix,folder,"train_label{}.h5".format(fold)))
    parser.add_argument('--feature-test-path', type=str,
                        help='the file of the fbank of testing data',\
                        default=os.path.join(prefix,folder,"test_{}.h5".format(feature_name)))
                        #default=os.path.join(prefix,folder,"test_{}.h5".format(feature_name)))
    parser.add_argument('--emo-test-path', type=str,
                        help='the file of the target of testing data',\
                        default=os.path.join(prefix,folder,"test_label{}.h5".format(fold)))


    parser.add_argument('--load', type = str, default=None,
                        help = "Specify if want to load emotion model")
    parser.add_argument('--save', type=str,
                        help='For Saving the emotion Model',
                        default=None)
                        #default="model/idp_lena_5way_combined{}_multiple_fisher_1000_weighted_sampler.pt".format(fold))
    parser.add_argument('--joint', type=str2bool, default="False",
                        help='if training the joint tier')
    parser.add_argument('--weighted_sampler', action='store_true', default=False,
                        help='use weighted sampler or not, default not use')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    num_classes=util.get_num_classes(args.emo_test_path)
    data_type=args.emo_test_path.split('/')[5]
    shuffle=True
    if data_type=="streaming" or args.eval_mode:
        shuffle=False

    feature_dim=util.get_feature_dim(args.feature_test_path)
    labels,_ = util.read_h5(args.emo_train_path)

    weights = torch.DoubleTensor(util.make_weights_balance_classes(labels.flatten()))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=True)

    train_loader = torch.utils.data.DataLoader(
            dataset=util.dataset_lena_h5(args.feature_train_path, args.emo_train_path),
            collate_fn=util.collate_lena_fn,batch_size=args.batch_size, 
            shuffle=True,
            **kwargs)
    if args.weighted_sampler:
        print("Use weighted sampler")
        train_loader = torch.utils.data.DataLoader(
            dataset=util.dataset_lena_h5(args.feature_train_path, args.emo_train_path),
            collate_fn=util.collate_lena_fn,batch_size=args.batch_size, 
            sampler=sampler, 
            **kwargs)

    test_loader = torch.utils.data.DataLoader(
        dataset=util.dataset_lena_h5(args.feature_test_path, args.emo_test_path),
        collate_fn=util.collate_lena_fn,batch_size=args.batch_size, shuffle=shuffle, **kwargs)

    model = Net(num_filters=384, num_classes=num_classes,attention=args.attention).to(device)
    model = load_pretrain_model(model, args.load,device)

    #train emotion classifier
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    #scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    if args.eval_mode:
        if args.joint:
            test_joint_tier(args,model,device,test_loader)
        else:
            test(args, model, device, test_loader)
    else:
        best_acc = -1
        uar=-1
        f1_mac=-1
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch, args.joint)
            if args.joint:
                curr_f1,curr_f1_mac, curr_acc, curr_cm,curr_built_in_error =test_joint_tier(args, model, device, test_loader)
                avg_f1=sum(curr_f1)/len(curr_f1)
                avg_f1_mac=sum(curr_f1_mac)/len(curr_f1_mac)
                avg_acc=sum(curr_acc)/len(curr_acc)
                if avg_acc>best_acc:
                    f1 = curr_f1
                    f1_mac=curr_f1_mac
                    best_acc = curr_acc
                    cm = curr_cm
                    best_epoch=epoch
                    best_built_in_error=curr_built_in_error
                    save_model(model,args.save)                    
            else:
                curr_f1,curr_f1_mac, curr_acc, curr_uar, curr_cm = test(args, model, device, test_loader)
                if curr_f1_mac>f1_mac:
                    f1 = curr_f1
                    f1_mac=curr_f1_mac
                    best_acc = curr_acc
                    uar = curr_uar
                    cm = curr_cm
                    best_epoch=epoch
                    save_model(model,args.save)
        print("epoch",best_epoch)
        if args.joint:
            print("Best accuracy are ",acc)
            print("Best F1 weighted scores are ",f1)
            print("Best F1 macro scores are ",f1_mac)
            print("Average of Best Acc, weighted F1, macro F1",sum(acc)/len(acc),sum(f1)/len(f1),sum(f1_mac)/len(f1_mac))
            print("Confusion matrices",cm)  
        else:          
            print("Best accuracy is ",best_acc)
            print("Best F1 weighted score is ",f1)
            print("Best F1 macro score is ",f1_mac)
            print("Best uar is",uar)
            print("Confusion matrix",cm)
    return best_acc, f1_mac, f1, cm

if __name__ == '__main__':
    # fold=""
    # prefix="/media/jialu/Elements/"
    # #feature_name="compare2016{}_norm".format(fold)
    # #feature_name="combined{}_multiple_norm".format(fold)
    # feature_name="embo{}_norm".format(fold)
    # folder="CRIED_features"
    # feature_name+="_fisher_1000_padded"
    # selected_feature_folder="/home/jialu/disk1/infant-vocalize/feature_selection/CRIED/"
    # main()

    prefix="/home/jialu/disk1/infant-vocalize/"
    selected_feature_folder="/home/jialu/disk1/infant-vocalize/feature_selection/lena_idp_5way"
    folder="full_mode/merged/merged_idp_lena_5way"
    #f=open(os.path.join(prefix, "feature_selection","combined_fisher_weighted_sampler_CNN.txt"),"w")
    for feature_num in [1000]:
        accs, mf1s, wf1s = [],[],[]
        for j in [1,2,3]:
            fold = j 
            #feature_name="combined{}_multiple_norm_fisher_1000_padded".format(fold)
            feature_name="combined{}_multiple_norm".format(fold)
            acc, mf1, wf1, conf=main()
            if j==1: confs = conf
            else: confs += conf
            accs.append(acc)
            mf1s.append(mf1)
            wf1s.append(wf1)
        #     f.write(feature_name+"\n")
        #     f.write("acc,{}\n".format(acc))
        #     f.write("weighted f1,{}\n".format(wf1))
        #     f.write("macro f1,{}\n".format(mf1))
        # f.write("mean acc {} std acc {}\n".format(np.mean(accs),np.std(accs)))
        # f.write("mean weighted F1 scores {} std weighted F1 scores {}\n".format(np.mean(wf1s),np.std(wf1s)))
        # f.write("mean macro F1 scores {} std macro F1 scores {}\n".format(np.mean(mf1s),np.std(mf1s)))
        # f.write("Composite confusion matrix\n")
        # f.write(str(confs)+"\n")
