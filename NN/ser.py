from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from collections import Counter,OrderedDict
from sklearn.metrics import confusion_matrix,f1_score,recall_score
from deepFmeasure import deepFmeasureLoss
from loss import FmeasureLoss
from tsne import plot_tsne
from focalloss import FocalLoss
import os
import util
import fc
def train(args, model, device, train_loader, optimizer, epoch, fine_tune=True):
    model.train()
    #criterion=nn.CrossEntropyLoss(reduction='sum')
    fmeasure_loss=FmeasureLoss()
    focal_loss = FocalLoss(gamma=0.2)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # if fine_tune:
        #     loss = fmeasure_loss.forward(output,target)
        # else:
        loss = F.cross_entropy(output,target,reduction='mean')
        #loss = focal_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, fine_tune=True):
    model.eval()
    test_loss = 0
    correct = 0
    ypred=torch.tensor([],dtype=torch.long,device=torch.device("cuda"))
    ytrue=torch.tensor([],dtype=torch.long,device=torch.device("cuda"))
    fmeasure_loss=FmeasureLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #batch loss
            # if fine_tune:
            #     test_loss += fmeasure_loss.forward(output,target)
            # else:
            #test_loss += F.cross_entropy(output,target,reduction='mean')
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            ytrue=torch.cat((ytrue,target),dim=0)
            ypred=torch.cat((ypred,pred),dim=0)
    test_loss /= len(test_loader.dataset)
    ytrue=ytrue.cpu().data.numpy()
    ypred=ypred.cpu().data.numpy()
    np.save("/home/jialu/disk1/infant-vocalize/{}_fcn/test_{}_fisher_norm_pred.npy".format(folder,feature_name),ypred)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("Confusion matrix")
    cm=confusion_matrix(ytrue,ypred)
    print(cm)
    print("Acc")
    print(correct / len(test_loader.dataset))
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

def str2bool(v):
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

        pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
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
    parser.add_argument('--fine_tune', type=str2bool, default="False",
                        help='if use deepFmeasure loss')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval_mode', type=str2bool, default=False, metavar='E',
                        help='whether evaluate the model only without training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--feature-train-path', type=str,
                        help='the file of the opensmile features of training data',\
                        default=os.path.join(selected_feature_folder,"train_"+feature_name+"_fisher_1000.h5"))
                        #default=os.path.join(prefix,folder,"train_"+feature_name+".h5"))
    parser.add_argument('--emo-train-path', type=str,
                        help='the file of the label of training data',\
                        #default=os.path.join(prefix,"full_mode/merged/merged_idp_lena_google_freesound/","train_selected_label{}.h5".format(fold)))
                        default=os.path.join(prefix,folder,"train_label{}.h5".format(fold)))
    parser.add_argument('--feature-additional-train-path', type=str,
                        help='the file of the label of testing data',\
                        default=None)
    parser.add_argument('--feature-test-path', type=str,
                        help='the file of the opensmile features of testing data',\
                        default=os.path.join(selected_feature_folder,"test_"+feature_name+"_fisher_1000.h5"))
                        #default=os.path.join(prefix,folder,"test_"+feature_name+".h5"))
    parser.add_argument('--emo-test-path', type=str,
                        help='the file of the label of testing data',\
                        default=os.path.join(prefix,folder,"test_label{}.h5".format(fold)))
    parser.add_argument('--feature-additional-test-path', type=str,
                        help='the file of the label of testing data',\
                        default=None)
    parser.add_argument('--load', type = str, 
                        #default=None,
                        default = os.path.join("model/idp_lena_5way_combined_fisher_1000_{}.pt".format(fold)),
                        help = "Specify if want to load any emotion model")
    parser.add_argument('--save', type=str, 
                        help='Specify the save path for the model',
                        default=None)
                        #default=os.path.join("model/idp_lena_5way_combined_fisher_1000_{}.pt".format(fold)))
    parser.add_argument('--weighted_sampler', action='store_true', default=False,
                        help='use weighted sampler or not, default not use')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    num_classes=util.get_num_classes(args.emo_train_path)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    input_size=util.get_feature_dim(args.feature_train_path)
    if args.feature_additional_train_path:
        input_size+=util.get_feature_dim(args.feature_additional_train_path)
    model=fc.FC(input_size=input_size,\
            hidden_size=128, fine_tune=args.fine_tune,num_classes=num_classes,stack=False).to(device)
    model = load_pretrain_model(model, args.load, device)

    labels,_ = util.read_h5(args.emo_train_path)
    weights = torch.DoubleTensor(util.make_weights_balance_classes(labels.flatten()))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
            dataset=util.dataset_h5(args.feature_train_path, args.emo_train_path,args.feature_additional_train_path,num_classes),
            batch_size=args.batch_size, 
            shuffle=True,
            **kwargs)

    if args.weighted_sampler:
        train_loader = torch.utils.data.DataLoader(
            dataset=util.dataset_h5(args.feature_train_path, args.emo_train_path,args.feature_additional_train_path,num_classes),
            batch_size=args.batch_size, 
            sampler=sampler, 
            **kwargs)

    test_loader = torch.utils.data.DataLoader(
        dataset=util.dataset_h5(args.feature_test_path, args.emo_test_path,args.feature_additional_test_path,num_classes),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    #train emotion classifier
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    if args.fine_tune:
        print("freezing entire model except last layer")
        for param in model.parameters():
            param.requires_grad=False
        model.fc2.weight.requires_grad=True
        model.fc2.bias.requires_grad=True

    if args.eval_mode:
        test(args, model, device, test_loader,args.fine_tune)
    else:
        uar = -1
        f1_mac = -1 
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch,args.fine_tune)
            curr_f1,curr_f1_mac, curr_acc, curr_uar, curr_cm = test(args, model, device, test_loader,args.fine_tune)
            if curr_f1_mac>f1_mac:
                f1 = curr_f1
                f1_mac=curr_f1_mac
                acc = curr_acc
                uar = curr_uar
                cm=curr_cm
                best_epoch=epoch
                save_model(model,args.save)
        print("epoch",best_epoch)
        print("Best accuracy is ",acc)
        print("Best F1 weighted score is ",f1)
        print("Best F1 macro score is ",f1_mac)
        print("Best uar is",uar)
        print("Confusion matrix",cm)
        return acc, f1_mac, f1, cm

if __name__ == '__main__':

    # prefix="/media/jialu/Elements/"
    # folder="CRIED_features"
    # selected_feature_folder="/home/jialu/disk1/infant-vocalize/feature_selection/CRIED"
    # #feature_name = "embo_norm"
    # feature_name = "combined_multiple_norm"
    # #feature_name = "compare2016_norm"
    # fold=""
    # main()
    prefix="/home/jialu/disk1/infant-vocalize/"
    selected_feature_folder="/home/jialu/disk1/infant-vocalize/feature_selection/lena_idp_5way"
    f=open(os.path.join(prefix, "feature_selection","combined_fisher_1000_2_layer_NN.txt"),"w")
    for feature_num in [1000]:
        accs, mf1s, wf1s = [],[],[]
        for j in [1,2,3]:
            fold = j 
            feature_name="combined{}_multiple_norm".format(fold)
            folder="full_mode/merged/merged_idp_lena_5way"
            #selected_feature_name = "overlap_at_least_3_{}".format(fold)
            acc, mf1, wf1, conf=main()
            if j==1: confs = conf
            else: confs += conf
            accs.append(acc)
            mf1s.append(mf1)
            wf1s.append(wf1)
            f.write(feature_name+"\n")
            f.write("acc,{}\n".format(acc))
            f.write("weighted f1,{}\n".format(wf1))
            f.write("macro f1,{}\n".format(mf1))
        f.write("mean acc {} std acc {}\n".format(np.mean(accs),np.std(accs)))
        f.write("mean weighted F1 scores {} std weighted F1 scores {}\n".format(np.mean(wf1s),np.std(wf1s)))
        f.write("mean macro F1 scores {} std macro F1 scores {}\n".format(np.mean(mf1s),np.std(mf1s)))
        f.write("Composite confusion matrix\n")
        f.write(str(confs)+"\n")


