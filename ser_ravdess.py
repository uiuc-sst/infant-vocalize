from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from model_py.cnn_multi_filters_multitask import Net,selfAtt
from model_py.cnn_multi_filters import Net
import util
import numpy as np
from collections import Counter,OrderedDict
from sklearn.metrics import confusion_matrix,f1_score

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    #criterion=nn.CrossEntropyLoss(reduction='sum')
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if len(output.size())>2:
            loss = 0
            for j in range(output.size(1)):
                loss += F.cross_entropy(output[:,j,:], target)
            loss/= output.size(1)
        else:
            loss = F.cross_entropy(output,target)
        #loss = F.cross_entropy(output,target,reduction='sum')
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
    total = 0
    ypred=torch.tensor([],dtype=torch.long,device=device)
    ytrue=torch.tensor([],dtype=torch.long,device=device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #print(output.size())
            if len(output.size())>2:
                loss = 0
                for j in range(output.size(1)):
                    loss += F.cross_entropy(output[:,j,:], target)
                loss/= output.size(1)
            else:
                loss = F.cross_entropy(output,target)

            test_loss+=loss
            pred = output.argmax(dim=-1)
            pred = torch.squeeze(pred,0)  # get the index of the max log-probability
            pred = pred.view(1,-1).squeeze(0)
            counter=Counter(pred)
            pred = counter.most_common()[0][0]
            correct +=pred.eq(target).sum().item()
            total += 1
            ytrue=torch.cat((ytrue,target),dim=0)
            ypred=torch.cat((ypred,pred.unsqueeze_(0)),dim=0)

            # target = torch.zeros_like(pred).cuda()+target
            # correct += pred.eq(target).sum().item()
            # total += pred.size(0)
            # print(pred,target)

    test_loss /= total

    ytrue=ytrue.cpu().data.numpy()
    ypred=ypred.cpu().data.numpy()
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("Confusion matrix")
    cm=confusion_matrix(ytrue,ypred)
    print(cm)
    print("weighted f1 score")
    f1=f1_score(ytrue,ypred,average="weighted")
    print(f1)
    print("macro f1 score")
    f1_mac=f1_score(ytrue,ypred,average="macro")
    print(f1_mac)
    return f1,f1_mac,correct / len(test_loader.dataset),cm

def pred_helper(output):
    pred = torch.zeros(output[0].size(0),len(output))

    for k in range(len(output)): #nemo
        curr_pred = output[k].argmax(dim=-1)
        val, _ = torch.mode(curr_pred,dim=1)
        pred[:,k]=val
    return pred

def stack_pred_target(orig,new):
    if orig.shape[0]!=0:
        orig=np.vstack((orig,new))
    else:
        orig=new
    return orig

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

        pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CNN and Attention Network')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=21, metavar='N',
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
    parser.add_argument('--fbank-train-path', type=str,
                        help='the file of the fbank of training data',\
                        default="/home/jialu/disk1/Audio_Speech_Actors_01-24/train_fbank.h5")
    parser.add_argument('--emo-train-path', type=str,
                        help='the file of the target of training data',\
                        default="/home/jialu/disk1/Audio_Speech_Actors_01-24/train_label.h5")

    parser.add_argument('--fbank-test-path', type=str,
                        help='the file of the fbank of testing data',\
                        default="/home/jialu/disk1/Audio_Speech_Actors_01-24/test_fbank.h5")
    parser.add_argument('--emo-test-path', type=str,
                        help='the file of the target of testing data',\
                        default="/home/jialu/disk1/Audio_Speech_Actors_01-24/test_label.h5")

    parser.add_argument('--load', type = str, default=None,
                        help = "Specify if want to load emotion model")
    parser.add_argument('--save', type=str, default=None,
                        help='For Saving the emotion Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        dataset=util.dataset_h5(args.fbank_train_path, args.emo_train_path),
        collate_fn=util.collate_fn,batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=util.dataset_h5(args.fbank_test_path, args.emo_test_path),
        collate_fn=util.collate_fn,batch_size=1, shuffle=True, **kwargs)

    num_classes=util.get_num_classes(args.emo_test_path)
    model = Net(num_filters=384, num_classes=num_classes).to(device)
    model = load_pretrain_model(model, args.load,device)

    #train emotion classifier
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    if args.eval_mode:
        test(args, model, device, test_loader)
    else:
        f1 = -1
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch)
            curr_f1,curr_f1_mac, curr_acc, curr_cm = test(args, model, device, test_loader)
            if curr_f1>f1:
                f1 = curr_f1
                f1_mac=curr_f1_mac
                acc = curr_acc
                cm=curr_cm
                best_epoch=epoch
                save_model(model,args.save)
                #hidden_nodes1,hidden_nodes2=model.get_hidden_nodes()
        print("epoch",best_epoch)
        print("Best accuracy is ",acc)
        print("Best F1 weighted score is ",f1)
        print("Best F1 macro score is ",f1_mac)
        print("Confusion matrix",cm)


if __name__ == '__main__':
    main()
