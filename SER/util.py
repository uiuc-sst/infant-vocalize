import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import h5py
from scipy.stats import pearsonr
from collections import Counter
from torch.utils.data.sampler import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


#joint_label_str_to_code_dict={'113': 1, '123': 2, '132': 3, '142': 4, '150': 5, '100': 6, '141': 7, '200': 8, '210': 9, '220': 10, '232': 11, '241': 12, '251': 13, '260': 14, '300': 15, '310': 16, '320': 17, '332': 18, '341': 19, '351': 20, '360': 21, '400': 22, '410': 23, '413': 24, '423': 25, '432': 26, '441': 27, '451': 28, '000': 0}
joint_label_str_to_code_dict={'11': 1, '12': 2, '13': 3, '14': 4, '15': 5, '10': 6, '20': 7, '21': 8, '22': 9, '23': 10, '24': 11, '25': 12, '26': 13, '30': 14, '31': 15, '32': 16, '33': 17, '34': 18, '35': 19, '36': 20, '40': 21, '41': 22, '42': 23, '43': 24, '44': 25, '45': 26, '00': 0}
joint_label_code_to_str_dict = {v: k for k, v in joint_label_str_to_code_dict.items()}
SIL_idx=0
CHN_idx=1
FAN_idx=2


def calculate_corr_mat(mom_label,chi_label):
    nmom_emo, nchi_emo = mom_label.shape[1],chi_label.shape[1]
    corr=np.zeros((nmom_emo,nchi_emo))
    for i in range(nmom_emo):
        for j in range(nchi_emo):
            corr[i][j],_=pearsonr(mom_label[:,i],chi_label[:,j])
    print(corr)

def calculate_corr_mat_pred_target(pred_label,target_label):
    pred_emo = pred_label.shape[1]
    corr=np.zeros((pred_emo))
    for i in range(pred_emo):
        corr[i],_=pearsonr(pred_label[:,i],target_label[:,i])
    print(corr)

def acc_helper(pred,target):
    """
    Single tier accuracy helper
    """
    correct = np.equal(pred,target).sum(axis=0)
    return correct

def multi_f1_score_helper(pred,target,score):
    for i in range(len(pred)):
        nonzero_pred, nonzero_target=np.nonzero(pred[i,:]),np.nonzero(target[i,:])
        intersection=len(np.intersect1d(nonzero_pred, nonzero_target))
        union=max(1,len(np.union1d(nonzero_pred, nonzero_target)))
        prec_deno = max(1,len(nonzero_pred[0]))
        recall_deno=max(1,len(nonzero_target[0]))
        score[0] += intersection/union
        score[1] += intersection/prec_deno
        score[2] += intersection/recall_deno
    return score

def calculate_corr(data1,data2):
    corr, p_val = pearsonr(data1,data2)
    return corr

def label_convertion(target):
    nutt = len(target)
    nemo = len(target[0])
    nlevels=4
    labels = np.zeros((nutt,nemo,nlevels))
    for i in range(nutt):
        for j in range(nemo):
            for k in range(nlevels):
                if target[i][j]==k: labels[i][j][k]=1
                else: labels[i][j][k]=0
    return labels

def collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    if max_seqlength % 32!=0:
        max_seqlength = (max_seqlength // 32+1)*32
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
    inputs.unsqueeze_(1)
    targets = torch.LongTensor([item[1] for item in batch])
    return inputs, targets

def collate_lena_fn(batch):
    def func(p):
        return p[0].size(1)
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    if max_seqlength % 64!=0:
        max_seqlength = (max_seqlength // 64+1)*64
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
    inputs.unsqueeze_(1)
    targets = torch.LongTensor([item[1] for item in batch])
    return inputs, targets

def collate_lena_joint_tier(batch):
    def func(p):
        return p[0].size(1)
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = min(40,longest_sample.size(0))
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    if max_seqlength % 32!=0:
        max_seqlength = (max_seqlength // 32+1)*32
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    target_sizes = torch.IntTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
    inputs.unsqueeze_(1)
    targets=[]
    for item in batch:
        labels=item[1].unsqueeze(0)
        targets.append(torch.zeros(labels.size(0),27).scatter_(1,labels,1).squeeze().long())
    targets=torch.LongTensor(torch.stack(targets,dim=1)).T
    #print(targets.size())
    return inputs, targets

def collate_lena_multitask_train(batch):
    def func(p):
        return p[0].size(1)        
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    if max_seqlength % 32!=0:
        max_seqlength = (max_seqlength // 32+1)*32
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
    inputs.unsqueeze_(1)
    targets_sp,targets_chn,targets_fan,targets_man,targets_cxn=[],[],[],[],[]
    for item in batch:
        labels=str(item[1])
        if labels=="0": labels="00000"
        #targets_sp.append(min(3,int(labels[0])))
        targets_sp.append(int(labels[0]))
        targets_chn.append(int(labels[1]))
        targets_fan.append(int(labels[2]))
        targets_man.append(int(labels[3]))
        targets_cxn.append(int(labels[4]))
    targets_sp,targets_chn,targets_fan,targets_man,targets_cxn=torch.LongTensor(targets_sp),\
                                    torch.LongTensor(targets_chn),torch.LongTensor(targets_fan),\
                                    torch.LongTensor(targets_man),torch.LongTensor(targets_cxn)

    sp_one_hot=torch.zeros(len(batch),6).scatter_(1,torch.LongTensor(targets_sp).view(-1,1),1).long()
    chn_one_hot=torch.zeros(len(batch),6).scatter_(1,torch.LongTensor(targets_chn).view(-1,1),1).long()
    fan_one_hot=torch.zeros(len(batch),7).scatter_(1,torch.LongTensor(targets_fan).view(-1,1),1).long()
    man_one_hot=torch.zeros(len(batch),7).scatter_(1,torch.LongTensor(targets_man).view(-1,1),1).long()
    cxn_one_hot=torch.zeros(len(batch),6).scatter_(1,torch.LongTensor(targets_cxn).view(-1,1),1).long()

    non_chn_index=(targets_chn==0).nonzero().squeeze()
    non_fan_index=(targets_fan==0).nonzero().squeeze()
    non_man_index=(targets_man==0).nonzero().squeeze()
    non_cxn_index=(targets_cxn==0).nonzero().squeeze()

    chn_one_hot[non_chn_index,:]=0
    fan_one_hot[non_fan_index,:]=0
    man_one_hot[non_man_index,:]=0
    cxn_one_hot[non_cxn_index,:]=0

    #sil_index=(targets_sp==0).nonzero().squeeze()
    # chn_one_hot[sil_index,0]=1
    # fan_one_hot[sil_index,0]=1
    # man_one_hot[sil_index,0]=1
    # cxn_one_hot[sil_index,0]=1

    sp_one_hot,chn_one_hot,fan_one_hot,man_one_hot,cxn_one_hot=torch.LongTensor(sp_one_hot),\
                                    torch.LongTensor(chn_one_hot),torch.LongTensor(fan_one_hot),\
                                    torch.LongTensor(man_one_hot),torch.LongTensor(cxn_one_hot)
    return inputs, sp_one_hot, chn_one_hot[:,1:], fan_one_hot[:,1:], man_one_hot[:,1:], cxn_one_hot[:,1:]

def collate_lena_multitask_test(batch):
    def func(p):
        return p[0].size(1)        
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    if max_seqlength % 32!=0:
        max_seqlength = (max_seqlength // 32+1)*32
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
    inputs.unsqueeze_(1)
    targets_sp,targets_chn,targets_fan,targets_man,targets_cxn=[],[],[],[],[]
    for item in batch:
        labels=str(item[1])
        if labels=="0": labels="00000"
        targets_sp.append(int(labels[0]))
        targets_chn.append(int(labels[1]))
        targets_fan.append(int(labels[2]))
        targets_man.append(int(labels[3]))
        targets_cxn.append(int(labels[4]))
    targets_sp,targets_chn,targets_fan,targets_man,targets_cxn=torch.LongTensor(targets_sp),\
                                    torch.LongTensor(targets_chn),torch.LongTensor(targets_fan),\
                                    torch.LongTensor(targets_man),torch.LongTensor(targets_cxn)
    return inputs, targets_sp, targets_chn, targets_fan, targets_man,targets_cxn

def collate_multi_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    if max_seqlength % 32!=0:
        max_seqlength = (max_seqlength // 32+1)*32
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        #input_percentages[x] = seq_length / float(max_seqlength)
        #target_sizes[x] = len(target)
        #targets.extend(target)
    #targets = torch.IntTensor(targets)
    #return inputs, targets, input_percentages, target_sizes
    inputs.unsqueeze_(1)
    targets_mom = torch.stack([item[1] for item in batch]).long()
    targets_chi = torch.stack([item[2] for item in batch]).long()

    return inputs, targets_mom, targets_chi

class dataset_lena_h5(torch.utils.data.Dataset):
    def __init__(self, input_file, label_file,transform=None):
        super(dataset_lena_h5, self).__init__()

        self.input_file = h5py.File(input_file, "r")
        self.label_file = h5py.File(label_file,"r")
        self.transform = transform
        self.length=len(self.label_file)

    def __getitem__(self, index):
        content,label = np.asarray(self.input_file[str(index)]).T,np.asarray(self.label_file[str(index)]).T
        total_dimension=(len(content)//40+1)*40
        input=np.zeros((total_dimension,1))
        input[:len(content),0]=content
        input=input.reshape(-1,40)
        input,label=torch.from_numpy(input).float(),torch.from_numpy(label).long()
        if self.transform:
            input=self.transform(input)
            label=self.transform(label)
        return input,label

    def __len__(self):
        return self.length

class dataset_lena_multitask_h5(torch.utils.data.Dataset):
    def __init__(self, input_file, label_file,transform=None):
        super(dataset_lena_multitask_h5, self).__init__()

        self.input_file = h5py.File(input_file, "r")
        self.label_file = h5py.File(label_file,"r")
        self.transform = transform
        self.length=len(self.label_file)

    def __getitem__(self, index):
        content,label = np.asarray(self.input_file[str(index)]).T,int(np.asarray(self.label_file[str(index)]))
        total_dimension=(len(content)//40+1)*40
        input=np.zeros((total_dimension,1))
        input[:len(content),0]=content
        input=input.reshape(-1,40)
        input=torch.from_numpy(input).float()

        return input,label

    def __len__(self):
        return self.length

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, input_file, label_file, transform=None):
        super(dataset_h5, self).__init__()

        self.input_file = h5py.File(input_file, "r")
        self.label_file = h5py.File(label_file,"r")
        self.length=len(self.label_file)
        self.transform = transform

    def __getitem__(self, index):
        input=(np.asarray(self.input_file[str(index)]).T)
        label=torch.from_numpy(np.asarray(self.label_file[str(index)]).T).long()
        # if input.shape[1]<32:
        #     concatenate_zeros=32-input.shape[1]
        #     input=np.column_stack((input,np.zeros((40,concatenate_zeros))))
        input=torch.from_numpy(input).float()
        if self.transform:
            input=self.transform(input)
        return input,label

    def __len__(self):
        return self.length

class dataset_multi_h5(torch.utils.data.Dataset):
    def __init__(self, input_file, label_mom_file, label_chi_file, intensity=True,transform=None):
        super(dataset_multi_h5, self).__init__()

        self.input_file = h5py.File(input_file, "r")
        self.label_mom_file = h5py.File(label_mom_file,"r")
        self.label_chi_file = h5py.File(label_chi_file,"r")
        self.transform = transform
        self.intensity = intensity

    def __getitem__(self, index):
        input=torch.from_numpy(np.asarray(self.input_file[str(index)]).T).float()
        label_mom = np.delete(np.asarray(self.label_mom_file[str(index)]).T,1)
        label_chi = np.asarray(self.label_chi_file[str(index)]).T
        if self.intensity:
            label_chi = np.where(label_chi>=5, label_chi-1, label_chi)
        else:
            label_mom = np.where(label_mom>0,1,0)
            label_chi = np.where(label_chi>0,1,0)
        label_mom, label_chi = torch.from_numpy(label_mom).int(),torch.from_numpy(label_chi).int()
        return input,label_mom,label_chi

    def __len__(self):
        return len(self.label_mom_file)

def collate_mfcc_spectro_fn(batch):
  (mfcc,si,spectro,emo) = zip(*batch)
  #x_lens = [len(x) for x in mfcc]
  mfcc_pad = pad_sequence(mfcc, batch_first=True, padding_value=0).permute(0,2,1)
  mfcc_pad.unsqueeze_(1)
  si_label= torch.LongTensor([y-1 for y in si])

  spectro_pad = pad_sequence(spectro, batch_first=True, padding_value=0).permute(0,2,1)
  spectro_pad.unsqueeze_(1)
  emo_label= torch.LongTensor([y-1 for y in emo])

  return mfcc_pad, si_label, spectro_pad, emo_label

class dataset_mfcc_spectro_h5(torch.utils.data.Dataset):
    def __init__(self, mfcc_file, si_file, spectro_file, emo_file, transform=None):
        super(dataset_mfcc_spectro_h5, self).__init__()

        self.mfcc_file = h5py.File(mfcc_file, "r")
        self.si_file = h5py.File(si_file,"r")
        self.spectro_file = h5py.File(spectro_file, "r")
        self.emo_file = h5py.File(emo_file,"r")
        self.transform = transform

    def __getitem__(self, index):
        mfcc=torch.from_numpy(np.asarray(self.mfcc_file[str(index)])).float()
        si=torch.from_numpy(np.asarray(self.si_file[str(index)])).int()
        spectro=torch.from_numpy(np.asarray(self.spectro_file[str(index)])).float()
        emo=torch.from_numpy(np.asarray(self.emo_file[str(index)])).int()

        return mfcc,si,spectro,emo

    def __len__(self):
        return len(self.si_file)

class BalanceSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    labels: Sequence[int]

    def __init__(self, labels: Sequence[int], generator=None) -> None:
        self.labels = labels
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        labels = np.asarray(self.labels).flatten()
        nclass = np.unique(labels)
        indices = []
        for c in nclass:
            curr_indices = np.argwhere(labels==c).flatten()
            np.random.shuffle(curr_indices)
            indices+=list(curr_indices[:40])
        return (indices[i] for i in torch.randperm(len(indices), generator=self.generator))

    def __len__(self) -> int:
        #return len(self.indices)
        return 120


class WeightedSubsetRandomSampler(Sampler):
    r"""Samples elements from a given list of indices with given probabilities (weights), with replacement.

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, indices, weights, num_samples=0):
        # if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool):
        #     raise ValueError("num_samples should be a non-negative integeral "
        #                      "value, but got num_samples={}".format(num_samples))
        self.indices = indices
        weights = [ weights[i] for i in self.indices ]
        self.weights = torch.tensor(weights, dtype=torch.double)
        if num_samples == 0:
            self.num_samples = len(self.weights)
        else:
            self.num_samples = num_samples
        self.replacement = True

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples


def make_weights_balance_classes(labels):
    counter = Counter(labels)
    N=len(labels)
    for k,v in counter.items():
        counter[k] = float(N/v)
    weights = np.zeros_like(labels)
    for i in range(len(weights)):
        weights[i]=counter[labels[i]]
    return weights

def read_h5(filename):
    f=h5py.File(filename,"r")
    keys=list(f.keys())
    if type(f[keys[0]].value)==np.int64:
        content=np.zeros((len(keys),1))
    else:
        content=np.zeros((len(keys),len(f[keys[0]])))
    for k in keys:
        content[int(k)]=np.asarray(f[(str(k))])
    return content,keys

def get_num_classes(label_file):
    labels = h5py.File(label_file,"r")
    keys=labels.keys()
    label_counts=[]
    for k in keys:
        label_counts.append(labels[k].value)
    return len(Counter(label_counts).keys())

def get_feature_dim(input_file):
    h5_in=h5py.File(input_file,"r")
    feature_dim=len(np.asarray(h5_in[str(0)]))
    h5_in.close()
    return feature_dim

def get_counts_classes(label_file):
    labels = h5py.File(label_file,"r")
    label_counts=[]
    for i in range(len(labels)):
        label_counts.append(labels[str(i)].value)
    return Counter(label_counts)

def write_h5(content,fout_name):
    fout=h5py.File(fout_name,"w")

    for k in range(len(content)):
        fout.create_dataset(name=str(k),data=content[k])

def merge_hf(fout_name,f1_name,f2_name,f3_name=None):
    f1=h5py.File(f1_name,"r")
    f2=h5py.File(f2_name,"r")
    if f3_name: f3=h5py.File(f3_name,"r")
    fout=h5py.File(fout_name,"w")

    keys1=f1.keys()
    keys2=f2.keys()
    idx=0
    for k in range(len(keys1)):
        fout.create_dataset(name=str(idx),data=f1[str(k)].value)
        idx+=1
    for k in range(len(keys2)):
        fout.create_dataset(name=str(idx),data=f2[str(k)].value)
        idx+=1
    if f3_name:
        keys3=f3.keys()
        for k in range(len(keys3)):
            fout.create_dataset(name=str(idx),data=f3[str(k)].value)
            idx+=1
