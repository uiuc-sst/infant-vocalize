import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import h5py
from scipy.stats import pearsonr
from collections import Counter

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

# def multi_f1_score_helper_intensity(pred,target,score):
#     print(pred,target)
#     for i in range(len(pred)):
#         nonzero_pred, nonzero_target=np.nonzero(pred[i,:]),np.nonzero(target[i,:])
#         intersection=len(np.intersect1d(pred[i,nonzero_pred], target[i,nonzero_target]))
#         union=max(1,len(np.union1d(pred[i,nonzero_pred], target[i,nonzero_target])))
#         prec_deno = max(1,len(nonzero_pred[0]))
#         recall_deno=max(1,len(nonzero_target[0]))
#         print(intersection,union,prec_deno,recall_deno)
#         score[0] += intersection/union
#         score[1] += intersection/prec_deno
#         score[2] += intersection/recall_deno
#     return score

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

# def collate_fn(batch):
#   (xx, yy) = zip(*batch)
#   x_lens = [len(x) for x in xx]
#   xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).permute(0,2,1)
#   xx_pad.unsqueeze_(1)
#   targets= torch.LongTensor([y-1 for y in yy])
#   return xx_pad, targets

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
    def __init__(self, input_file, label_file, padding_size=1600,transform=None):
        super(dataset_lena_h5, self).__init__()

        self.input_file = h5py.File(input_file, "r")
        self.label_file = h5py.File(label_file,"r")
        self.transform = transform
        self.padding_size=padding_size
        self.length=len(self.label_file)

    def __getitem__(self, index):
        content,label = np.asarray(self.input_file[str(index)]).T,np.asarray(self.label_file[str(index)]).T
        if self.padding_size:
            input=np.zeros((self.padding_size,1))
            input[:len(content),0]=content
            input=input.reshape(-1,40)
        input,label=torch.from_numpy(input).float(),torch.from_numpy(label).long()
        if self.transform:
            input=self.transform(input)
            label=self.transform(label)
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

def get_num_classes(label_file):
    labels = h5py.File(label_file,"r")
    keys=labels.keys()
    label_counts=[]
    for k in keys:
        label_counts.append(labels[k].value)
    return len(Counter(label_counts).keys())
