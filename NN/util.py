import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import h5py
from collections import Counter
class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, input_file, label_file, transform=None):
        super(dataset_h5, self).__init__()

        self.input_file = h5py.File(input_file, "r")
        self.label_file = h5py.File(label_file,"r")
        self.length=len(self.label_file)
        self.transform = transform

    def __getitem__(self, index):
        input=torch.from_numpy(np.asarray(self.input_file[str(index)]).T).float()
        label=torch.from_numpy(np.asarray(self.label_file[str(index)]).T).long()
        #print(input.shape,label.shape)
        if self.transform:
            input=self.transform(input)
        return input,label

    def __len__(self):
        return self.length

# class dataset_segment_h5(torch.utils.data.Dataset):
#     def __init__(self, input_file, label_file, transform=None):
#         super(dataset_h5, self).__init__()
#
#         self.input_file = h5py.File(input_file, "r")
#         self.label_file = h5py.File(label_file,"r")
#         self.length=len(self.label_file)
#         self.transform = transform
#
#     def __getitem__(self, index):
#         input=torch.from_numpy(np.asarray(self.input_file[str(index)]).T).float()
#         label=torch.from_numpy(np.asarray([self.label_file[str(index)]).T]*input.shape[0]).long()
#         #print(input.shape,label.shape)
#         if self.transform:
#             input=self.transform(input)
#         return input,label
#
#     def __len__(self):
#         return self.length

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


def get_feature_dim(input_file):
    h5_in=h5py.File(input_file,"r")
    feature_dim=len(np.asarray(h5_in[str(0)]))
    h5_in.close()
    return feature_dim

def get_counts_classes(label_file,num_classes=4):
    labels = h5py.File(label_file,"r")
    label_counts=[]
    for i in range(len(labels)):
        label_counts.append(labels[str(i)].value)
    return Counter(label_counts)

def write_h5(content,fout_name):
    fout=h5py.File(fout_name,"w")

    for k in range(len(content)):
        fout.create_dataset(name=str(k),data=content[k])

def merge_hf(f1_name,f2_name,fout_name):
    f1=h5py.File(f1_name,"r")
    f2=h5py.File(f2_name,"r")
    fout=h5py.File(fout_name,"w")

    keys1=f1.keys()
    keys2=f2.keys()
    idx=0
    for k in keys1:
        fout.create_dataset(name=str(idx),data=f1[k])
        idx+=1
    for k in keys2:
        fout.create_dataset(name=str(idx),data=f2[k])
        idx+=1
