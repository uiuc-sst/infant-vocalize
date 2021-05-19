import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import h5py
from collections import Counter
class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, input_file, label_file, feature_additional_file=None, num_classes=4, transform=None):
        super(dataset_h5, self).__init__()

        self.input_file = h5py.File(input_file, "r")
        self.label_file = h5py.File(label_file,"r")
        self.length=len(self.label_file)
        self.transform = transform
        self.feature_additional_file=None
        self.num_classes=num_classes
        if feature_additional_file:
            self.feature_additional_file=h5py.File(feature_additional_file,"r")

    def __getitem__(self, index):
        input=torch.from_numpy(np.asarray(self.input_file[str(index)]).T).float()
        label=torch.from_numpy(np.asarray(self.label_file[str(index)]).T).long()
        #print(input.shape,label.shape)
        if self.transform:
            input=self.transform(input)
        if self.feature_additional_file:
            # age_code=torch.from_numpy(np.asarray(self.age_file[str(index)])).int()
            # one_hot_age=torch.zeros(self.num_classes)
            # one_hot_age[age_code]=1
            additional_feature=torch.from_numpy(np.asarray(self.feature_additional_file[str(index)])).float()
            input=torch.cat((input,additional_feature))
        return input,label

    def __len__(self):
        return self.length

class dataset_h5_two_tiers(torch.utils.data.Dataset):
    def __init__(self, input_file, label_file, age_file=None, num_classes=4, transform=None):
        super(dataset_h5_two_tiers, self).__init__()

        self.input_file = h5py.File(input_file, "r")
        self.label_file = h5py.File(label_file,"r")
        self.length=len(self.label_file)
        self.transform = transform
        self.age_file=None
        self.num_classes=num_classes
        if age_file:
            self.age_file=h5py.File(age_file,"r")

    def __getitem__(self, index):
        input=torch.from_numpy(np.asarray(self.input_file[str(index)]).T).float()
        label=self.label_file[str(index)]
        label_chi,label_mom=torch.from_numpy(np.asarray(label[0])).long(),torch.from_numpy(np.asarray(label[1])).long()
        #print(input.shape,label.shape)
        if self.transform:
            input=self.transform(input)
        if self.age_file:
            age_code=torch.from_numpy(np.asarray(self.age_file[str(index)])).int()
            one_hot_age=torch.zeros(self.num_classes)
            one_hot_age[age_code]=1
            input=torch.cat((input,one_hot_age))
        return input,label_chi,label_mom

    def __len__(self):
        return self.length

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


def make_weights_balance_classes(labels):
    counter = Counter(labels)
    N=len(labels)
    for k,v in counter.items():
        counter[k] = float(N/v)
    weights = np.zeros_like(labels)
    for i in range(len(weights)):
        weights[i]=counter[labels[i]]
    return weights

def get_feature_dim(input_file):
    h5_in=h5py.File(input_file,"r")
    print(len(h5_in.keys()))
    feature_dim=len(np.asarray(h5_in[str(0)]))
    h5_in.close()
    return feature_dim

def get_counts_classes(label_file):
    labels = h5py.File(label_file,"r")
    label_counts=[]
    for i in range(len(labels)):
        label_counts.append(labels[str(i)].value)
    return Counter(label_counts)

def get_num_classes(label_file):
    labels = h5py.File(label_file,"r")
    keys=labels.keys()
    label_counts=[]
    for k in keys:
        label_counts.append(labels[k].value)
    return len(Counter(label_counts).keys())

def write_h5(content,fout_name):
    fout=h5py.File(fout_name,"w")

    for k in range(len(content)):
        fout.create_dataset(name=str(k),data=content[k])

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

def merge_hf(fout_name,f1_name,f2_name,f3_name=None):
    f1=h5py.File(f1_name,"r")
    f2=h5py.File(f2_name,"r")
    if f3_name: f3=h5py.File(f3_name,"r")
    fout=h5py.File(fout_name,"w")

    keys1=f1.keys()
    keys2=f2.keys()
    idx=0
    #print(len(keys1),len(keys2),len(keys1)+len(keys2))
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

def compute_match_pair_error(errors_A1,errors_A2,labels):
    n=len(labels)
    Z=(np.where(errors_A1!=labels)-np.where(errors_A2!=labels))
    mu_hat_Z=np.mean(Z)
    var_hat_Z=1/(n-1)*np.square(np.sum(Z-mu_hat_Z))
    var_mu=var_hat_Z/n
    W=mu_hat_Z/(np.sqrt(var_hat_Z/n))
    rv=norm()
    P_p=rv.pdf(abs(W))
    P=1-P_p
    print("p value",P)

