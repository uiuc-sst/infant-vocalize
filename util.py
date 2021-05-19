import h5py
import numpy as np
import random
from scipy.io.wavfile import read,write
import librosa
from collections import Counter
def read_name(file="IS10_name.txt"):
    f=open(file,"r")
    content=f.readlines()
    f.close()
    names=[]
    for row in content:
        name = row.split()[1]
        names.append(name)
    return tuple(names[1:len(names)-1])

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

def read_h5_var(filename):
    f=h5py.File(filename,"r")
    keys=list(f.keys())
    content=[]
    for k_idx in range(len(keys)):
        content.append(np.asarray(f[str(k_idx)]))
    return content,keys

def write_file(idx,total_files,f_out):
    for i in idx:
        f_out.write(total_files[i])
    f_out.close()

def write_h5(content,filename,type=np.float64):
    f=h5py.File(filename,"w")
    for k in range(len(content)):
        f.create_dataset(name=str(k),data=content[k].astype(type))
    f.close()

def merge_file(f_out_file,f1_total,f2_total,f3_total=None):
    f1_total_files=open(f1_total,"r")
    f2_total_files=open(f2_total,"r")
    content1=f1_total_files.readlines()
    content2=f2_total_files.readlines()
    total_files=content1+content2
    if f3_total:
        f3_total_files=open(f3_total,"r")
        total_files+=f3_total_files.readlines()
    f_out=open(f_out_file,"w")
    for i in range(len(total_files)):
        f_out.write(total_files[i])

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
    #for k in range(300):
        fout.create_dataset(name=str(idx),data=f2[str(k)].value)
        idx+=1

    if f3_name:
        keys3=f3.keys()
        for k in range(len(keys3)):
        #for k in range(500):
            fout.create_dataset(name=str(idx),data=f3[str(k)].value)
            idx+=1

def merge_hf_selected(fout_name,f1_name,f2_name,f3_name=None):
    f1=h5py.File(f1_name,"r")
    f2=h5py.File(f2_name,"r")
    if f3_name: f3=h5py.File(f3_name,"r")
    fout=h5py.File(fout_name,"w")

    keys1=f1.keys()
    keys2=list(f2.keys())[:100]
    idx=0
    #print(len(keys1),len(keys2),len(keys1)+len(keys2))
    for k in range(len(keys1)):
        fout.create_dataset(name=str(idx),data=f1[str(k)].value)
        idx+=1

    #for k in range(len(keys2)):
    for k in keys2:
        fout.create_dataset(name=str(idx),data=f2[str(k)].value)
        idx+=1

    if f3_name:
        keys3=list(f3.keys())[:300]
        for k in keys3:
            fout.create_dataset(name=str(idx),data=f3[str(k)].value)
            idx+=1

def concatenate_hf(f1_name,f2_name,fout_name):
    f1=h5py.File(f1_name,"r")
    f2=h5py.File(f2_name,"r")
    fout=h5py.File(fout_name,"w")

    keys1=f1.keys()
    #print(len(keys1),len(keys2),len(keys1)+len(keys2))
    #selected is09 index rms energy/zero crossing rate/voice prob/F0/rms energy sma de
    # selected_is09_idx=list(range(1,13))+list(range(157,169))+\
    #         list(range(169,181))+list(range(181,193))+\
    #         list(range(193,205))
    for k in keys1:
        feature=np.concatenate((f1[k].value,f2[k].value))
        fout.create_dataset(name=k,data=feature)

def convert_5way_to_4way(f_5way,f_5way_label,f_4way,f_4way_label,f_total_5way=None,f_total=None):
    f_in=h5py.File(f_5way,"r")
    f_in_label=h5py.File(f_5way_label,"r")
    f_out=h5py.File(f_4way,"w")
    f_out_label=h5py.File(f_4way_label,"w")
    if f_total_5way:
        f_total_files=open(f_total_5way,"r")
        total_files=f_total_files.readlines()
        f_total_4way_files=open(f_total,"w")

    keys=f_in.keys()
    idx=0
    scr=0
    for k in range(len(keys)):
        if f_in_label[str(k)].value!=4:
            f_out.create_dataset(name=str(idx),data=f_in[str(k)])
            f_out_label.create_dataset(name=str(idx),data=f_in_label[str(k)])
            if f_total:
                f_total_4way_files.write(total_files[k])
            idx+=1

def get_num_classes(label_file):
    labels = h5py.File(label_file,"r")
    keys=labels.keys()
    label_counts=[]
    for k in keys:
        label_counts.append(labels[k].value)
    print(Counter(label_counts))

def write_selected_dataset(idx,h5in_file,h5out_file):
    fin=h5py.File(h5in_file,"r")
    fout=h5py.File(h5out_file,"w")
    k=0
    for i in idx:
        fout.create_dataset(name=str(k),data=fin[str(i)].value)
        k+=1

def normalize_features(h5in_file,h5out_file):
    fin=h5py.File(h5in_file,"r")
    fout=h5py.File(h5out_file,"w")
    keys=list(fin.keys())
    content=np.zeros((len(keys),len(fin[keys[0]])))
    for k_idx in range(len(keys)):
        k=keys[k_idx]
        content[k_idx]=np.asarray(fin[(str(k))])
    means=np.mean(content,axis=0)
    std=np.maximum(np.std(content,axis=0),1e-6)
    content=(content-means)/std
    for k_idx in range(len(keys)):
        k=keys[k_idx]
        fout.create_dataset(name=str(k),data=content[k_idx])

def normalize_features_segment(h5in_file,h5out_file):
    fin=h5py.File(h5in_file,"r")
    fout=h5py.File(h5out_file,"w")
    keys=list(fin.keys())
    content=np.asarray([])
    print(len(keys))
    for k in range(len(keys)):
        if k%1000==0: print(k)
        if content.shape==(0,):
            content=np.asarray(fin[str(k)])
        else:
            content=np.concatenate((content,fin[(str(k))]))
    print("finish concatenate")
    means=np.mean(content,axis=0)
    std=np.maximum(np.std(content,axis=0),1e-6)
    for k in range(len(keys)):
        curr_content=fin[str(k)]
        fout.create_dataset(name=str(k),data=(curr_content-mean)/std)

def get_sample_idx(f1_name,f2_name,sample_num=50):
    labels_lena=[[] for _ in range(4)]
    labels_yt=[[] for _ in range(4)]

    f1=h5py.File(f1_name,"r")
    f2=h5py.File(f2_name,"r")
    keys1=list(f1.keys())
    keys2=list(f2.keys())
    for k in keys1:
        labels_lena[f1[k].value].append(int(k))

    for k in keys2:
        labels_yt[f2[k].value].append(int(k)+len(f1))

    for emo in range(4):
        random.shuffle(labels_lena[emo])
        labels_lena[emo]=labels_lena[emo][:sample_num]
        random.shuffle(labels_yt[emo])
        labels_yt[emo]=labels_yt[emo][:sample_num]
    return labels_lena,labels_yt


def write_selected_sample(f_data_name,selected_idx,out_data_name,out_label_name,lena=True,sample_num=50):
    f_data=h5py.File(f_data_name,"r")
    f_out_data=h5py.File(out_data_name,"w")
    f_out_label=h5py.File(out_label_name,"w")
    for emo_idx in range(len(selected_idx)):
        for i in selected_idx[emo_idx]:
            f_out_data.create_dataset(name=str(i),data=f_data[str(i)])
            if lena:
                f_out_label.create_dataset(name=str(i),data=emo_idx+4)
            else:
                f_out_label.create_dataset(name=str(i),data=emo_idx)

def clear_data(f_data_name,f_label_name,f_out_data_name,f_out_label_name):
    f_data=h5py.File(f_data_name,"r")
    f_label=h5py.File(f_label_name,"r")
    f_out_data=h5py.File(f_out_data_name,"w")
    f_out_label=h5py.File(f_out_label_name,"w")
    keys=list(f_data.keys())
    k=0
    for i in range(len(f_label)):
        new_label=f_label[keys[i]].value
        if new_label<4:
            f_out_data.create_dataset(name=str(k),data=f_data[keys[i]])
            f_out_label.create_dataset(name=str(k),data=new_label)
            k+=1

def clear_data_4way(f_data_name,f_label_name,f_age_name,f_out_data_name,f_out_label_name,f_out_age_name):
    f_data=h5py.File(f_data_name,"r")
    f_label=h5py.File(f_label_name,"r")
    f_age=h5py.File(f_age_name,"r")
    f_out_data=h5py.File(f_out_data_name,"w")
    f_out_label=h5py.File(f_out_label_name,"w")
    f_out_age=h5py.File(f_out_age_name,"w")

    keys=list(f_data.keys())
    k=0
    for i in range(len(f_label)):
        new_label=f_label[keys[i]].value
        if new_label<4:
            f_out_data.create_dataset(name=str(k),data=f_data[keys[i]])
            f_out_label.create_dataset(name=str(k),data=new_label)
            f_out_age.create_dataset(name=str(k),data=f_age[keys[i]])
            k+=1

def split_data(in_file,in_label,out_cry,out_laugh,out_fuss,out_bab):
    f_in=h5py.File(in_file,"r")
    f_in_label=h5py.File(in_label,"r")
    f_out_cry=h5py.File(out_cry,"w")
    f_out_fuss=h5py.File(out_fuss,"w")
    f_out_laugh=h5py.File(out_laugh,"w")
    f_out_bab=h5py.File(out_bab,"w")
    f_out=[f_out_cry,f_out_fuss,f_out_laugh,f_out_bab]
    for i in range(len(f_in)):
        f_out[f_in_label[str(i)].value].create_dataset(name=str(i),data=f_in[str(i)])

def split_data_mom(in_file,in_label,out_adu,out_rhy,out_lau,out_whi):
    f_in=h5py.File(in_file,"r")
    f_in_label=h5py.File(in_label,"r")
    f_out_adu=h5py.File(out_adu,"w")
    f_out_rhy=h5py.File(out_rhy,"w")
    f_out_lau=h5py.File(out_lau,"w")
    f_out_whi=h5py.File(out_whi,"w")
    Mom_labels_dict={"M":0,"A":1,"P":2,"R":3,"L":4,"W":5}
    for i in range(len(f_in)):
        if f_in_label[str(i)].value==Mom_labels_dict["A"]:
            f_out_adu.create_dataset(name=str(i),data=f_in[str(i)])
        if f_in_label[str(i)].value==Mom_labels_dict["R"]:
            f_out_rhy.create_dataset(name=str(i),data=f_in[str(i)])
        if f_in_label[str(i)].value==Mom_labels_dict["L"]:
            f_out_lau.create_dataset(name=str(i),data=f_in[str(i)])
        if f_in_label[str(i)].value==Mom_labels_dict["W"]:
            f_out_whi.create_dataset(name=str(i),data=f_in[str(i)])

def find_avg_magnitude(input_wav_files):
    """
    Find average log fft magnitude
    """
    mag=0
    count=0
    for i in range(len(input_wav_files)):
        rate,data=read(input_wav_files[i])
        if len(data.shape)>1: data=data[:,0]
        if len(data)!=0:
            count+=1
            mag+=np.sum(np.absolute(np.fft.fft(data)),axis=0)/len(data)
    mag/=count
    print("Average log magnitude (db):",20*np.log10(np.sum(mag)))

def write_yaml(files,output_yaml):
    f_yaml=open(output_yaml,"w")
    for wav_name in files:
        wav_name=wav_name.strip('\n')
        rate,data = read(wav_name)
        duration = len(data)/rate
        f_yaml.write("- {{ duration: {}, offset: 0, speaker_id: 001, wav: {} }}\n".format(\
                str(duration),wav_name))

def concatenate_hf_list(h5,out_h5,axis=0):
    content,_=read_h5_var(h5[0])
    content = np.asarray(content)
    if len(content.shape)<2: content=content.reshape(-1,1)
    print(content.shape)
    for j in range(1,len(h5)):
        curr_content,_=read_h5_var(h5[j])
        curr_content=np.asarray(curr_content)
        print(curr_content.shape)
        if len(curr_content.shape)<2: curr_content=curr_content.reshape(-1,1)
        if axis==1:
            content=np.hstack((content,curr_content))
        else:
            content=np.vstack((content,curr_content))
        print(j,content.shape)
    write_h5(content,out_h5)


def merge_minority_class(in_file,in_label,out_file,out_label,classes=[]):
    f_in=h5py.File(in_file,"r")
    f_in_label=h5py.File(in_label,"r")
    f_out=h5py.File(out_file,"w")
    f_out_label=h5py.File(out_label,"w")

    k=0
    for i in range(len(f_in_label.keys())):
        curr_class=f_in_label[str(i)].value 
        if curr_class in classes:
            f_out_label[str(k)]=curr_class
            f_out.create_dataset(name=str(k),data=f_in[str(i)].value)
            k+=1

#get_num_classes("/home/jialu/disk1/infant-vocalize/full_mode/idp_mom_face/test_label1.h5")