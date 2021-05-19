from multiprocessing import Pool
import numpy as np
import os
import glob
from sklearn.model_selection import KFold,train_test_split,LeavePGroupsOut,GroupKFold
import opensmile_feature_extraction as ofe
import h5py
import csv
import util
import calc_laugh_features
from praatio import tgio
from scipy.io.wavfile import read,write
from collections import Counter


# prefix="/home/jialu/disk1/infant-vocalize/"
# output_dir="full_mode/merged/merged_idp_lena_5way/"

prefix="/media/jialu/Backup Plus/"
output_dir="LB_virtual_visits/"

if not os.path.exists(prefix+output_dir):
    os.mkdir(prefix+output_dir)

babblecor_dir="/home/jialu/disk1/babblecor/"
providence_dir="/media/jialu/Elements/Providence/Providence_wav_16/"
bergelson_dir="/media/jialu/Backup Plus/Bergelson_wav/2s_mode_wav_16/"

lena_segment_dir1="/home/jialu/disk1/lena/CHN_lena_first_set_segments_2s/"
lena_segment_dir2="/home/jialu/disk1/lena/CHN_lena_second_set_segments_2s/"
#10mins lena file
lena_10mins_3m="/home/jialu/disk1/10mins_lena/NEW PROTOCOL/03m_2s/"
lena_10mins_6m="/home/jialu/disk1/10mins_lena/NEW PROTOCOL/06m_2s/"
lena_10mins_9m="/home/jialu/disk1/10mins_lena/NEW PROTOCOL/09m_2s/"
lena_10mins_12m="/home/jialu/disk1/10mins_lena/NEW PROTOCOL/12m_2s/"
lena_10mins_12m_24m="/home/jialu/disk1/10mins_lena/NEW PROTOCOL/12m_24m_2s/"

lena_110_3m="/media/jialu/Elements/LENA_110/2s_wav/03m_IDP/"
lena_110_6m="/media/jialu/Elements/LENA_110/2s_wav/06m_IDP/"
lena_110_9m="/media/jialu/Elements/LENA_110/2s_wav/09m_IDP/"
lena_110_12m="/media/jialu/Elements/LENA_110/2s_wav/12m_IDP/"

three_month_dir="/home/jialu/disk1/idp/Chi_Mom_wav/03m_2s/"
six_month_dir="/home/jialu/disk1/idp/Chi_Mom_wav/06m_2s/"
nine_month_dir="/home/jialu/disk1/idp/Chi_Mom_wav/09m_2s/"

wtimit_dir="/home/jialu/disk1/ifp-08.ifp.uiuc.edu/protected/wTIMIT/whisper/"
ami_dir="/home/jialu/disk1/ami/wav/"
labels_dict={"CRY":0,"FUS":1,"LAU":2,"BAB":3,"SCR":4,"C":0,\
            "F":1,"L":2,"P":3,"SIL":5}
Mom_labels_dict={"M":0,"A":1,"P":2,"R":3,"L":4,"W":5}

adu_type_dict={"SIL":"0","N":"0",\
            "CDS":"1","M":"1",\
            "FAN":"2","MAN":"2","A":"2",\
            "LAU":"3","LAUC":"3","L":"3",\
            "SNG":"4","SNGC":"4","R":"4",\
            "PLA":"5","PLAC":"5","P":"5",\
            "W":"6"}
chi_type_dict={"CRY":"1","FUS":"2","LAU":"3","BAB":"4","SCR":"5","SIL":"0",\
                "C":"1","F":"2","L":"3","P":"4","N":"0"}
                
age_code_dict={"03m":0,"06m":1,"09m":2}
Mom_face_dict={"I":0,"F":0,"B":1,"S":1}

google_cry_dir="/media/jialu/Elements/google_audioset/wav/CRY/2s_wav/"
google_laugh_dir="/media/jialu/Elements/google_audioset/wav/LAU/2s_wav/"
google_babble_dir="/media/jialu/Elements/google_audioset/wav/BAB/2s_wav/"

CRIED_dir = "/media/jialu/Elements/ComParE2018_fussingCrying/"
#
# google_adult_dir="/home/jialu/disk1/google_audioset/wav/ADU/2s_wav/"
# google_laugh_mom_dir="/home/jialu/disk1/google_audioset/wav/LAU_MOM/2s_wav/"
# google_rhythmic_dir="/home/jialu/disk1/google_audioset/wav/RHY/2s_wav/"
# google_whisper_dir="/home/jialu/disk1/google_audioset/wav/WHI/2s_wav/"
#
fs_cry_dir="/media/jialu/Elements/freesound/wav/Cry/2s_wav/"
fs_laugh_dir="/media/jialu/Elements/freesound/wav/Laugh/2s_wav/"
fs_babble_dir="/media/jialu/Elements/freesound/wav/Babble/2s_wav/"
fs_fuss_dir="/media/jialu/Elements/freesound/wav/Fuss/2s_wav/"

# lena_files1 = sorted(glob.glob(lena_segment_dir1+"*.wav"))
# lena_files2 = sorted(glob.glob(lena_segment_dir2+"*.wav"))
#
lena_10mins_3m_files=sorted(glob.glob(lena_10mins_3m+"*.wav"))
lena_10mins_6m_files=sorted(glob.glob(lena_10mins_6m+"*.wav"))
lena_10mins_9m_files=sorted(glob.glob(lena_10mins_9m+"*.wav"))
lena_10mins_12m_files=sorted(glob.glob(lena_10mins_12m+"*.wav"))
lena_10mins_12m_24m_files=sorted(glob.glob(lena_10mins_12m_24m+"*.wav"))

lena_110_3m_files=sorted(glob.glob(lena_110_3m+"*.wav"))
lena_110_6m_files=sorted(glob.glob(lena_110_6m+"*.wav"))
lena_110_9m_files=sorted(glob.glob(lena_110_9m+"*.wav"))
lena_110_12m_files=sorted(glob.glob(lena_110_12m+"*.wav"))

babblecor_files=sorted(glob.glob(babblecor_dir+"clips/clips_corpus/*.wav"))
providence_files=sorted(glob.glob(providence_dir+"**/*.wav"))

bergelson_files=sorted(glob.glob(bergelson_dir+"**/*.wav"))

lb_files=sorted(glob.glob(prefix+output_dir+"audio_classification/*.wav"))
CRIED_files = sorted(glob.glob(CRIED_dir+"*.wav"))

# idp_03m_files = (glob.glob(three_month_dir+"*P.wav"))+(glob.glob(three_month_dir+"*C.wav"))\
#                 +(glob.glob(three_month_dir+"*F.wav"))+(glob.glob(three_month_dir+"*L.wav"))
# idp_06m_files = (glob.glob(six_month_dir+"*P.wav"))+(glob.glob(six_month_dir+"*C.wav"))\
#                 +(glob.glob(six_month_dir+"*F.wav"))+(glob.glob(six_month_dir+"*L.wav"))
# idp_09m_files = (glob.glob(nine_month_dir+"*P.wav"))+(glob.glob(nine_month_dir+"*C.wav"))\
#                 +(glob.glob(nine_month_dir+"*F.wav"))+(glob.glob(nine_month_dir+"*L.wav"))

# idp_03m_files = (glob.glob(three_month_dir+"*.wav"))
# idp_06m_files = (glob.glob(six_month_dir+"*.wav"))
# idp_09m_files = (glob.glob(nine_month_dir+"*.wav"))

# lena_idp_laugh= glob.glob(lena_segment_dir1+"LAU.wav")+glob.glob(lena_segment_dir2+"LAU.wav")+\
#                 glob.glob(three_month_dir+"*L.wav")+glob.glob(six_month_dir+"*L.wav")+glob.glob(nine_month_dir+"*L.wav")
# lena_idp_cry= glob.glob(lena_segment_dir1+"CRY.wav")+glob.glob(lena_segment_dir2+"CRY.wav")+\
#                 glob.glob(three_month_dir+"*C.wav")+glob.glob(six_month_dir+"*C.wav")+glob.glob(nine_month_dir+"*C.wav")
# lena_idp_fuss= glob.glob(lena_segment_dir1+"FUS.wav")+glob.glob(lena_segment_dir2+"FUS.wav")+\
#                 glob.glob(three_month_dir+"*F.wav")+glob.glob(six_month_dir+"*F.wav")+glob.glob(nine_month_dir+"*F.wav")
# lena_idp_babble= glob.glob(lena_segment_dir1+"BAB.wav")+glob.glob(lena_segment_dir2+"BAB.wav")+\
#                 glob.glob(three_month_dir+"*P.wav")+glob.glob(six_month_dir+"*P.wav")+glob.glob(nine_month_dir+"*P.wav")

# youtube_cry_files =glob.glob(youtube_cry_dir+"fb*.wav")
# youtube_babble_files =glob.glob(youtube_babble_dir+"fb*.wav")
# youtube_fuss_files =glob.glob(youtube_fuss_dir+"fb*.wav")
# youtube_laugh_files =glob.glob(youtube_laugh_dir+"fb*.wav")
# youtube_screech_files =glob.glob(youtube_screech_dir+"fb*.wav")

# idp_03m_files = (glob.glob(three_month_dir+"*.wav"))
# idp_06m_files = (glob.glob(six_month_dir+"*.wav"))
# idp_09m_files = (glob.glob(nine_month_dir+"*.wav"))

# idp_03m_mom_face_files = sorted(glob.glob(three_month_mom_face_dir+"*.wav"))
# idp_06m_mom_face_files = sorted(glob.glob(six_month_mom_face_dir+"*.wav"))
# idp_09m_mom_face_files = sorted(glob.glob(nine_month_mom_face_dir+"*.wav"))

google_cry_files =sorted(glob.glob(google_cry_dir+"*.wav"))
google_laugh_files =sorted(glob.glob(google_laugh_dir+"*.wav"))
google_babble_files =sorted(glob.glob(google_babble_dir+"*.wav"))
#
# google_adult_files =glob.glob(google_adult_dir+"*.wav")
# google_rhy_files =glob.glob(google_rhythmic_dir+"*.wav")
# google_lau_mom_files =glob.glob(google_laugh_mom_dir+"*.wav")
# google_whi_files=glob.glob(google_whisper_dir+"*.wav")
#
fs_cry_files =sorted(glob.glob(fs_cry_dir+"*.wav"))
fs_laugh_files =sorted(glob.glob(fs_laugh_dir+"*.wav"))
fs_babble_files =sorted(glob.glob(fs_babble_dir+"*.wav"))
fs_fuss_files =sorted(glob.glob(fs_fuss_dir+"*.wav"))

# wtimit_files = sorted(glob.glob(wtimit_dir+"*.wav"))
# ami_files=sorted(glob.glob(ami_dir+"*.wav"))

def get_lena_label(files,label):
    selected_files=[]
    for i in range(len(files)):
        file=files[i]
        filename_list = (file.split('.wav')[0]).split('-')
        curr_label=labels_dict[filename_list[-1]]
        if curr_label==label: selected_files.append(file)
    return selected_files

def write_label_h5(curr_files,out,label_type="multitask"):
    #joint_label_dict={'113': 1, '123': 2, '132': 3, '142': 4, '150': 5, '100': 6, '141': 7, '200': 8, '210': 9, '220': 10, '232': 11, '241': 12, '251': 13, '260': 14, '300': 15, '310': 16, '320': 17, '332': 18, '341': 19, '351': 20, '360': 21, '400': 22, '410': 23, '413': 24, '423': 25, '432': 26, '441': 27, '451': 28, '000': 0}    
    joint_label_dict={'11': 1, '12': 2, '13': 3, '14': 4, '15': 5, '21': 6, '22': 7, '23': 8, '24': 9, '25': 10, '26': 11, '31': 12, '32': 13, '33': 14, '34': 15, '35': 16, '41': 17, '42': 18, '43': 19, '44': 20, '45': 21, '00': 0}
    label_dict_babble=np.load(babblecor_dir+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_3m_play=np.load(three_month_dir+"labels_play_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_3m_stillface=np.load(three_month_dir+"labels_stillface_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_6m=np.load(six_month_dir+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_9m=np.load(nine_month_dir+"labels_{}.npy".format(label_type),allow_pickle=True)[()]

    label_dict_lena1=np.load(lena_segment_dir1+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_lena2=np.load(lena_segment_dir2+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_10mins_lena_3m=np.load(lena_110_3m+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_10mins_lena_6m=np.load(lena_110_6m+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_10mins_lena_9m=np.load(lena_110_9m+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_10mins_lena_12m=np.load(lena_110_12m+"labels_{}.npy".format(label_type),allow_pickle=True)[()]

    for i in range(len(curr_files)):
        filename=curr_files[i].strip('\n')
        key = str(filename.split('.wav')[0].split("/")[-1])
        print(key)
        if key in label_dict_3m_play: curr_dict=label_dict_3m_play
        if key in label_dict_3m_stillface: curr_dict=label_dict_3m_stillface
        if key in label_dict_6m: curr_dict=label_dict_6m
        if key in label_dict_9m: curr_dict=label_dict_9m
        if key in label_dict_lena1: curr_dict=label_dict_lena1
        if key in label_dict_lena2: curr_dict=label_dict_lena2
        if key in label_dict_10mins_lena_3m: curr_dict=label_dict_10mins_lena_3m
        if key in label_dict_10mins_lena_6m: curr_dict=label_dict_10mins_lena_6m
        if key in label_dict_10mins_lena_9m: curr_dict=label_dict_10mins_lena_9m
        if key in label_dict_10mins_lena_12m: curr_dict=label_dict_10mins_lena_12m
        if key in label_dict_babble: curr_dict=label_dict_babble

        curr_label=[]
        if label_type=="multitask":
            out.create_dataset(name=str(i),data=np.asarray([int(curr_dict[key][0])]))
        else:
            for value in curr_dict[key]:
                curr_label.append(joint_label_dict[value])
            out.create_dataset(name=str(i),data=np.asarray(curr_label))

def write_label_h5_idp_virtual(wav_files,out):
    for i in range(len(wav_files)):
        wav_file=wav_files[i].strip()
        infant_label = os.path.basename(wav_file).split("_")[-3]
        mother_label = os.path.basename(wav_file).split("_")[-2]
        if infant_label=="N" and mother_label=="N":
            out.create_dataset(name=str(i),data=np.asarray([0]))
        elif infant_label=="N" and mother_label!="N":
            out.create_dataset(name=str(i),data=np.asarray([int("20{}00".format(adu_type_dict[mother_label]))]))
        elif infant_label!="N" and mother_label=="N":
            out.create_dataset(name=str(i),data=np.asarray([int("1{}000".format(chi_type_dict[infant_label]))]))
        else:
            out.create_dataset(name=str(i),data=np.asarray([int("5{}{}00".format(chi_type_dict[infant_label],adu_type_dict[mother_label]))]))
     

def remove_sil_label(curr_files,in_h5,out_files,out_h5_label,out_h5_features,label_type="multitask"):
    label_dict_10mins_lena_3m=np.load(lena_110_3m+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_10mins_lena_6m=np.load(lena_110_6m+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_10mins_lena_9m=np.load(lena_110_9m+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    label_dict_10mins_lena_12m=np.load(lena_110_12m+"labels_{}.npy".format(label_type),allow_pickle=True)[()]
    f=open(out_files,"w")

    selected_idx=[]
    k=0
    for i in range(len(curr_files)):
        filename=curr_files[i].strip('\n')
        key = str(filename.split('.wav')[0].split("/")[-1])
        print(key)
        if key in label_dict_10mins_lena_3m: curr_dict=label_dict_10mins_lena_3m
        if key in label_dict_10mins_lena_6m: curr_dict=label_dict_10mins_lena_6m
        if key in label_dict_10mins_lena_9m: curr_dict=label_dict_10mins_lena_9m
        if key in label_dict_10mins_lena_12m: curr_dict=label_dict_10mins_lena_12m
        
        if int(curr_dict[key][0])!=0:
            out_h5_label.create_dataset(name=str(k),data=np.asarray([int(curr_dict[key][0])]))
            f.write(curr_files[i])
            selected_idx.append(i)
            k+=1
    util.write_selected_dataset(selected_idx,in_h5,out_h5_features)
    

def write_h5(curr_files,hfX,type_name,hfY=None,hfZ=None,file_description=None,laugh_feature=False):
    labels=[]
    valid_idx=0
    for i in range(len(curr_files)):
        file=curr_files[i].strip('\n')
        print(file)
        #### Child label scheme
        filename_list = (file.split('.wav')[0]).split("_")
        #label=labels_dict[filename_list[-1]]
        if not laugh_feature:
            feature = ofe.get_feature_opensmile(file,prefix+output_dir,type_name)
        else:
            feature = ofe.get_feature_opensmile(file,prefix+output_dir,type_name)
            feature = np.concatenate((feature,calc_laugh_features.process_data(file)))
        if feature.shape!=(0,):
            if hfZ:
                hfZ.create_dataset(name=str(valid_idx),data=age_code)
            if hfY:
                hfY.create_dataset(name=str(valid_idx),data=label)
            if file_description:
                file_description.write(file+"\n")
            hfX.create_dataset(name=str(valid_idx),data=feature)
            valid_idx+=1

def write_streaming_h5(curr_files,hfX,type_name,hfY=None,hfZ=None,file_description=None,laugh_feature=False):
    labels=[]
    valid_idx=0
    for i in range(len(curr_files)):
        file=curr_files[i].strip('\n')
        print(file)
        #### Child label scheme
        filename_list = (file.split('.wav')[0]).split('-')
        label=int(filename_list[-1])

        if not laugh_feature:
            feature = ofe.get_feature_opensmile(file,prefix+output_dir,type_name)
        else:
            feature = calc_laugh_features.process_data(file)
        if feature.shape!=(0,):
            if hfZ:
                hfZ.create_dataset(name=str(valid_idx),data=age_code)
            if hfY:
                hfY.create_dataset(name=str(valid_idx),data=label)
            if file_description:
                file_description.write(file+"\n")
            hfX.create_dataset(name=str(valid_idx),data=feature)
            valid_idx+=1


### Junk from here
#files=[]
# files += CRIED_files
#files+=lb_files
# files+=google_cry_files
# files+=google_laugh_files
# files+=google_babble_files
# files+=fs_cry_files
# files+=fs_fuss_files
# files+=fs_laugh_files
# files+=fs_babble_files

# files+=lena_110_3m_files
# files+=lena_110_6m_files
# files+=lena_110_9m_files
# files+=lena_110_12m_files

#f_total = open(prefix+output_dir+"features/total_files.txt","r")
#files=f_total.readlines()
f_total_new = open(prefix+output_dir+"features/total_files.txt","r")
files=f_total_new.readlines()
#hf_trainX=h5py.File(prefix+output_dir+"features/total_embo.h5","w")
hf_trainY=h5py.File(prefix+output_dir+"features/total_label.h5","w")
#write_h5(files,hf_trainX,"emobase2010",file_description=f_total)
# # remove_sil_label(files,prefix+output_dir+"total_embo.h5",f_no_sil,\
# #     hf_trainY,prefix+output_dir+"total_embo_no_sil.h5")
write_label_h5_idp_virtual(files,hf_trainY)

# selected_idx=[]
# for i in range(len(files)):
#     wav_file=files[i].strip()
#     infant_label = os.path.basename(wav_file).split("_")[-3]
#     mother_label = os.path.basename(wav_file).split("_")[-2]

#     if infant_label=="N" and mother_label=="N": continue
#     selected_idx.append(i)
#     f_total_new.write(files[i])

# util.write_selected_dataset(selected_idx,\
#         prefix+output_dir+"features/total_embo.h5",\
#         prefix+output_dir+"features/no_sil/total_embo.h5")

# util.normalize_features(prefix+output_dir+"features/no_sil/total_embo.h5",\
#      prefix+output_dir+"features/no_sil/total_embo_norm.h5")
# f_total_new.close()

#write_label_h5(files,hf_trainY,"multitask")
# training_label=[labels_dict["CRY"]]*len(google_cry_files)+[labels_dict["LAU"]]*len(google_laugh_files)+[labels_dict["BAB"]]*len(google_babble_files)+\
#             [labels_dict["CRY"]]*len(fs_cry_files)+[labels_dict["FUS"]]*len(fs_fuss_files)+[labels_dict["LAU"]]*len(fs_laugh_files)+[labels_dict["BAB"]]*len(fs_babble_files)
# util.write_h5(np.asarray(training_label),prefix+output_dir+"total_label.h5")
# util.clear_data(prefix+"idp_mom/train_embo.h5",prefix+"idp_mom/train_label.h5",prefix+"idp_mom/train_embo_clear.h5",prefix+"idp_mom/train_label_clear.h5")
# util.clear_data(prefix+"idp_mom/test_embo.h5",prefix+"idp_mom/test_label.h5",prefix+"idp_mom/test_embo_clear.h5",prefix+"idp_mom/test_label_clear.h5")
# util.clear_data(prefix+"idp_mom/train_label.h5",prefix+"idp_mom/train_label.h5",prefix+"idp_mom/train_embo_clear.h5",prefix+"idp_mom/train_label_clear.h5")
# util.clear_data(prefix+"idp_mom/test_label.h5",prefix+"idp_mom/train_label.h5",prefix+"idp_mom/train_embo_clear.h5",prefix+"idp_mom/train_label_clear.h5")
# util.split_data_mom(prefix+"full_mode/google_audioset_mom/train_embo.h5",prefix+"full_mode/google_audioset_mom/train_label.h5",\
#                     prefix+"full_mode/google_audioset_mom/train_embo_ADU.h5",prefix+"full_mode/google_audioset_mom/train_embo_RHY.h5",\
#                     prefix+"full_mode/google_audioset_mom/train_embo_LAU_MOM.h5",prefix+"full_mode/google_audioset_mom/train_embo_WHI.h5")

#util.merge_hf(prefix+output_dir+"total_label.h5",prefix+"2s_mode/idp_chi_mom/total_label.h5",prefix+"2s_mode/lena_CHN/total_label.h5")
#util.merge_hf(prefix+output_dir+"total_embo.h5",prefix+"2s_mode/idp_chi_mom/total_embo.h5",prefix+"2s_mode/lena_CHN/total_embo.h5")
# original_label,_=util.read_h5(prefix+"full_mode/idp_mom/train_label.h5")
# adult_content,_=util.read_h5(prefix+"full_mode/google_audioset_mom/train_embo_ADU_norm.h5")
# whi_content,_=util.read_h5(prefix+"full_mode/google_audioset_mom/train_embo_WHI_norm.h5")
#
# labels=np.concatenate((original_label,np.asarray([Mom_labels_dict["A"]]*len(adult_content))))
# labels=np.concatenate((labels,np.asarray([Mom_labels_dict["W"]]*len(whi_content))))
# util.write_h5(labels,prefix+"full_mode/merged_idp_google_mom/train_label_ADU_WHI.h5")

#util.merge_hf(prefix+"google_audioset/train_label_CRY_LAU.h5",prefix+"google_audioset/train_label_BAB.h5",prefix+"google_audioset/train_label.h5")
# util.normalize_features(prefix+"google_audioset/train_embo.h5",prefix+"google_audioset/train_embo_norm.h5")
#util.merge_hf(prefix+"merged_idp_lena/train_embo_norm.h5",prefix+"merged_google_freesound/train_embo_laugh.h5",prefix+"merged_idp_lena_google_fs/train_embo_4way_laugh_norm.h5")
#util.merge_hf(prefix+"merged_idp_lena/train_label.h5",prefix+"merged_google_freesound/train_label_laugh.h5",prefix+"merged_idp_lena_google_fs/train_label_laugh.h5")
#util.merge_hf(prefix+"google_audioset/train_label.h5",prefix+"freesound/train_label.h5",prefix+"merged_google_freesound/train_label.h5")
#util.merge_hf(prefix+"full_mode/merged/merged_google_freesound/train_embo.h5",prefix+"full_mode/google_audioset/train_embo.h5",prefix+"full_mode/freesound/train_embo.h5")
#util.merge_hf(prefix+"full_mode/merged/merged_google_freesound/train_label.h5",prefix+"full_mode/google_audioset/train_label.h5",prefix+"full_mode/freesound/train_label.h5")
#util.normalize_features(prefix+"full_mode/merged/merged_google_freesound/train_embo.h5",prefix+"full_mode/merged/merged_google_freesound/train_embo_norm.h5")


#util.split_data(prefix+"merged_google_freesound/train_embo_norm.h5",prefix+"merged_google_freesound/train_label.h5",
#                        prefix+"merged_google_freesound/train_embo_cry.h5",prefix+"merged_google_freesound/train_embo_laugh.h5",
#                        prefix+"merged_google_freesound/train_embo_fuss.h5",prefix+"merged_google_freesound/train_embo_babble.h5")
# util.normalize_features(prefix+"youtube_norm/train_embo_cry.h5",prefix+"youtube_norm/train_embo_cry_norm.h5")
# util.normalize_features(prefix+"youtube_norm/train_embo_laugh.h5",prefix+"youtube_norm/train_embo_laugh_norm.h5")
# util.normalize_features(prefix+"youtube_norm/train_embo_fuss.h5",prefix+"youtube_norm/train_embo_fuss_norm.h5")
# util.normalize_features(prefix+"youtube_norm/train_embo_babble.h5",prefix+"youtube_norm/train_embo_babble_norm.h5")
# content,_=util.read_h5(prefix+"merged_idp_lena_5way/train_embo_norm.h5")

#f_total = open(prefix+output_dir+"total_files.txt","r")
#files=f_total.readlines()
#files_list = train_test_split(files, train_size=0.8, shuffle=True, random_state=2)
#hf_trainX=h5py.File(prefix+output_dir+"total_is09.h5","w")
#hf_trainY=h5py.File(prefix+output_dir+"total_label.h5","w")
# #hf_trainZ=h5py.File(prefix+output_dir+"train_age_code.h5","w")
#hf_testX=h5py.File(prefix+output_dir+"test_embo.h5","w")
#hf_testY=h5py.File(prefix+output_dir+"test_label.h5","w")
# #hf_testZ=h5py.File(prefix+output_dir+"test_age_code.h5","w")
#f_train=open(prefix+output_dir+"train_files.txt","w")
#f_test=open(prefix+output_dir+"test_files.txt","w")
#write_h5(files,hf_trainX,"total_",None,None,None,False)
# util.concatenate_hf(prefix+output_dir+"total_embo.h5",prefix+output_dir+"total_laughter.h5",prefix+output_dir+"total_combined.h5")
#write_streaming_h5(test_files,hf_testX,"test_",hf_testY,None,f_test)
#util.normalize_features(prefix+output_dir+"train_embo.h5",prefix+output_dir+"train_embo_norm.h5",)
#util.normalize_features(prefix+output_dir+"test_embo.h5",prefix+output_dir+"test_embo_norm.h5",)

#util.normalize_features(prefix+output_dir+"test_is09.h5",prefix+output_dir+"test_is09_norm.h5")
#util.normalize_features(prefix+output_dir+"test_is09.h5",prefix+output_dir+"test_is09_norm.h5")
#util.normalize_features("/home/jialu/disk1/Audio_Speech_Actors_01-24/test_embo.h5","/home/jialu/disk1/Audio_Speech_Actors_01-24/test_embo_norm.h5")
#util.normalize_features("/home/jialu/disk1/Audio_Speech_Actors_01-24/train_embo.h5","/home/jialu/disk1/Audio_Speech_Actors_01-24/train_embo_norm.h5")

# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_5way/total_combined.h5",prefix+"full_mode/lena_child_5way/total_combined.h5",
#    prefix+"full_mode/idp_child_4way/total_combined.h5",prefix+"full_mode/10mins_child_5way/total_combined.h5")
# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_5way/total_label.h5",prefix+"full_mode/lena_child_5way/total_label.h5",
#     prefix+"full_mode/idp_child_4way/total_label.h5",prefix+"full_mode/10mins_child_5way/total_label.h5")
# util.merge_file(prefix+"full_mode/merged/merged_idp_lena_5way/total_files.txt",prefix+"full_mode/lena_child_5way/total_files.txt",
#     prefix+"full_mode/idp_child_4way/total_files.txt",prefix+"full_mode/10mins_child_5way/total_files.txt")

# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_5way/total_age.h5",\
#     prefix+"full_mode/lena_child_5way/total_age.h5",\
#     prefix+"full_mode/idp_child_4way/total_age.h5",
#     prefix+"full_mode/10mins_child_5way/total_age.h5")
# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_label3.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_5way/train_label3.h5",\
#     prefix+"full_mode/google_audioset/train_label.h5",\
#     prefix+"full_mode/freesound/train_label.h5")
# labels,_=util.read_h5(prefix+"full_mode/merged/merged_idp_lena_5way/train_label1.h5")
# labels=np.concatenate((labels,[labels_dict["CRY"]]*100))
# labels=np.concatenate((labels,[labels_dict["LAU"]]*300))
# util.write_h5(labels,prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_label1_CRY_LAU.h5")
# util.normalize_features(prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_embo1_CRY_LAU.h5",\
#             prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_embo1_CRY_LAU_norm.h5")

# util.convert_5way_to_4way(prefix+"full_mode/merged/merged_idp_lena_5way/total_embo.h5",prefix+"full_mode/merged/merged_idp_lena_5way/total_label.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_4way/total_embo.h5",prefix+"full_mode/merged/merged_idp_lena_4way/total_label.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_5way/total_files.txt",prefix+"full_mode/merged/merged_idp_lena_4way/total_files.txt")
#

# f=open(prefix+output_dir+"total_files_no_sil.txt","r")
# total_files=f.readlines()
# kf=KFold(n_splits=3,random_state=1,shuffle=True)
# k=1
# for train_index,test_index in kf.split(total_files):
#     f_train=open(prefix+output_dir+"train_files_no_sil{}.txt".format(k),"w")
#     f_test=open(prefix+output_dir+"test_files{}_no_sil.txt".format(k),"w")
#     util.write_file(train_index,total_files,f_train)
#     util.write_file(test_index,total_files,f_test)
#     util.write_selected_dataset(train_index,prefix+output_dir+"total_embo_no_sil.h5",prefix+output_dir+"train_embo_no_sil"+str(k)+".h5")
#     util.write_selected_dataset(test_index,prefix+output_dir+"total_embo_no_sil.h5",prefix+output_dir+"test_embo_no_sil"+str(k)+".h5")
#     util.write_selected_dataset(train_index,prefix+output_dir+"total_label_no_sil.h5",prefix+output_dir+"train_label_no_sil"+str(k)+".h5")
#     util.write_selected_dataset(test_index,prefix+output_dir+"total_label_no_sil.h5",prefix+output_dir+"test_label_no_sil"+str(k)+".h5")
#     util.normalize_features(prefix+output_dir+"train_embo_no_sil"+str(k)+".h5",prefix+output_dir+"train_embo_no_sil"+str(k)+"_norm.h5")
#     util.normalize_features(prefix+output_dir+"test_embo_no_sil"+str(k)+".h5",prefix+output_dir+"test_embo_no_sil"+str(k)+"_norm.h5")
#     k+=1
#     break

# f_total=open(prefix+output_dir+"test_files1.txt","r")
# total_files=f_total.readlines()
#
# f_out=open(prefix+output_dir1+"test_files1.txt","w")
# selected_idx=[]
# for i in range(len(total_files)):
#     rate,data=read(total_files[i].strip())
#     if len(data)/rate<1: continue
#     selected_idx.append(i)
#     f_out.write(total_files[i])
#
# util.write_selected_dataset(selected_idx,\
#         prefix+output_dir+"test_combined1_multiple.h5",\
#         prefix+output_dir1+"test_combined1_multiple.h5")
#
# util.write_selected_dataset(selected_idx,\
#         prefix+output_dir+"test_label1.h5",\
#         prefix+output_dir1+"test_label1.h5")
# util.normalize_features(prefix+output_dir1+"test_combined1_multiple.h5",prefix+output_dir1+"test_combined1_multiple_norm.h5")


# util.merge_hf(prefix+"2s_mode/merged/merged_10mins_idp_lab/total_combined_multiple.h5",\
#             prefix+"2s_mode/10mins_lena/total_combined_multiple.h5",\
#             prefix+"2s_mode/idp_chi_mom/total_combined_multiple.h5")
# util.merge_hf(prefix+"2s_mode/merged/merged_10mins_idp_lab/total_label_multitask.h5",\
#             prefix+"2s_mode/10mins_lena/total_label_multitask.h5",\
#             prefix+"2s_mode/idp_chi_mom/total_label_multitask.h5")
# util.merge_hf(prefix+"full_mode/merged/merged_10mins_idp_lena/train_total_label.h5",\
#             prefix+"full_mode/merged/merged_idp_lena_5way/train_label1.h5",\
#             prefix+"full_mode/merged/merged_idp_lena_5way/test_label1.h5")
# util.normalize_features(prefix+output_dir+"test_final_embo.h5",\
#             prefix+output_dir+"test_final_embo_norm.h5")
#f_total=open(prefix+output_dir+"total_files.txt","r")

# ages=[]
# for file in f_total:
#     #file_id=(file.split('/')[-1]).split('.')[0][:23]
#     #if file_id in under_3months_id: ages.append(0)
#     file_id=file.split('/')[5]
#     if file_id=="03m_wav": ages.append(0)
#     else: ages.append(1)
#
# util.concatenate_hf_list([prefix+output_dir+"total_embo.h5",\
# prefix+output_dir+"total_is09.h5",\
# prefix+output_dir+"approx_roughness/total_approx_rough.h5",\
# prefix+output_dir+"formants/total_formants.h5",\
# prefix+output_dir+"H1_A1/total_H1_A1.h5",\
# prefix+output_dir+"H1_H2/total_H1_H2.h5",\
# prefix+output_dir+"hnr/total_hnr.h5",\
# prefix+output_dir+"intensity/total_intensity.h5",\
# prefix+output_dir+"pitch/total_pitch.h5",\
# prefix+output_dir+"signal_energy/total_energy.h5"],
# prefix+output_dir+"total_combined_multiple.h5",axis=1)

# util.concatenate_hf_list([prefix+output_dir+"total_embo.h5",\
# prefix+output_dir+"total_is09.h5",\
# prefix+output_dir+"approx_roughness/total_approx_rough.h5",\
# prefix+output_dir+"formants/total_formants.h5",\
# prefix+output_dir+"H1_A1/total_H1_A1.h5",\
# prefix+output_dir+"H1_H2/total_H1_H2.h5",\
# prefix+output_dir+"hnr/total_hnr.h5",\
# prefix+output_dir+"intensity/total_intensity.h5",\
# prefix+output_dir+"pitch/total_pitch.h5",\
# prefix+output_dir+"signal_energy/total_energy.h5",\
# prefix+output_dir+"dur/dur.h5"],
# prefix+output_dir+"total_combined_multiple_dur.h5",axis=1)
# util.normalize_features(prefix+output_dir+"total_combined_multiple_dur.h5",prefix+output_dir+"total_combined_multiple_dur_norm.h5")
#util.normalize_features(prefix+output_dir+"total_embo.h5",prefix+output_dir+"total_embo_norm.h5")

# util.concatenate_hf_list([prefix+output_dir+"test_combined1_multiple.h5",\
# prefix+output_dir+"dur/test_dur1.h5"],
# prefix+output_dir+"test_combined1_multiple_dur.h5",axis=1)
# util.normalize_features(prefix+output_dir+"test_combined1_multiple_dur.h5",prefix+output_dir+"test_combined1_multiple_dur_norm.h5")

# util.concatenate_hf_list([prefix+"2s_mode/10mins_lena/"+"train_combined_multiple1.h5",\
# prefix+"2s_mode/babblecor/"+"train_combined_multiple1_norm.h5",\
# prefix+"2s_mode/providence/"+"train_combined_multiple1_norm.h5"],
# prefix+"2s_mode/merged/merged_10mins_babblecor_providence"+"train_combined_multiple1_norm.h5",0)

# util.concatenate_hf_list([prefix+"2s_mode/10mins_lena/"+"train_label_multitask_1.h5",\
# prefix+"2s_mode/babblecor/"+"train_label_multitask_1.h5",\
# prefix+"2s_mode/providence/"+"train_label_multitask_1.h5"],
# prefix+"2s_mode/merged/merged_10mins_babblecor_providence"+"train_label_multitask_1.h5",0)

# util.concatenate_hf_list([prefix+"2s_mode/10mins_lena/"+"test_combined_multiple1_norm.h5",\
# prefix+"2s_mode/babblecor/"+"test_combined_multiple1_norm.h5",\
# prefix+"2s_mode/providence/"+"test_combined_multiple1_norm.h5"],
# prefix+"2s_mode/merged/merged_10mins_babblecor_providence"+"test_combined_multiple1_norm.h5",0)

# util.concatenate_hf_list([prefix+"2s_mode/10mins_lena/"+"test_label_multitask_1.h5",\
# prefix+"2s_mode/babblecor/"+"test_label_multitask_1.h5",\
# prefix+"2s_mode/providence/"+"test_label_multitask_1.h5"],
# prefix+"2s_mode/merged/merged_10mins_babblecor_providence"+"test_label_multitask_1.h5",0)

# util.merge_minority_class(prefix+"full_mode/merged/merged_google_freesound/train_selected_embo3_norm.h5",\
#     prefix+"full_mode/merged/merged_google_freesound/train_selected_label3.h5",\
#     prefix+"full_mode/merged/merged_google_freesound/train_selected_embo3_norm_CRY_LAU_FUS.h5",\
#     prefix+"full_mode/merged/merged_google_freesound/train_selected_label3_CRY_LAU_FUS.h5",\
#     [0,1,2])

# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_embo3_norm_CRY_LAU_FUS.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_5way/train_embo3_norm.h5",
#     prefix+"full_mode/merged/merged_google_freesound/train_selected_embo3_norm_CRY_LAU_FUS.h5")

# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_label3_CRY_LAU_FUS.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_5way/train_label3.h5",
#     prefix+"full_mode/merged/merged_google_freesound/train_selected_label3_CRY_LAU_FUS.h5")

### train/test loso cried feature set
# f_total = open(prefix+output_dir+"total_files.txt","r")
# files=f_total.readlines()
# f_total.close()
# labels=[]
# groups=[]
# feature_name="compare2016"
# total_features,_ = util.read_h5(prefix+output_dir+"total_{}.h5".format(feature_name))

# for row in files:
#     basename= os.path.basename(row).strip(".wav\n")
#     labels.append(int(basename.split("_")[-1]))
#     groups.append(basename.split("_")[0])
# labels = np.asarray(labels)

# logo = LeavePGroupsOut(n_groups=10)
# for train, test in logo.split(labels,groups=groups):
#     train_labels,test_labels = labels[train],labels[test]
#     counter_train, counter_test = Counter(train_labels),Counter(test_labels)
#     if counter_train[0]==2292 and counter_train[1]==368 and counter_train[2]==178:
#         # util.write_h5(labels[train],prefix+output_dir+"train_label.h5",type=np.int64)
#         # util.write_h5(labels[test],prefix+output_dir+"test_label.h5",type=np.int64)
#         util.write_h5(total_features[train,:],prefix+output_dir+"train_{}.h5".format(feature_name))
#         util.write_h5(total_features[test,:],prefix+output_dir+"test_{}.h5".format(feature_name))
#         util.normalize_features(prefix+output_dir+"train_{}.h5".format(feature_name),prefix+output_dir+"train_{}_norm.h5".format(feature_name))
#         util.normalize_features(prefix+output_dir+"test_{}.h5".format(feature_name),prefix+output_dir+"test_{}_norm.h5".format(feature_name))

#         # train_files =[files[i] for i in train]
#         # test_files = [files[i] for i in test]
#         # f=open(prefix+output_dir+"train_files.txt","w")
#         # f.writelines(train_files)
#         # f.close()
#         # f=open(prefix+output_dir+"test_files.txt","w")
#         # f.writelines(test_files)
#         # f.close()
#         break


