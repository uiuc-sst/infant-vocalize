from multiprocessing import Pool
import numpy as np
import os
import glob
from sklearn.model_selection import KFold,train_test_split
import opensmile_feature_extraction as ofe
import h5py
import csv
import util
import calc_laugh_features
from praatio import tgio
from scipy.io.wavfile import read,write
from collections import Counter


prefix="/home/jialu/disk1/infant-vocalize/"
output_dir="full_mode/idp_mom/"

lena_segment_dir1="/home/jialu/lena/CHN_lena_first_set_segments/"
lena_segment_dir2="/home/jialu/lena/CHN_lena_second_set_segments/"

#10mins lena file
lena_10mins_3m="/home/jialu/disk1/10mins_lena/03m_wav/"
lena_10mins_6m="/home/jialu/disk1/10mins_lena/06m_wav/"
lena_10mins_9m="/home/jialu/disk1/10mins_lena/09m_wav/"
lena_10mins_12m="/home/jialu/disk1/10mins_lena/12m_wav/"
lena_10mins_12m_24m="/home/jialu/disk1/10mins_lena/12m_24m_wav/"

three_month_dir="/home/jialu/idp/Child_wav/03m_wav/"
six_month_dir="/home/jialu/idp/Child_wav/06m_wav/"
nine_month_dir="/home/jialu/idp/Child_wav/09m_wav/"

three_month_mom_dir="/home/jialu/idp/Mom_wav/03m_wav/"
six_month_mom_dir="/home/jialu/idp/Mom_wav/06m_wav/"
nine_month_mom_dir="/home/jialu/idp/Mom_wav/09m_wav/"

three_month_mom_face_dir="/home/jialu/idp/Mom_wav_face/03m_wav/"
six_month_mom_face_dir="/home/jialu/idp/Mom_wav_face/06m_wav/"
nine_month_mom_face_dir="/home/jialu/idp/Mom_wav_face/09m_wav/"

three_month_chi_mom_dir="/home/jialu/idp/Chi_Mom_wav/03m_wav/"
six_month_chi_mom_dir="/home/jialu/idp/Chi_Mom_wav/06m_wav/"
nine_month_chi_mom_dir="/home/jialu/idp/Chi_Mom_wav/09m_wav/"

labels_dict={"CRY":0,"FUS":1,"LAU":2,"BAB":3,"SCR":4,"C":0,\
            "F":1,"L":2,"P":3,"SIL":5}
Mom_labels_dict={"M":0,"A":1,"P":2,"R":3,"L":4,"W":5}
age_code_dict={"03m":0,"06m":1,"09m":2}
Mom_face_dict={"I":0,"F":0,"B":1,"S":1}

#
# youtube_cry_dir="/media/jialu/Elements/youtube_data/wav/Cry/2s_wav/"
# youtube_babble_dir="/media/jialu/Elements/youtube_data/wav/Babble/0.5s_wav/"
# youtube_fuss_dir="/media/jialu/Elements/youtube_data/wav/Fuss/0.5s_wav/"
# youtube_laugh_dir="/media/jialu/Elements/youtube_data/wav/Laugh/2s_wav/"
# youtube_screech_dir="/media/jialu/Elements/youtube_data/wav/Screech/2s_wav/"
#
# google_cry_dir="/home/jialu/disk1/google_audioset/wav/CRY/2s_wav/"
# google_laugh_dir="/home/jialu/disk1/google_audioset/wav/LAU/2s_wav/"
# google_babble_dir="/home/jialu/disk1/google_audioset/wav/BAB/2s_wav/"
#
# google_adult_dir="/home/jialu/disk1/google_audioset/wav/ADU/2s_wav/"
# google_laugh_mom_dir="/home/jialu/disk1/google_audioset/wav/LAU_MOM/2s_wav/"
# google_rhythmic_dir="/home/jialu/disk1/google_audioset/wav/RHY/2s_wav/"
# google_whisper_dir="/home/jialu/disk1/google_audioset/wav/WHI/2s_wav/"
#
# fs_cry_dir="/home/jialu/disk1/freesound/wav/Cry/2s_wav/"
# fs_laugh_dir="/home/jialu/disk1/freesound/wav/Laugh/2s_wav/"
# fs_babble_dir="/home/jialu/disk1/freesound/wav/Babble/2s_wav/"
# fs_fuss_dir="/home/jialu/disk1/freesound/wav/Fuss/2s_wav/"

lena_files1 = sorted(glob.glob(lena_segment_dir1+"*.wav"))
lena_files2 = sorted(glob.glob(lena_segment_dir2+"*.wav"))

lena_10mins_3m_files=sorted(glob.glob(lena_10mins_3m+"*.wav"))
lena_10mins_6m_files=sorted(glob.glob(lena_10mins_6m+"*.wav"))
lena_10mins_9m_files=sorted(glob.glob(lena_10mins_9m+"*.wav"))
lena_10mins_12m_files=sorted(glob.glob(lena_10mins_12m+"*.wav"))
lena_10mins_12m_24m_files=sorted(glob.glob(lena_10mins_12m_24m+"*.wav"))

idp_03m_files = (glob.glob(three_month_dir+"*P.wav"))+(glob.glob(three_month_dir+"*C.wav"))\
                +(glob.glob(three_month_dir+"*F.wav"))+(glob.glob(three_month_dir+"*L.wav"))
idp_06m_files = (glob.glob(six_month_dir+"*P.wav"))+(glob.glob(six_month_dir+"*C.wav"))\
                +(glob.glob(six_month_dir+"*F.wav"))+(glob.glob(six_month_dir+"*L.wav"))
idp_09m_files = (glob.glob(nine_month_dir+"*P.wav"))+(glob.glob(nine_month_dir+"*C.wav"))\
                +(glob.glob(nine_month_dir+"*F.wav"))+(glob.glob(nine_month_dir+"*L.wav"))

idp_03m_mom_files = (glob.glob(three_month_mom_dir+"*.wav"))
idp_06m_mom_files = (glob.glob(six_month_mom_dir+"*.wav"))
idp_09m_mom_files = (glob.glob(nine_month_mom_dir+"*.wav"))

lena_idp_laugh= glob.glob(lena_segment_dir1+"LAU.wav")+glob.glob(lena_segment_dir2+"LAU.wav")+\
                glob.glob(three_month_dir+"*L.wav")+glob.glob(six_month_dir+"*L.wav")+glob.glob(nine_month_dir+"*L.wav")
lena_idp_cry= glob.glob(lena_segment_dir1+"CRY.wav")+glob.glob(lena_segment_dir2+"CRY.wav")+\
                glob.glob(three_month_dir+"*C.wav")+glob.glob(six_month_dir+"*C.wav")+glob.glob(nine_month_dir+"*C.wav")
lena_idp_fuss= glob.glob(lena_segment_dir1+"FUS.wav")+glob.glob(lena_segment_dir2+"FUS.wav")+\
                glob.glob(three_month_dir+"*F.wav")+glob.glob(six_month_dir+"*F.wav")+glob.glob(nine_month_dir+"*F.wav")
lena_idp_babble= glob.glob(lena_segment_dir1+"BAB.wav")+glob.glob(lena_segment_dir2+"BAB.wav")+\
                glob.glob(three_month_dir+"*P.wav")+glob.glob(six_month_dir+"*P.wav")+glob.glob(nine_month_dir+"*P.wav")

# youtube_cry_files =glob.glob(youtube_cry_dir+"fb*.wav")
# youtube_babble_files =glob.glob(youtube_babble_dir+"fb*.wav")
# youtube_fuss_files =glob.glob(youtube_fuss_dir+"fb*.wav")
# youtube_laugh_files =glob.glob(youtube_laugh_dir+"fb*.wav")
# youtube_screech_files =glob.glob(youtube_screech_dir+"fb*.wav")

# idp_03m_files = (glob.glob(three_month_dir+"*.wav"))
# idp_06m_files = (glob.glob(six_month_dir+"*.wav"))
# idp_09m_files = (glob.glob(nine_month_dir+"*.wav"))

idp_03m_mom_face_files = (glob.glob(three_month_mom_face_dir+"*.wav"))
idp_06m_mom_face_files = (glob.glob(six_month_mom_face_dir+"*.wav"))
idp_09m_mom_face_files = (glob.glob(nine_month_mom_face_dir+"*.wav"))

# google_cry_files =glob.glob(google_cry_dir+"*.wav")
# google_laugh_files =glob.glob(google_laugh_dir+"*.wav")
# google_babble_files =glob.glob(google_babble_dir+"*.wav")
#
# google_adult_files =glob.glob(google_adult_dir+"*.wav")
# google_rhy_files =glob.glob(google_rhythmic_dir+"*.wav")
# google_lau_mom_files =glob.glob(google_laugh_mom_dir+"*.wav")
# google_whi_files=glob.glob(google_whisper_dir+"*.wav")
#
# fs_cry_files =glob.glob(fs_cry_dir+"*.wav")
# fs_laugh_files =glob.glob(fs_laugh_dir+"*.wav")
# fs_babble_files =glob.glob(fs_babble_dir+"*.wav")
# fs_fuss_files =glob.glob(fs_fuss_dir+"*.wav")

def get_lena_label(files,label):
    selected_files=[]
    for i in range(len(files)):
        file=files[i]
        filename_list = (file.split('.wav')[0]).split('-')
        curr_label=labels_dict[filename_list[-1]]
        if curr_label==label: selected_files.append(file)
    return selected_files

def write_h5(curr_files,hfX,type_name,hfY=None,hfZ=None,file_description=None,laugh_feature=False):
    labels=[]
    valid_idx=0
    for i in range(len(curr_files)):
        file=curr_files[i].strip('\n')
        print(file)
        #### Child label scheme
        filename_list = (file.split('.wav')[0]).split('_')
        label=labels_dict[filename_list[-1]]
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


files=[]
files+=lena_files1
files+=lena_files2
hf_trainX=h5py.File(prefix+output_dir+"total_embo.h5","w")
hf_trainY=h5py.File(prefix+output_dir+"total_label.h5","w")
f_train=open(prefix+output_dir+"train_files.txt","w")
write_h5(files,hf_trainX,"train_",hf_trainY,None,f_train,False)

### Junk from here
#files=[]
#files+=google_adult_files
# files+=google_rhy_files
# files+=google_lau_mom_files
# files+=google_whi_files
#hf_trainX=h5py.File(output_dir+"train_embo_ADU_entire_segment.h5","w")
#hf_trainY=h5py.File(output_dir+"train_label.h5","w")
#write_h5(files,hf_trainX,"train_")
# hf_trainY.close()
#util.normalize_features(output_dir+"train_embo_ADU_entire_segment.h5",output_dir+"train_embo_ADU_entire_segment_norm.h5")
# training_label=np.asarray([Mom_labels_dict["A"]]*len(google_adult_files)+\
#      [Mom_labels_dict["R"]]*len(google_rhy_files)+\
#      [Mom_labels_dict["L"]]*len(google_lau_mom_files)+\
#      [Mom_labels_dict["W"]]*len(google_whi_files))
# util.write_h5(np.asarray(training_label),output_dir+"train_label.h5")
# util.clear_data(prefix+"idp_mom/train_embo.h5",prefix+"idp_mom/train_label.h5",prefix+"idp_mom/train_embo_clear.h5",prefix+"idp_mom/train_label_clear.h5")
# util.clear_data(prefix+"idp_mom/test_embo.h5",prefix+"idp_mom/test_label.h5",prefix+"idp_mom/test_embo_clear.h5",prefix+"idp_mom/test_label_clear.h5")
# util.clear_data(prefix+"idp_mom/train_label.h5",prefix+"idp_mom/train_label.h5",prefix+"idp_mom/train_embo_clear.h5",prefix+"idp_mom/train_label_clear.h5")
# util.clear_data(prefix+"idp_mom/test_label.h5",prefix+"idp_mom/train_label.h5",prefix+"idp_mom/train_embo_clear.h5",prefix+"idp_mom/train_label_clear.h5")
# util.split_data_mom(prefix+"full_mode/google_audioset_mom/train_embo.h5",prefix+"full_mode/google_audioset_mom/train_label.h5",\
#                     prefix+"full_mode/google_audioset_mom/train_embo_ADU.h5",prefix+"full_mode/google_audioset_mom/train_embo_RHY.h5",\
#                     prefix+"full_mode/google_audioset_mom/train_embo_LAU_MOM.h5",prefix+"full_mode/google_audioset_mom/train_embo_WHI.h5")

#util.merge_hf(prefix+"full_mode/google_audioset_mom/train_embo_ADU_norm.h5",prefix+"full_mode/google_audioset_mom/train_embo_WHI_norm.h5",prefix+"full_mode/google_audioset_mom/train_embo_ADU_WHI_norm.h5")
#util.merge_hf(prefix+"full_mode/idp_mom/train_embo_norm.h5",prefix+"full_mode/google_audioset_mom/train_embo_ADU_WHI_norm.h5",prefix+"full_mode/merged_idp_google_mom/train_embo_ADU_WHI_norm.h5")
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
# print(content.shape)

#files=[]
#files+=lena_files1
#files+=lena_files2
# train_files+=lena_10mins_3m_files[:int(len(lena_10mins_3m_files)*0.8)]
# train_files+=lena_10mins_6m_files[:int(len(lena_10mins_6m_files)*0.8)]
# train_files+=lena_10mins_9m_files[:int(len(lena_10mins_9m_files)*0.8)]
# train_files+=lena_10mins_12m_files[:int(len(lena_10mins_12m_files)*0.8)]
# train_files+=lena_10mins_12m_24m_files[:int(len(lena_10mins_12m_24m_files)*0.8)]
#
# test_files+=lena_10mins_3m_files[int(len(lena_10mins_3m_files)*0.8):]
# test_files+=lena_10mins_6m_files[int(len(lena_10mins_6m_files)*0.8):]
# test_files+=lena_10mins_9m_files[int(len(lena_10mins_9m_files)*0.8):]
# test_files+=lena_10mins_12m_files[int(len(lena_10mins_12m_files)*0.8):]
# test_files+=lena_10mins_12m_24m_files[int(len(lena_10mins_12m_24m_files)*0.8):]
#files+=(lena_10mins_3m_files)
# # files+=(idp_06m_mom_files)
# # files+=(idp_09m_mom_files)
# f_total = open(prefix+output_dir+"total_files.txt","r")
# files=f_total.readlines()
# #files_list = train_test_split(files, train_size=0.8, shuffle=True, random_state=42)
#hf_trainX=h5py.File(prefix+output_dir+"total_laughter.h5","w")
#hf_trainY=h5py.File(prefix+output_dir+"total_label.h5","w")
# #hf_trainZ=h5py.File(prefix+output_dir+"train_age_code.h5","w")
#hf_testX=h5py.File(prefix+output_dir+"test_embo.h5","w")
#hf_testY=h5py.File(prefix+output_dir+"test_label.h5","w")
# #hf_testZ=h5py.File(prefix+output_dir+"test_age_code.h5","w")
#f_train=open(prefix+output_dir+"train_files.txt","w")
#f_test=open(prefix+output_dir+"test_files.txt","w")
#write_h5(files,hf_trainX,"train_",None,None,None,True)
# util.concatenate_hf(prefix+output_dir+"total_embo.h5",prefix+output_dir+"total_laughter.h5",prefix+output_dir+"total_combined.h5")
#write_streaming_h5(test_files,hf_testX,"test_",hf_testY,None,f_test)
#util.normalize_features(prefix+output_dir+"train_embo.h5",prefix+output_dir+"train_embo_norm.h5",)
#util.normalize_features(prefix+output_dir+"test_embo.h5",prefix+output_dir+"test_embo_norm.h5",)

#write_h5(files,hf_trainX,"train_",hf_trainY,None,f_total,False)
#write_h5(files,hf_trainX,"train_",None,None,None,False)
#write_h5(files_list[1],hf_testX,"test_",None,None,None,False)
#write_h5(files_list[0],hf_trainX,"train_",hf_trainY,hf_trainZ,f_train,False)
#write_h5(files_list[1],hf_testX,"test_",hf_testY,hf_testZ,f_test,False)
#util.concatenate_hf(prefix+output_dir+"total_embo.h5",prefix+output_dir+"total_laughter.h5",prefix+output_dir+"total_combined.h5")
#util.concatenate_hf(prefix+output_dir+"train_embo.h5",prefix+output_dir+"train_is09.h5",prefix+output_dir+"train_combined.h5")

#util.normalize_features(prefix+output_dir+"test_is09.h5",prefix+output_dir+"test_is09_norm.h5")
#util.normalize_features(prefix+output_dir+"test_is09.h5",prefix+output_dir+"test_is09_norm.h5")
#util.normalize_features("/home/jialu/disk1/Audio_Speech_Actors_01-24/test_embo.h5","/home/jialu/disk1/Audio_Speech_Actors_01-24/test_embo_norm.h5")
#util.normalize_features("/home/jialu/disk1/Audio_Speech_Actors_01-24/train_embo.h5","/home/jialu/disk1/Audio_Speech_Actors_01-24/train_embo_norm.h5")

#util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_5way/total_combined.h5",prefix+"full_mode/lena_child_5way/total_combined.h5",
#    prefix+"full_mode/idp_child_4way/total_combined.h5",prefix+"full_mode/10mins_child_5way/total_combined.h5")
# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_5way/total_label.h5",prefix+"full_mode/lena_child_5way/total_label.h5",
#     prefix+"full_mode/idp_child_4way/total_label.h5",prefix+"full_mode/10mins_child_5way/total_label.h5")
# util.merge_file(prefix+"full_mode/merged/merged_idp_lena_5way/total_files.txt",prefix+"full_mode/lena_child_5way/total_files.txt",
#     prefix+"full_mode/idp_child_4way/total_files.txt",prefix+"full_mode/10mins_child_5way/total_files.txt")

# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_embo3.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_5way/train_embo3.h5",\
#     prefix+"full_mode/google_audioset/train_embo.h5",
#     prefix+"full_mode/freesound/train_embo.h5")
# util.merge_hf(prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_label3.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_5way/train_label3.h5",\
#     prefix+"full_mode/google_audioset/train_label.h5",\
#     prefix+"full_mode/freesound/train_label.h5")
# util.normalize_features(prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_embo3.h5",\
#             prefix+"full_mode/merged/merged_idp_lena_google_freesound/train_embo3_norm.h5")

# util.merge_hf(prefix+"full_mode/merged/merged_idp_google_mom/train_embo_ADU_WHI_norm.h5",\
#     prefix+"full_mode/idp_mom/train_embo_norm.h5",\
#     prefix+"full_mode/google_audioset_mom/train_embo_ADU_norm.h5",\
#     prefix+"full_mode/google_audioset_mom/train_embo_WHI_norm.h5")
# adu,_=util.read_h5("full_mode/google_audioset_mom/train_embo_ADU_norm.h5")
# whi,_=util.read_h5("full_mode/google_audioset_mom/train_embo_WHI_norm.h5")
# adu=np.zeros((len(adu)))+Mom_labels_dict["A"]
# whi=np.zeros((len(whi)))+Mom_labels_dict["W"]
# total_labels,_=util.read_h5(prefix+"full_mode/idp_mom/train_label.h5")
# total_labels=np.concatenate((total_labels,adu))
# total_labels=np.concatenate((total_labels,whi))
# util.write_h5(total_labels,"full_mode/merged/merged_idp_google_mom/train_label_ADU_WHI.h5")
#util.normalize_features(prefix+"full_mode/merged/merged_idp_google_mom/train_embo_ADU.h5",prefix+"full_mode/merged/merged_idp_google_mom/train_embo_ADU_norm.h5")

# util.convert_5way_to_4way(prefix+"full_mode/merged/merged_idp_lena_5way/total_embo.h5",prefix+"full_mode/merged/merged_idp_lena_5way/total_label.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_4way/total_embo.h5",prefix+"full_mode/merged/merged_idp_lena_4way/total_label.h5",\
#     prefix+"full_mode/merged/merged_idp_lena_5way/total_files.txt",prefix+"full_mode/merged/merged_idp_lena_4way/total_files.txt")

# f=open(prefix+output_dir+"total_files.txt","r")
# total_files=f.readlines()
# kf=KFold(n_splits=3,random_state=1,shuffle=True)
# k=1
# for train_index,test_index in kf.split(total_files):
#     f_train=open(prefix+output_dir+"train_files{}.txt".format(k),"w")
#     f_test=open(prefix+output_dir+"test_files{}.txt".format(k),"w")
#     util.write_file(train_index,total_files,f_train)
#     util.write_file(test_index,total_files,f_test)
#     util.write_selected_dataset(train_index,prefix+output_dir+"total_embo.h5",prefix+output_dir+"train_embo"+str(k)+".h5")
#     util.write_selected_dataset(test_index,prefix+output_dir+"total_embo.h5",prefix+output_dir+"test_embo"+str(k)+".h5")
#     util.write_selected_dataset(train_index,prefix+output_dir+"total_label.h5",prefix+output_dir+"train_label"+str(k)+".h5")
#     util.write_selected_dataset(test_index,prefix+output_dir+"total_label.h5",prefix+output_dir+"test_label"+str(k)+".h5")
#     util.normalize_features(prefix+output_dir+"train_embo"+str(k)+".h5",prefix+output_dir+"train_embo"+str(k)+"_norm.h5")
#     util.normalize_features(prefix+output_dir+"test_embo"+str(k)+".h5",prefix+output_dir+"test_embo"+str(k)+"_norm.h5")
#     k+=1

# files_list_idx = train_test_split(list(range(len(total_files))), train_size=0.8, shuffle=True,random_state=32) #32: 0.7936
# util.write_file(files_list_idx[0],total_files,f_train)
# util.write_file(files_list_idx[1],total_files,f_test)
# util.write_selected_dataset(files_list_idx[0],prefix+output_dir+"total_fbank_norm.h5",prefix+output_dir+"train_fbank_norm.h5")
# util.write_selected_dataset(files_list_idx[1],prefix+output_dir+"total_fbank_norm.h5",prefix+output_dir+"test_fbank_norm.h5")
# util.write_selected_dataset(files_list_idx[0],prefix+output_dir+"total_combined.h5",prefix+output_dir+"train_combined1.h5")
# util.write_selected_dataset(files_list_idx[1],prefix+output_dir+"total_combined.h5",prefix+output_dir+"test_combined1.h5")
# util.write_selected_dataset(files_list_idx[0],prefix+output_dir+"total_embo.h5",prefix+output_dir+"train_embo.h5")
# util.write_selected_dataset(files_list_idx[1],prefix+output_dir+"total_embo.h5",prefix+output_dir+"test_embo.h5")
# util.write_selected_dataset(files_list_idx[0],prefix+output_dir+"total_label.h5",prefix+output_dir+"train_label.h5")
# util.write_selected_dataset(files_list_idx[1],prefix+output_dir+"total_label.h5",prefix+output_dir+"test_label.h5")
# util.normalize_features(prefix+output_dir+"train_combined1_ac_peak.h5",prefix+output_dir+"train_combined1_ac_peak_norm.h5")
# util.normalize_features(prefix+output_dir+"test_combined1_ac_peak.h5",prefix+output_dir+"test_combined1_ac_peak_norm.h5")
# util.normalize_features(prefix+output_dir+"train_combined1.h5",prefix+output_dir+"train_combined1_norm.h5")
# util.normalize_features(prefix+output_dir+"test_combined1.h5",prefix+output_dir+"test_combined1_norm.h5")
# util.normalize_features(prefix+output_dir+"train_embo.h5",prefix+output_dir+"train_embo_norm.h5")
# util.normalize_features(prefix+output_dir+"test_embo.h5",prefix+output_dir+"test_embo_norm.h5")
# util.merge_hf(prefix+"full_mode/merged_idp_lena_age/train_embo.h5",prefix+"full_mode/merged_idp_lena_age/test_embo.h5",prefix+"full_mode/merged_idp_lena_age/total_embo.h5")
# util.merge_hf(prefix+"full_mode/merged_idp_lena_age/train_label.h5",prefix+"full_mode/merged_idp_lena_age/test_label.h5",prefix+"full_mode/merged_idp_lena_age/total_label.h5")
# #util.concatenate_hf(prefix+output_dir+"total_embo.h5",prefix+output_dir+"total_is09.h5",prefix+output_dir+"total_combined.h5")
# # hf_totalX=h5py.File(prefix+output_dir+"total_ac_peak.h5","w")
# # f_total=open(prefix+output_dir+"total_files.txt","r")
# # total_files=f_total.readlines()
# # write_h5(total_files,hf_totalX,"train_",None,None,None,True)
# # util.concatenate_hf(prefix+output_dir+"total_combined.h5",prefix+output_dir+"total_ac_peak.h5",prefix+output_dir+"total_combined_ac_peak.h5")
# util.get_num_classes(prefix+output_dir+"train_label.h5")
# util.get_num_classes(prefix+output_dir+"test_label.h5")
# util.get_num_classes(prefix+output_dir+"total_label.h5")

#util.merge_hf(prefix+"full_mode/merged_idp_lena_age/train_embo1_norm.h5",prefix+"full_mode/idp_mom/train_embo_norm.h5",prefix+"full_mode/merged_chi_mom/train_embo_norm.h5")
#util.merge_hf(prefix+"full_mode/merged_idp_lena_age/test_embo1_norm.h5",prefix+"full_mode/idp_mom/test_embo_norm.h5",prefix+"full_mode/merged_chi_mom/test_embo_norm.h5")
#util.merge_hf(prefix+"full_mode/merged_idp_lena_age/train_label1.h5",prefix+"full_mode/idp_mom/train_label.h5",prefix+"full_mode/merged_chi_mom/train_label.h5")
#util.merge_hf(prefix+"full_mode/merged_idp_lena_age/test_label1.h5",prefix+"full_mode/idp_mom/test_label.h5",prefix+"full_mode/merged_chi_mom/test_label.h5")

# labels=h5py.File(prefix+"full_mode/merged/merged_idp_lena_5way/train_label.h5","r")
# f_total=open(prefix+"full_mode/merged/merged_idp_lena_5way/train_files.txt","r")
# files=f_total.readlines()
# for k in range(10):
#     print(k,files[k],labels[str(k)].value)

#f_total=open(prefix+"full_mode/lena_child_5way/total_files.txt")
