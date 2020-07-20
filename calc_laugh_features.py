from scipy.io.wavfile import read
import numpy as np
threshold=5000
frame_length=0.025
frame_skip=0.01

def calc_AC_peak(frames):
    #normalized cross-correlation(NCC)
    peak=-1
    N=len(frames[0])
    for i in range(len(frames)-1):
        x,y=frames[i,:],frames[i+1,:]
        sigma_x,sigma_y=np.std(x),np.std(y)
        curr_peak=1/N*np.sum(x*y)/(sigma_x*sigma_y)
        peak=max(peak,curr_peak)
    return peak

def calc_prob_voiced_frames(frames):
    voiced=0
    for i in range(len(frames)):
        energy=np.sqrt(np.sum(np.square(frames[i])))
        if energy>threshold: voiced+=1
    return voiced/len(frames)

def calc_zero_cross_rate(data):
    return ((data[:-1]*data[1:]<0).sum())/len(data)

def calc_frames(data,frame_length,frame_skip):
    num_frames=1+int((len(data)-frame_length)/frame_skip)
    frames=np.zeros((num_frames,frame_length))
    for i in range(num_frames):
        frames[i,:]=data[i*frame_skip:i*frame_skip+frame_length]
    return frames

def process_data(file):
    rate,data=read(file)
    frames=calc_frames(data,int(frame_length*rate),int(frame_skip*rate))
    peak=calc_AC_peak(frames)
    #prob=calc_prob_voiced_frames(frames)
    #zcr=calc_zero_cross_rate(data)
    #print(peak,prob,zcr)
    #return np.asarray([peak,prob,zcr]).astype(np.float_)
    return np.asarray([peak])
