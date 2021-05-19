import numpy as np
from matplotlib import pyplot as plt
import parselmouth
from scipy.io.wavfile import read, write
import util
from python_speech_features.base import mfcc, fbank, logfbank, get_filterbanks
from python_speech_features.sigproc import magspec, framesig, logpowspec, powspec
import librosa
import librosa.display
from scipy.fft import dct
from pysptk.sptk import lpc,lpc2lsp

frame_dur=0.032
frame_overlap=0.010
prefix="/home/jialu/disk1/infant-vocalize/"
output_dir="full_mode/merged/merged_idp_lena_5way/"
# jitterlocal_idx=1528
# loudness_idx=14 #14 FUS/BAB 16 FOR LAU SCR #20 FOR CRY/FUS
f0_idx=1619
energy_idx=1847 #cry bab
rms_idx=1587 #1631 cry fus #1587 fus bab
mfcc_idx=269 #178 #CRY FUS #826 #FUS LAU #44 #BAB SCR
logMelFreqBand_idx=1063 #442 BAB SCR #354 #LAU BAB #401 #CRY FUS
lspidx=537 #BAB SCR & LAU BAB & CRY SCR & CRY BAB & FUS SCR
formant_idx = 1711
labels_dict={"CRY":0,"FUS":1,"LAU":2,"BAB":3,"SCR":4}


cry_jitter_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20170823_154650_011542_RD_41310.43-41312.09-CRY.wav"
fus_jitter_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20170801_122506_011543_GRP_39990.76-39993.26-FUS.wav"

cry_loudness_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20170801_122506_011543_GRP_39151.68-39154.52-CRY.wav"
fus_loudness_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20170906_165204_011543_LC_20494.58-20497.41-FUS.wav"
lau_loudness_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20171108_143005_023430_GRP_27674.16-27675.32-LAU.wav"
scr_loudness_file="/home/jialu/disk1/lena//CHN_lena_second_set_segments/e20180329_180441_023430_EF_17345.61-17346.21-SCR.wav"
bab_loudness_file="/home/jialu/disk1/lena//CHN_lena_second_set_segments/e20171025_081856_011543_RD_25360.93-25362.13-BAB.wav"

cry_f0_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20170801_122506_011543_GRP_22726.46-22727.06-CRY.wav"
scr_f0_file="/home/jialu/disk1/lena//CHN_lena_second_set_segments/e20180329_180441_023430_EF_16140.37-16141.53-SCR.wav"
lau_f0_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20170906_165204_011543_LC_14206.9-14207.5-LAU.wav"
bab_f0_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20170906_165204_011543_LC_32461.16-32463.080582784187-BAB.wav"
fus_f0_file="/home/jialu/disk1/lena//CHN_lena_first_set_segments/e20170906_165204_011543_LC_24572.16-24575.47-FUS.wav"

loudness_files=[cry_loudness_file,fus_loudness_file,\
            lau_loudness_file,bab_loudness_file,scr_loudness_file]
f0_files=[cry_f0_file,fus_f0_file,\
            lau_f0_file,bab_f0_file,scr_f0_file]

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='viridis')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=5, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    #plt.ylim(0)
    plt.ylabel("intensity [dB]")

def draw_logMelFreqBand(wav_file,label,feature_name,logmelband_nums=[0,1]):
    plt.figure()
    (y, sr) = librosa.load(wav_file)
    plt.subplot(2,1,1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', cmap="viridis")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.ylim([0,8192])
    plt.subplot(2,1,2)
    S=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=8,fmax=8192)
    #print(S.shape)
    logmel_delta=librosa.feature.delta(S)
    print(logmel_delta.shape)
    #M=1125*np.log(1+np.asarray([0,1000,2000,3000,4000,5000,6000,7000])/700)
    # librosa.display.specshow(librosa.power_to_db(S**2),
    #                       sr=sr, y_axis='mel', x_axis='time',cmap='viridis',fmax=8192)
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Log-Power spectrogram')
    colors=["r","g","b"]
    for i in range(len(logmelband_nums)):
        logmelband_num=logmelband_nums[i]
        X=np.linspace(0,len(logmel_delta[logmelband_num]),len(logmel_delta[logmelband_num]))
        plt.plot(X,logmel_delta[logmelband_num],'o',markersize=5,\
            color=colors[i],label="logMelFreqBands(de)[{}]".format(logmelband_num))

    #plt.tight_layout()
    plt.title('logMelFrequencyBands(de)')
    plt.ylabel("Filterbank")
    plt.xlabel("Frame Idx")    
    plt.legend(loc="upper right",prop={"size":8})
    plt.savefig("/home/jialu/voice_quality_plots/v2/logMelFreqBand_de/logmel_012/"+label+"_"+feature_name+".png")

def draw_mfcc(wav_file,label,feature_name,n_mfcc=15):
    (y, sr) = librosa.load(wav_file)
    plt.figure()
    plt.subplot(2,1,1)
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', cmap="viridis")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.ylim([0,8192])
    plt.subplot(2,1,2)
    mfccs = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc)
    plt.imshow(mfccs,origin="lower",aspect="auto",interpolation="nearest")
    plt.colorbar()
    #plt.tight_layout()
    plt.title('MFCC 0-14')
    plt.ylabel("MFCC coefficient index")
    plt.xlabel("Frame Idx")
    plt.savefig("/home/jialu/voice_quality_plots/mfcc/"+label+"_"+feature_name+".png")
    plt.close()

def draw_logmel(wav_file,label,feature_name,logmelband_nums=[4,5]):
    (y, sr) = librosa.load(wav_file)
    rate,data=read(wav_file)
    plt.figure()
    plt.subplot(2,1,1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', cmap="viridis")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.ylim([0,8192])
    plt.subplot(2,1,2)
    feat,_=fbank(y,samplerate=rate,nfft=2048)
    logfbank_energy=np.log(feat).T
    logfbank_energy=logfbank(data,samplerate=rate,nfft=2048)
    colors=["r","g","b"]
    for i in range(len(logmelband_nums)):
        logmelband_num=logmelband_nums[i]
        X=np.linspace(0,len(logfbank_energy[logmelband_num]),len(logfbank_energy[logmelband_num]))
        plt.plot(X,logfbank_energy[logmelband_num],'o',markersize=5,color=colors[i],label="logMelFreqBands[{}]".format(logmelband_num))
    #quantile_value=np.quantile(logfbank_energy[logmelband_num],0.25*2)
    #plt.plot(X,[quantile_value]*len(X),markersize=2,color="r",label="quartile2")
    #quantile_value=np.quantile(logfbank_energy[logmelband_num],0.25*3)
    #plt.plot(X,[quantile_value]*len(X),markersize=2,color="g",label="quartile3")  
    plt.title('logMelFrequencyBands(de)')
    plt.ylabel("Filterbank")
    plt.xlabel("Frame Idx")    
    plt.legend(loc="upper right",prop={"size":8})
    plt.savefig("/home/jialu/voice_quality_plots/v2/logMelFreqBand/"+label+"_"+feature_name+".png")

def draw_dct(k=1):
    plt.figure(figsize=(30,10))
    rate=16000
    filterbanks=get_filterbanks(nfilt=26,nfft=2048,samplerate=rate)
    mel_index=np.argmax(filterbanks,axis=1)
    #mel=librosa.mel_frequencies(n_mels=26,fmin=0,fmax=8192,htk=True)
    mel=(mel_index*rate/2048).astype(int).flatten()
    print(mel)
    plt.plot(mel,np.cos(np.pi*k*(2*np.arange(26)+1)/(2*26)),linewidth=3)
    plt.xlabel("Hz",fontsize=40)
    #plt.xticks([100,500,1000,1500,2000]+list(mel[15:]),fontsize=20)
    selected_indexes=[0,4,8,12,15,19,22,25]
    plt.xticks(mel[selected_indexes],fontsize=40)
    plt.yticks(fontsize=40)
    #plt.title("DCT",fontsize=40)
    plt.savefig("/home/jialu/voice_quality_plots/v2/dct/dct_{}.png".format(k))
    #plt.show()
    
def draw_mfcc_quantile(wav_file,label,feature_name,n_mfcc=15,mfcc_nums=None,quantile_num=2):
    (y, sr) = librosa.load(wav_file)
    plt.figure()
    plt.subplot(2,1,1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', cmap="viridis")
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.ylim([0,8192])
    plt.subplot(2,1,2)
    colors=["r","b"]
    mfccs = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc)
    for mfcc_num in mfcc_nums:    
        X=np.linspace(0,len(mfccs[mfcc_num]),len(mfccs[mfcc_num]))
        plt.plot(X,mfccs[mfcc_num],'o',markersize=5,\
            color=colors[mfcc_num-mfcc_nums[0]],label = "MFCC[{}]".format(mfcc_num))
        #quantile_value=np.average(mfccs[mfcc_num])
        #quantile_value=np.quantile(mfccs[mfcc_num],0.25*2)
        #plt.plot(X,[quantile_value]*len(X),markersize=2,color="r",label="quartile1")
        #quantile_value=np.quantile(mfccs[mfcc_num],0.25*3)
    #plt.plot(X,[quantile_value]*len(X),markersize=2,color="g",label="quartile3")    
    plt.title('MFCC coefficients')
    plt.ylabel("MFCC coefficient value")
    plt.xlabel("Frame Idx")
    plt.legend(loc="upper right")
    #plt.savefig("/home/jialu/voice_quality_plots/mfcc/"+label+"_"+feature_name+"_quartile{}.png".format(quantile_num))
    plt.savefig("/home/jialu/voice_quality_plots/v2/mfcc/"+label+"_"+feature_name+".png")
    plt.close()

def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

def draw_formant(formant):
    time_stamps = formant.xs()
    f1=[]
    for t in time_stamps:
        f1.append(formant.get_value_at_time(formant_number=1,time=t))
    plt.plot(time_stamps,f1,'o', markersize=5, color='w')
    plt.plot(time_stamps,f1,'o', markersize=2,label="formant frequency F1")
    plt.grid(False)
    plt.ylim(0, 5000)
    plt.ylabel("first formant frequency F1 [Hz]")  

def draw_energy(time,energy):
    plt.plot(time, energy, 'o',markersize=2, color='w')
    plt.grid(False)
    #plt.ylim(0)
    plt.ticklabel_format(style="sci",axis="y",scilimits=(0,0))
    plt.ylabel("energy")

def compute_lpc_lsp(wav_file,label,feature_name,lsp_nums=[0,1]):
    y,sr=librosa.load(wav_file)
    #rate,data=read(wav_file)
    plt.figure()
    snd=parselmouth.Sound(wav_file)
    spectrogram=snd.to_spectrogram()
    draw_spectrogram(spectrogram)
    plt.twinx()
    intensity=snd.to_intensity(time_step=0.025,minimum_pitch=75)
    time_stamps=intensity.xs()
    colors=["r","b"]
    for lsp_num in lsp_nums:
        lsp_values=[]
        for i in range(len(time_stamps)-1):
            curr_data=y[int(time_stamps[i]*sr):int((time_stamps[i]+1)*sr)]
            lpc_value=librosa.lpc(curr_data,8)
            lsp=lpc2lsp(lpc_value,otype=0)[lsp_num+1]
            lsp_values.append(lsp/(np.pi)*sr)
        plt.ylabel("LSP frequency [Hz]")
        plt.plot(time_stamps[:-1],lsp_values,'o',markersize=5,\
            color="w")
        plt.plot(time_stamps[:-1],lsp_values,'o', markersize=2, color=colors[lsp_num],label="LSPfreuqnecy[{}]".format(lsp_num))
        #quantile_value=np.mean(lsp_values)
        # plt.plot(time_stamps[:-1],[quantile_value]*len(time_stamps[:-1]),markersize=2,\
        #     color=colors[lsp_num],label="LSPfrequency[{}] mean".format(lsp_num))
        #quantile_value=min(lsp_values)
        #plt.plot(time_stamps[:-1],[quantile_value]*len(time_stamps[:-1]),markersize=2,color="b",label="percentile1.0")
        plt.xlim([snd.xmin, snd.xmax])
        plt.ylim([0, 5000])
    plt.legend(loc="upper right")
    plt.savefig("/home/jialu/voice_quality_plots/v2/lspFrequency/"+label+"_"+feature_name+".png")
    plt.close()

def get_selected_files(idx,files):
    target_files=[]
    for i in idx:
        target_files.append(files[i])
    return target_files

def smooth_signal(sig,M=3): #M window length
    smoothed=np.zeros(len(sig))
    for i in range(len(smoothed)):
        count=0
        for j in range(-M,M+1):
            if i+j>=0 and i+j<len(sig):
                smoothed[i]+=sig[i+j]
                count+=1
        smoothed[i]/=count
    return smoothed

def compute_de(sig,N=2):
    delta=np.zeros(len(sig))
    denom=2*np.sum([n*n for n in range(1,N+1)])
    for i in range(len(sig)):
        for n in range(1,N+1):
            if i-n<0: delta[i]+=n*(sig[i+n]-sig[i])
            elif i+n>len(sig): delta[i]+=n*(sig[i]-sig[i-n])
            else: delta[i]+=n*(sig[i+n]-sig[i-n])
    delta/=denom

def normalize_sig(sig):
    sig=(sig-min(sig))/max(sig)-min(sig)
    return sig

def jitter_local(wav_file,frame_dur=0.032):
    rate,sig=read(wav_file)
    frame_num=(len(sig)-frame_dur*rate)//(frame_overlap*rate)
    jitter=np.zeros((frame_num))
    for i in range(len(frame_num)):
        curr_frame=sig[int(i*frame_overlap*rate):int(i*frame_overlap*rate+frame_dur*rate)]
        jitter[i]=np.mean(np.abs(curr_frame[1:len(curr_frame)]-curr_frame[0:len(curr_frame)-1]))

def compute_intensity(wav_file, label, feature_name):
    snd=parselmouth.Sound(wav_file)
    intensity=snd.to_intensity(time_step=0.025,minimum_pitch=75)
    spectrogram=snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.title(label+" "+feature_name)
    # q2=np.percentile(intensity.values.T,50)
    # q3=np.percentile(intensity.values.T,75)
    # print(q2,q3)
    # plt.plot(intensity.xs(),[q2]*len(intensity.xs()), linewidth=3)
    # plt.plot(intensity.xs(),[q3]*len(intensity.xs()), linewidth=3)
    plt.savefig("/home/jialu/voice_quality_plots/intensity/"+label+"_"+feature_name+".png")

def compute_pitch(wav_file, label, feature_name):
    snd=parselmouth.Sound(wav_file)
    spectrogram=snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    pitch_ceiling=1000
    if label=="SCR": pitch_ceiling=4000
    pitch=snd.to_pitch(time_step=0.025,pitch_ceiling=pitch_ceiling)
    draw_pitch(pitch)
    plt.xlim([snd.xmin, snd.xmax])
    #plt.title(label+" "+feature_name)
    plt.savefig("/home/jialu/voice_quality_plots/f0/"+label+"_"+feature_name+".png")
    plt.close()

def compute_f1(wav_file, label, feature_name):
    snd=parselmouth.Sound(wav_file)
    spectrogram=snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    formant = snd.to_formant_burg(time_step=0.025)
    draw_formant(formant)
    plt.xlim([snd.xmin, snd.xmax])
    #plt.title(label+" "+feature_name)
    plt.legend(loc="upper right")
    plt.savefig("/home/jialu/voice_quality_plots/v2/f1/"+label+"_"+feature_name+".png")
    plt.close()

def compute_energy(wav_file,label,feature_name,RMS=True):
    snd=parselmouth.Sound(wav_file)
    spectrogram=snd.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    time_grid=spectrogram.t_grid()
    energy=np.zeros((len(time_grid)-1))
    for i in range(len(energy)):
        #print(time_grid[i],time_grid[i+1])
        if RMS: energy[i]=snd.get_rms(from_time=time_grid[i],to_time=time_grid[i+1])
        else: energy[i]=snd.get_energy(from_time=time_grid[i],to_time=time_grid[i+1])
    draw_energy(time_grid[1:],energy)
    plt.xlim([snd.xmin, snd.xmax])
    #plt.title(label+" "+feature_name)
    if RMS:
        plt.savefig("/home/jialu/voice_quality_plots/RMS_energy/"+label+"_"+feature_name+".png")
    else:
        plt.savefig("/home/jialu/voice_quality_plots/energy/"+label+"_"+feature_name+".png")

def rms_energy(wav_file,frame_dur=0.032):
    rate,sig=read(wav_file)
    frame_num=(len(sig)-frame_dur*rate)//(frame_overlap*rate)
    rms=np.zeros((frame_num))
    for i in range(len(frame_num)):
        curr_frame=sig[int(i*frame_overlap*rate):int(i*frame_overlap*rate+frame_dur*rate)]
        rms[i]=np.sqrt(1/len(curr_frame)*np.sum(np.square(curr_frame)))

def get_wav_file(feature_idx,target_class,feature_h5,label_h5,files):
    content,_=util.read_h5(feature_h5)
    labels,_=util.read_h5(label_h5)
    f=open(files,"r")
    total_files=f.readlines()
    target_class_index=np.where(labels==target_class)
    target_files=get_selected_files(list(target_class_index[0]),total_files)
    mean=np.mean(content[target_class_index,feature_idx])
    file_idx=np.argsort(np.abs(content[target_class_index,feature_idx-1]-mean))
    all_files=get_selected_files(list(file_idx[0]),target_files)
    target_files=[]
    for i in range(len(all_files)):
        rate,data=read(all_files[i].strip())
        dur=len(data)/rate
        if dur>1 and dur<2:
            target_files.append(all_files[i].strip())
            if len(target_files)>=10: break
    return target_files

#draw_dct(11)
draw_dct(12)
# for key,value in labels_dict.items():
#     if key=="CRY" or key=="BAB" or key=="FUS" or key=="LAU":
#         target_files=get_wav_file(logMelFreqBand_idx,labels_dict[key],\
#                     prefix+output_dir+"train_combined1_multiple_norm.h5",\
#                     prefix+output_dir+"train_label1.h5",\
#                     prefix+output_dir+"train_files1.txt")
#         for i in range(10):
#             print(key+"_"+str(i),target_files[i])
#             #compute_f1(target_files[i],key,"F1_"+str(i))
#             #compute_lpc_lsp(target_files[i],key,"lspFrequency_"+str(i))
#             #compute_pitch(target_files[i],key,"f0_"+str(i))
#             # compute_energy(target_files[i],key,"energy_"+str(i),False)
#             draw_logMelFreqBand(target_files[i],key,"logMelFreqBand_de"+str(i))
#             #draw_mfcc_quantile(target_files[i],key,"mfcc[11]_mfcc[12]_"+str(i),mfcc_nums=[11,12])
#             #draw_logmel(target_files[i],key,"logmel_[4][5]_"+str(i))
