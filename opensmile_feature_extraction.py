import os
import csv
import numpy as np
from scipy.io.wavfile import read,write

prefix="/home/jialu"
open_smile_path = prefix+"/opensmile-2.3.0/"
is10="IS10_paraling"
prosodic="prosodyAcf"
emobase_2010="emobase2010"
is09="IS09_emotion"
emobase="emobase"
#output_path=prefix+"/infant-vocalize/youtube_os_4way_features/"
def get_feature_opensmile(wav_path,output_path,type_name="train_"):
    #print(wav_path)
    #rate,wav_data=read(wav_path)
    #if len(wav_data)/rate==0: return np.asarray([])
    output_arff=output_path + type_name+'single_feature.arff'
    cmd = 'cd ' + open_smile_path + ' && ./SMILExtract -C config/' + is09 + \
        '.conf -I ' + "'"+wav_path+"'" + ' -O ' + "'"+output_arff+"'"
    print("Opensmile cmd: ", cmd)
    os.system(cmd)

    reader = csv.reader(open(output_path +type_name+'single_feature.arff','r'))
    rows = list(reader)
    last_line=rows[-1]
    if len(last_line)==0: return np.asarray([])
    data=np.asarray(last_line[1:len(last_line)-1])
    #if len(last_line)==0: return np.asarray([])
    #data = np.asarray(rows[1589:])[:,1:len(last_line)-1]
    #print(len(wav_data)/rate,data.shape)
    return data.astype(np.float32)
