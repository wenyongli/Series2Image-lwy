# coding=utf-8
"""
对音频数据的几种数据转换方式的实现
"""
import numpy as np
from scipy.fftpack import fft, ifft
import librosa
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd

def read_wav(audio_path):
    '''使用scipy库下的方法读取音频文件wav,返回时间序列y（数据类型和声道数由文件本身决定）和采样率sr'''
    sr, y = wav.read(filename=audio_path)  # 读取音频文件，返回音频采样率和时间序列
    return y, sr


def write_wav(y, sr, save_path):
    '''使用scipy库下的方法，将时间序列保存为wav格式，y是音频时间序列，sr是采样率，save_path="***.wav"'''
    wav.write(filename=save_path, rate=sr, data=np.array(np.clip(np.round(y), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))



audio, sr = read_wav("120630_Figure3_saved.wav")
data = audio.reshape(1,-1)
#print(audio.shape)
#print(data[0].reshape(-1,1))
# plt.figure(figsize=(6, 4))
# plt.plot(data[0] ,'o--', ms=0.1, label='Original', linewidth=1.)
# plt.show()

from sklearn import preprocessing
min_max_scaler = preprocessing.MaxAbsScaler()
data_minMax = min_max_scaler.fit_transform(data[0].reshape(-1,1))
plt.figure(figsize=(6, 4))
plt.plot(data_minMax ,'o--', ms=0.1, label='Saved', linewidth=1.)
#plt.show()

save_data = data_minMax.reshape(1,-1).tolist()[0]
np.savetxt('saved_data.csv',save_data, delimiter = ',')

# print(data_minMax.reshape(1,-1).tolist()[0])

#test.to_csv('./save_data.csv')



