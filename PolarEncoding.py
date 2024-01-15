import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import  matplotlib.pyplot as plt
def read_wav(audio_path):
    '''使用scipy库下的方法读取音频文件wav,返回时间序列y（数据类型和声道数由文件本身决定）和采样率sr'''
    sr, y = wav.read(filename=audio_path)  # 读取音频文件，返回音频采样率和时间序列
    return y, sr


def write_wav(y, sr, save_path):
    '''使用scipy库下的方法，将时间序列保存为wav格式，y是音频时间序列，sr是采样率，save_path="***.wav"'''
    wav.write(filename=save_path, rate=sr, data=np.array(np.clip(np.round(y), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
data = np.loadtxt("saved.csv", delimiter=',')
origin_serie = data.reshape(1, -1)
#print('origin_serie[0]',origin_serie[0])
# Polar encoding
phi = np.arccos(origin_serie[0])
# Note! The computation of r is not necessary
r = np.linspace(0, 1, len(origin_serie[0]))
font = {
        'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

ax = plt.subplot(111,projection='polar')
#projection = 'polar' 指定为极坐标
# ax.set_title("Polar Encoding",fontdict=font)
ax.set_rticks([0, 1])
ax.set_rlabel_position(-22.5)
ax.plot(phi, r, linewidth=3,color='red')
#第一个参数为角度，第二个参数为极径

ax.grid(True) #是否有网格
plt.savefig("saved_polar.jpg")
plt.show()


# plt.figure(figsize=(6, 4))
# plt.plot(origin_serie[0], 'o--', ms=0.1, label='Origin', linewidth=1.)
# plt.show()
# print( 'phi',phi)
# print( 'r', r)

# Polar encoding
# ax_polar.plot(phi, r)
# ax_polar.set_title("Polar Encoding", fontdict=font)
# ax_polar.set_rticks([0, 1])
# ax_polar.set_rlabel_position(-22.5)
# ax_polar.grid(True)