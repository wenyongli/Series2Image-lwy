import numpy as np
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation

data = np.loadtxt("origin.csv", delimiter=',')

data = data.reshape(1, -1)
print(data.shape)
n_sampels, n_timestamps = data.shape #1,400
n_paa = 128
window_size = n_timestamps // n_paa
paa = PiecewiseAggregateApproximation(window_size=window_size)
X_paa = paa.transform(data)[:, :n_paa]
# np.savetxt("./paa_result.csv", X_paa.T, fmt='%.18f')

plt.figure(figsize=(6, 4))
plt.plot(data[0], 'o--', ms=0.1, label='Origin', linewidth=1.)
#plt.plot(np.arange(window_size // 2,
#                  n_timestamps + window_size // 2,
#                  window_size)[:n_paa], X_paa[0], 'o--', ms=2.5, label='PAA', linewidth=1.)
plt.savefig("origin.jpg", pad_inches=0)
plt.show()
