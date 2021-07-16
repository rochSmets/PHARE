
import numpy as np
from scipy.signal import find_peaks


tt = np.load('left2d.npy')
print(type(tt), tt.shape)

nm = 5




tt_ = np.sum(tt, axis=0)

idx = np.argsort(tt_)
kmodes = idx[-nm:]
print(kmodes)

nk = tt.shape[1]



for k in range(nk):
    #if k in range(nk):
    if k in [4, 8, 16, 32, 64]:
        peaks, _ = find_peaks(tt[:,k], height=8, threshold=6, distance=1500, prominence=6)
        w_ = np.argmax(tt[1:,k])
        print("k = ", k, "   --   w = ", w_)
        if k == 64:
            print(tt[:10, k])
        ### if len(peaks) != 0:
        ###     print(k, peaks, tt[peaks, k], tt[0:10, k])

#print(tt.min(), tt.max())
uu = tt.flatten()
vv = uu.sort()

#print(uu[-10:])
