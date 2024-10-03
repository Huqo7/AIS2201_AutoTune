import ipympl
import numpy as np                          # Importer funksjonalitet fra numpy biblioteket med prefiks "np"
import matplotlib.pyplot as plt

from IPython.display import Audio
import sounddevice as sd
from scipy.io import wavfile


fs, data = wavfile.read("../AudioFile/Actual.wav")
xn = data/0x8000


Ts= 1/fs
N = len(xn)

n = np.arange(0, N)
Xn = np.fft.fft(xn, N)
f = np.fft.fftfreq(N, d=1/fs)
Xm = np.abs(Xn)
plt.figure()
plt.plot(f, Xm)
