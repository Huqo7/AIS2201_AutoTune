import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import sounddevice as sd
from scipy.io import wavfile

fs, data = wavfile.read("AudioFile/Actual.wav")

if len(data.shape) > 1:
    data = data[:, 0]
data = data / max(abs(data))

window_size = int(fs / 40)
overlap_size = window_size // 2

window_func = np.hanning(window_size)

total_samples = len(data)

new_audio = np.zeros(total_samples + overlap_size)

output_index = 0
for begin in range(0, total_samples - window_size, overlap_size):
    end = begin + window_size
    window_square = data[begin:end]
    window = window_square * window_func
    window_freq = np.fft.rfft(window)

    frequency = np.argmax(window_freq) * fs / window_size
    n = np.round(12*np.log2(frequency/440))
    correct_frequency = np.pow(2,(n/12))*440

    print(frequency)
    adjuster = 1
    if frequency != 0:
        adjuster = correct_frequency/frequency
    adjusted_window = np.fft.rfft(window, n=int(window_size*adjuster))
    new_window = np.fft.irfft(adjusted_window, n=window_size)
    new_audio[output_index : output_index + window_size] += new_window
    output_index += overlap_size

plt.close(1)
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(data)
plt.subplot(2, 1, 2)
plt.plot(new_audio)
plt.show()

sd.play(new_audio, fs)
sd.wait()
