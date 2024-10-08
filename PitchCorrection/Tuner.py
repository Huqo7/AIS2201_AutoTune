import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import sounddevice as sd
from scipy.io import wavfile

fs, data = wavfile.read("AudioFile/Beeps.wav")

if len(data.shape) > 1:
    data = data[:, 0]
data = data / max(abs(data))

window_size = int(fs / 80)
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

    new_window = np.fft.irfft(window_freq, n=window_size)
    new_audio[output_index : output_index + window_size] += new_window
    output_index += overlap_size

f_axis, t_axis, S_xx = sig.spectrogram(data, 
                             fs, 
                             window = 'hann', 
                             nperseg = window_size,
                             noverlap = overlap_size,
                             nfft = 2*window_size
                            )

f_axis2, t_axis2, S_xx2 = sig.spectrogram(new_audio, 
                             fs, 
                             window = 'hann', 
                             nperseg = window_size,
                             noverlap = overlap_size,
                             nfft = 2*window_size
                            )

plt.close(1);plt.figure(1, figsize=(10,5))
plt.subplot(2,1,1)
plt.pcolormesh(t_axis,            # Array med tidsstempel. Produsert av spectrogram()
               f_axis,            # Array med frekvenser. Produsert av spectrogram()
               10*np.log10(S_xx),  # Konvertering av spektrogrammatrise til logaritmisk skala
               vmax =-40,         # Høyeste dB-verdi på fargekartet
               vmin = -70,        # Laveste dB-verdi på fargekartet
               shading='auto'
               )
plt.colorbar(aspect=50, label=r'Spectral Density (dB/Hz)')
plt.ylim([0, 3000]) # Du kan justere på grensene i y-aksen dersom du vil inspisere en spesiell del av plottet
#plt.xlim([21.5, 23.5]) # Du kan justere på grensene i x-aksen dersom du vil inspisere en spesiell del av plottet
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.subplot(2,1,2)
plt.pcolormesh(t_axis2,            # Array med tidsstempel. Produsert av spectrogram()
               f_axis2,            # Array med frekvenser. Produsert av spectrogram()
               10*np.log10(S_xx2),  # Konvertering av spektrogrammatrise til logaritmisk skala
               vmax =-40,         # Høyeste dB-verdi på fargekartet
               vmin = -70,        # Laveste dB-verdi på fargekartet
               shading='auto'
               )
plt.colorbar(aspect=50, label=r'Spectral Density (dB/Hz)')
plt.ylim([0, 3000]) # Du kan justere på grensene i y-aksen dersom du vil inspisere en spesiell del av plottet
#plt.xlim([21.5, 23.5]) # Du kan justere på grensene i x-aksen dersom du vil inspisere en spesiell del av plottet
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)
plt.show()

plt.close(2)
plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(data)
plt.subplot(2, 1, 2)
plt.plot(new_audio)
plt.show()

sd.play(new_audio, fs)
sd.wait()