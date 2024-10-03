import ipympl
import numpy as np                          # Importer funksjonalitet fra numpy biblioteket med prefiks "np"
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio


fs, data = wavfile.read('../AudioFile/piano.wav')
xn = data/0x8000
Audio(xn, rate=fs)
