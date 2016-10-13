import vamp
import librosa
import numpy as np

path = "test.mp3"
data, rate = librosa.load(path)

chroma = np.array(vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="chroma"))
basschroma = np.array(vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="basschroma"))

print(chroma)
print(basschroma)
