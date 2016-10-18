import vamp
import librosa
import numpy as np
import get_features

path = "test.mp3"
data, rate = librosa.load(path)

#chroma = np.array(vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="chroma"))
#basschroma = np.array(vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="basschroma"))

[rate,chroma] = get_features.get_wide_chromagram(data,rate)
#print(chroma)
dim = chroma.shape
print(dim)
chord_salience = np.zeros(chroma.shape)
row = chroma.shape[0]
print(row)

