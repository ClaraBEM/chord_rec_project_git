import vamp
import librosa
import numpy as np

path = "test.mp3"

data, rate = librosa.load(path)
pitch_salience = vamp.collect(data, rate, "libvamp_essentia:essentia_PitchSalience")

vector = np.array(pitch_salience['vector'])
output = vector[1]
print(output)