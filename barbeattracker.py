import vamp
import librosa
import numpy as np

path = "test.mp3"
data, rate = librosa.load(path)
beat = vamp.collect(data, rate, "qm-vamp-plugins:qm-barbeattracker")

print(beat)

# list = beat['list']

list = beat['list']

timestamp = [];
label = [];


for elem in list:
    timestamp.append(elem['timestamp'])
    label.append(elem['label'])



timestamp = np.array(timestamp)
label = np.array(label)
print(label)