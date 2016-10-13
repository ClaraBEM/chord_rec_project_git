import vamp
import librosa

path = "test.mp3"
data, rate = librosa.load(path)
beat = vamp.collect(data, rate, "qm-vamp-plugins:qm-barbeattracker")

list = beat['list']

timestamp = []
label = []

for elem in list:
    timestamp.append(elem['timestamp'])
    label.append(elem['label'])


print(timestamp)
print(label)