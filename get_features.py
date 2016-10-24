import vamp
import numpy as np
import librosa


def get_chromagram(data, rate):
    #dobbiamo capire quali siano i parametri da dare (normalizzazione, whitening,...)
    #sicuramente dobbiamo normalizzarlo per calcolare la chord salience
    #DA VERIFICARE: il chromagram calcolato da nnls corrisponde al wide chromagram?
    # parameters =
    dictionary = vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="chroma")
    matrix = dictionary['matrix']
    step = float(matrix[0])
    chromagram = np.array(matrix[1]).transpose()
    return [step, chromagram]


def get_bass_chromagram(data, rate):
    dictionary = vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="basschroma")
    matrix = dictionary['matrix']
    step = float(matrix[0])
    basschromagram = np.array(matrix[1]).transpose()
    return [step, basschromagram]


def get_beat(data, rate):
    beat = vamp.collect(data, rate, "qm-vamp-plugins:qm-barbeattracker")
    list = beat['list']
    timestamp = []
    for elem in list:
        timestamp.append(float(elem['timestamp']))

    timestamp = np.array(timestamp)
    return timestamp


def get_label (data , rate):
    beat = vamp.collect(data,rate,"qm-vamp-plugins:qm-barbeattracker")
    list = beat['list']
    label = []
    for elem in list:
        label.append(elem['label'])
    label = np.array(label)
    return label

if __name__=='__main__':
    path = "test.mp3"
    data, rate = librosa.load(path)
    #[step, chroma] = get_chromagram(data, rate)
    #print(chroma)
    beat = get_beat(data,rate)
    print(beat)