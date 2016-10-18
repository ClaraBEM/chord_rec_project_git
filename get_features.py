import vamp
import numpy as np


def get_wide_chromagram(data,rate):
    dictionary = vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="chroma")
    matrix = dictionary['matrix']
    step = matrix[0]
    chromagram = np.array(matrix[1])
    return [step , chromagram]


def get_bass_chromagram()
    dictionary = vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="basschroma")
    matrix = dictionary['matrix']
    step = matrix[0]
    basschromagram = np.array(matrix[1])
    return [step, basschromagram]


def get_beat(data , rate)
    beat = vamp.collect(data, rate, "qm-vamp-plugins:qm-barbeattracker")
    list = beat['list']
    timestamp = []
    for elem in list:
        timestamp.append(elem['timestamp'])

    timestamp = np.array(timestamp)
    return(timestamp)

def get_label (data, rate)
    beat = vamp.collect(data,rate,"qm-vamp-plugins:qm-barbeattracker")
    label = [];
    for elem in list:
        label.append(elem['label'])
    label = np.array(label)
    return(label)






