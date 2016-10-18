import numpy as np
import vamp
import librosa


def get_binary_model():
    N_chord = 24;
    N_pitch = 12;
    chord_binary_model = np.zeros(shape=(N_pitch,N_chord))

    maj = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
    min = np.array([1,0,0,1,0,0,0,1,0,0,0,0])

    for num in range(0,N_chord):
        if num % 2 != 0:
            chord_binary_model[:,num] = np.roll(maj,int(num/2))
        else:

            chord_binary_model[:,num] = np.roll(min,int(num/2))
    return(chord_binary_model)


def get_wide_chromagram(data,rate):
    chroma = np.array(vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="chroma"))
    return chroma


def get_chord_salience(data,rate):
    chromagram = get_wide_chromagram(data,rate)
    #print(chromagram)
    chord_template = get_binary_model()
    #print(chord_template)

    return






