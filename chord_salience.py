import numpy as np
import vamp
import librosa
import get_features

def get_binary_model():
    N_chord = 24;
    N_pitch = 12;
    chord_binary_model = np.zeros(shape=(N_pitch,N_chord))

    maj = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
    min = np.array([1,0,0,1,0,0,0,1,0,0,0,0])

    for num in range(0,N_chord):
        if num % 2 != 0:
            chord_binary_model[:,num] = np.roll(maj , int(num/2))
        else:

            chord_binary_model[:,num] = np.roll(min , int(num/2))
    return(chord_binary_model)


def get_chord_salience(data , rate):
    [step, chromagram] = get_features.get_wide_chromagram(data , rate)
    chord_template = get_binary_model()
    distance_matrix = np.zeros(chromagram.shape)
    chord_salience = np.zeros(chromagram.shape)
    row = chromagram.shape[0]
    col = chromagram.shape[1]
    for t in range(0,col-1):
        for k in range(0,row-1):
            sum = 0;
            for p in range(0,row-1):
                sum += chord_template[p,k]*np.log10(chord_template[p,k]/chord_salience[p,t]) + chord_salience[p,t] - chord_template[p,k]
            distance_matrix[k,t] = sum

    return distance_matrix

if (__name__=="__main__"):
    path = "test.mp3"
    data, rate = librosa.load(path)
    distance = get_chord_salience(data,rate)




