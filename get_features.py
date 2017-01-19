import vamp
import numpy as np
import librosa


def Get_Chromagram(data, rate):
    # dobbiamo capire quali siano i parametri da dare (normalizzazione, whitening,...)
    # sicuramente dobbiamo normalizzarlo per calcolare la chord salience
    # DA VERIFICARE: il chromagram calcolato da nnls corrisponde al wide chromagram?
    # parameters =
    dictionary = vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="chroma")
    matrix = dictionary['matrix']
    step = float(matrix[0])
    chromagram = np.array(matrix[1]).transpose()
    return step, chromagram


def Get_Bass_Chromagram(data, rate):
    dictionary = vamp.collect(data, rate, "nnls-chroma:nnls-chroma", output="basschroma")
    matrix = dictionary['matrix']
    step = float(matrix[0])
    basschromagram = np.array(matrix[1]).transpose()
    return step, basschromagram


def Get_Beat(data, rate):
    beat = vamp.collect(data, rate, "qm-vamp-plugins:qm-barbeattracker", output='beats')
    list = beat['list']
    timestamp = []
    for elem in list:
        timestamp.append(float(elem['timestamp']))
    timestamp = np.array(timestamp)
    return timestamp


def Get_Label(data, rate):
    beat = vamp.collect(data,rate,"qm-vamp-plugins:qm-barbeattracker", output='beats')
    list = beat['list']
    label = []
    for elem in list:
        label.append(elem['label'])
    label = np.array(label)
    return label


def Get_Pitch_Salience(data,rate):
    pitch_salience = vamp.collect(data, rate, "libvamp_essentia:essentia_PitchSalience")
    vector = np.array(pitch_salience['vector'])
    output = vector[1]
    return output


def Get_Chord_Binary_Model():
    N_chord = 24
    N_pitch = 12
    chord_binary_model = np.zeros(shape=(N_pitch, N_chord))

    maj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    min = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    for num in range(0, N_chord):
        if num % 2 != 0:
            chord_binary_model[:, num] = np.roll(maj, int(num / 2))
        else:
            chord_binary_model[:, num] = np.roll(min, int(num / 2))
    return chord_binary_model


def Get_Key_Binary_Model():
    # build the model for the keys
    # output is 48x12 matrix, each line represents the presence/no presence of the grade for the scale
    # maj mix dor min is the mode order

    maj_key_template = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    mix_key_template = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    min_key_template = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    dor_key_template = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    maj_keys = np.zeros((12, 12))
    mix_keys = np.zeros((12, 12))
    min_keys = np.zeros((12, 12))
    dor_keys = np.zeros((12, 12))
    for i in range(0, 12):
        maj_keys[i, :] = np.roll(maj_key_template, i)
        mix_keys[i, :] = np.roll(mix_key_template, i)
        min_keys[i, :] = np.roll(min_key_template, i)
        dor_keys[i, :] = np.roll(dor_key_template, i)
    key_template = np.concatenate((maj_keys, mix_keys, dor_keys, min_keys), axis=0)
    return key_template


# CONTROLLLAAAAAAAAAAAAA se conviene mandare in input chromagram direttamente
def Get_Chord_Salience(data, rate):
    [step, chromagram] = Get_Chromagram(data, rate)
    n_chords = 24
    eps = 2.2204e-16
    [row, col] = chromagram.shape
    chord_template = Get_Chord_Binary_Model()
    distance_matrix = np.zeros([n_chords, col])
    chord_salience = np.zeros([n_chords, col])

    for t in range(0, col):
        for k in range(0, n_chords):
            sum = 0
            for p in range(0, row):
                # se il numeratore del logaritmo è = 0 log(0)= -inf quindi lo sostituiamo con un epsilon
                if chord_template[p, k] == 0:
                        chord_template_elem = eps
                else:
                    chord_template_elem = chord_template[p, k]
                if chromagram[p, t] == 0:
                    chromagram_elem = eps
                else:
                    chromagram_elem = chromagram[p, t]
                sum = sum + chord_template_elem * np.log10(chord_template_elem / chromagram_elem) + chromagram_elem - chord_template_elem
            distance_matrix[k, t] = sum
        chord_salience[:, t] = min(distance_matrix[:, t]) / distance_matrix[:, t]

        # MANCA MEDIAN FILTER
        # return chord_salience
    return [step, chord_salience]



if __name__=='__main__':

    path = "test.mp3"
    data, rate = librosa.load(path)
    [step, cs] = Get_Chord_Salience(data, rate)
    print(cs.shape)