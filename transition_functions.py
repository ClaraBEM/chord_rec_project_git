import numpy as np
import librosa
import get_features

n_key_modes = 4         # maj mixolidian dorian minor
n_chord_types = 2       # maj min
n_root = 12
n_keys = n_key_modes * n_root
n_chords = n_chord_types * n_root
maj_chord_index = 0
min_chord_index = 1
maj_key_index = 0
mix_key_index = 1
dor_key_index = 2
min_key_index = 3


def Prevkey_To_Nextkey(treble_chromagram):
    # da chordRecognition/ChordDetection/KeyTransModel.m
    # da 4.2.5 Key Node

    # key to key parameters
    gamma_c = 0.4
    same_key_prob = 1
    parallel_key_bonus = 4
    diatonic_key_malus = 0.15
    circle_fifth_distance = [0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5]

    # 1) first we compute the musicological based key transition
    key_to_key_prob = np.zeros((n_key_modes*n_root, n_key_modes*n_root))
    for next_key_mode in range(0, n_key_modes):
        for next_key_root in range (0, n_root):
            for prev_key_mode in range(0, n_key_modes):
                for prev_key_root in range(0, n_root):

                    prev_key_index = prev_key_mode * n_root + prev_key_root
                    next_key_index = next_key_mode * n_root+ next_key_root
                    prev_maj_key_equivalent = (prev_key_root + 5*prev_key_mode) % n_root + 1
                    next_maj_key_equivalent = (next_key_root + 5*next_key_mode) % n_root + 1
                    next_eq_to_prev_eq = (next_maj_key_equivalent - prev_maj_key_equivalent) % n_root

                    if prev_key_index == next_key_index:
                        key_to_key_prob[prev_key_index, next_key_index] = same_key_prob
                    else:
                        key_to_key_prob[prev_key_index, next_key_index] = np.power(gamma_c, (circle_fifth_distance[next_eq_to_prev_eq] + 1))

                    if next_key_root == prev_key_root: #parallel case
                        key_to_key_prob[prev_key_index, next_key_index] = key_to_key_prob[prev_key_index, next_key_index] * parallel_key_bonus
                    else:  #diatonic case
                        key_to_key_prob[prev_key_index, next_key_index] = key_to_key_prob[prev_key_index, next_key_index] * diatonic_key_malus

    # normalization for stochastic row vectors
    key_to_key_prob_normalized = np.zeros((n_key_modes*n_root, n_key_modes*n_root))
    for i in range(0, n_key_modes * n_root):
        key_to_key_prob_normalized[i, :] = key_to_key_prob[i, :] / np.sum(key_to_key_prob[i, :])

    # 2) now compute key root salience vector= correlation between averaged chromagram and circular shift of key profile
    # ??? non trovo riferimenti su Matlab, non capisco dal pdf

    return key_to_key_prob_normalized


def Key_To_Chord():
    # da ChordRecognition/chordDetection/ChordGivenKeyModel.m
    # da 4.2.6 Chord Node

    # key to chord parameters
    diatonic_prob = 1
    tonic_chord = 2
    characteristic_chord = 1.2
    non_diatonic_primary_dominant = 0.5
    diminished_chord = 0.7
    epsilon = 0.7
    no_chord_prob = 0.7
    secondary_dominant_probability = 0.7
    secondary_subdominant_probability = 0.7

    # first build the diatonic chords matrix: 3 dimensions, maj min chord in maj mix dor min key

    diatonic_chords = np.zeros((n_chord_types, n_key_modes, n_root))
    diatonic_chords[maj_chord_index,:,:] = [[tonic_chord, 0, 0, 0, 0, characteristic_chord, 0, 1, 0, 0, 0, 0],
                                            [tonic_chord, 0, 0, 0, 0, 1, 0, 0, 0, 0, characteristic_chord, 0],
                                            [0, 0, 0, 1, 0, characteristic_chord, 0, non_diatonic_primary_dominant, 0, 0, 1, 0],
                                            [0, 0, 0, 1, 0, 0, 0, non_diatonic_primary_dominant, characteristic_chord, 0, 1, 0]]
    diatonic_chords[min_chord_index,:,:] = [[0, 0, 1, 0, 1, 0, 0, 0, 0, characteristic_chord, 0, diminished_chord],
                                            [0, 0, 1,  0, diminished_chord, 0, 0, characteristic_chord, 0, 1,  0, 0],
                                            [tonic_chord, 0, characteristic_chord, 0, 0, 0, 0, 1, 0, diminished_chord, 0, 0],
                                            [tonic_chord, 0, diminished_chord, 0, 0,  characteristic_chord, 0, 1, 0, 0, 0, 0]]

    secondary_dominant = np.array([[1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                                [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                                [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]])
    secondary_subdominant = np.array([[0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                                [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]])

    # in prob transition matrix we have keys in row, chords on columns

    key_to_chord_prob = np.zeros((n_key_modes * n_root, n_chord_types * n_root))

    for key_mode in range(0, n_key_modes):
        for key_root in range(0, n_root):
            for chord_root in range(0, n_root):
                for chord_type in range(0, n_chord_types):
                    key_index = key_mode * n_root + key_root
                    chord_index = chord_type * n_root + chord_root
                    chord_to_key = (chord_root - key_root) % n_root #+ 1
                    prob = diatonic_chords[chord_type, key_mode, chord_to_key] * diatonic_prob

                    # we transform probability in a tuple and the compute the max
                    if chord_type == maj_chord_index:
                        prob = [prob, secondary_dominant[key_mode, chord_to_key] * secondary_dominant_probability]
                    else:
                        if chord_type == min_chord_index:
                            prob = [prob, secondary_subdominant[key_mode, chord_to_key] * secondary_subdominant_probability]

                key_to_chord_prob[key_index, chord_index] = max(prob)

    # non ho riferimenti nel pdf ma aggiungo una colonna di probabilità che non ci sia alcun accordo
    no_chord_column = no_chord_prob * np.ones((n_keys, 1))
    key_to_chord_prob = np.append(arr=key_to_chord_prob, values=no_chord_column, axis=1)

    # substitute alle elements = 0 with epsilon
    sel = (key_to_chord_prob == 0)
    key_to_chord_prob[sel] = epsilon

    key_to_chord_prob_normalized = np.zeros(shape=(n_key_modes * n_root, n_chord_types * n_root +1))

    for i in range(0, n_chord_types * n_root + 1):
        key_to_chord_prob_normalized[i, :] = key_to_chord_prob[i, :] / np.sum(key_to_chord_prob[i, :])

    # in Matlab è previsto che possa essere operata una least square regression ma i parametri non la prevedono

    print('tuma')
    return


def Prevchord_Nextchord_To_Bass():
    n_chords_no = n_chords + 1
    no_chord_index = end(n_chords_no)
    bass_root = n_root
    chord_template = Chord_Template()

if __name__=='__main__':
    # path = "testcorto.wav"
    # data, rate = librosa.load(path)
    # [step, chroma] = get_features.get_chromagram(data, rate)
    #
    # trans_prob = key_to_key(chroma)
    #print(trans_prob)

    key_to_chord()
