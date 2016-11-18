import numpy as np
import librosa
import get_features

n_key_modes = 4         # maj mixolidian dorian minor
n_chord_types = 2       # maj min
n_roots = 12
n_keys = n_key_modes * n_roots
n_chords = n_chord_types * n_roots
maj_chord_index = 0
min_chord_index = 1
maj_key_index = 0
mix_key_index = 1
dor_key_index = 2
min_key_index = 3


def Prevkey_To_Nextkey(treble_chromagram):
    # da chordRecognition/ChordDetection/KeyTransModel.m
    # da 4.2.5 Key Node

    # params
    gamma_c = 0.4
    same_key_prob = 1
    parallel_key_bonus = 4
    diatonic_key_malus = 0.15
    circle_fifth_distance = [0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5]

    # 1) first we compute the musicological based key transition
    key_to_key = np.zeros((n_key_modes*n_roots, n_key_modes*n_roots))
    for next_key_mode in range(0, n_key_modes):
        for next_key_root in range(0, n_roots):
            for prev_key_mode in range(0, n_key_modes):
                for prev_key_root in range(0, n_roots):

                    prev_key_index = prev_key_mode * n_roots + prev_key_root
                    next_key_index = next_key_mode * n_roots+ next_key_root
                    prev_maj_key_equivalent = (prev_key_root + 5 * prev_key_mode) % n_roots + 1
                    next_maj_key_equivalent = (next_key_root + 5 * next_key_mode) % n_roots + 1
                    next_eq_to_prev_eq = (next_maj_key_equivalent - prev_maj_key_equivalent) % n_roots

                    if prev_key_index == next_key_index:
                        key_to_key[prev_key_index, next_key_index] = same_key_prob
                    else:
                        key_to_key[prev_key_index, next_key_index] = np.power(gamma_c, (circle_fifth_distance[next_eq_to_prev_eq] + 1))

                    if next_key_root == prev_key_root: #parallel case
                        key_to_key[prev_key_index, next_key_index] = key_to_key[prev_key_index, next_key_index] * parallel_key_bonus
                    else:  #diatonic case
                        key_to_key[prev_key_index, next_key_index] = key_to_key[prev_key_index, next_key_index] * diatonic_key_malus

    # normalization for stochastic row vectors
    key_to_key_prob = np.zeros((n_key_modes*n_roots, n_key_modes*n_roots))
    for i in range(0, n_key_modes * n_roots):
        key_to_key_prob[i, :] = key_to_key[i, :] / np.sum(key_to_key[i, :])

    # 2) now compute key root salience vector= correlation between averaged chromagram and circular shift of key profile
    # ??? non trovo riferimenti su Matlab, non capisco dal pdf

    return key_to_key_prob


def Key_To_Chord():
    # da ChordRecognition/chordDetection/ChordGivenKeyModel.m
    # da 4.2.6 Chord Node

    # params
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

    diatonic_chords = np.zeros((n_chord_types, n_key_modes, n_roots))
    diatonic_chords[maj_chord_index, :, :] = [[tonic_chord, 0, 0, 0, 0, characteristic_chord, 0, 1, 0, 0, 0, 0],
                                            [tonic_chord, 0, 0, 0, 0, 1, 0, 0, 0, 0, characteristic_chord, 0],
                                            [0, 0, 0, 1, 0, characteristic_chord, 0, non_diatonic_primary_dominant, 0, 0, 1, 0],
                                            [0, 0, 0, 1, 0, 0, 0, non_diatonic_primary_dominant, characteristic_chord, 0, 1, 0]]
    diatonic_chords[min_chord_index, :, :] = [[0, 0, 1, 0, 1, 0, 0, 0, 0, characteristic_chord, 0, diminished_chord],
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

    key_to_chord = np.zeros((n_key_modes * n_roots, n_chord_types * n_roots))

    for key_mode in range(0, n_key_modes):
        for key_root in range(0, n_roots):
            for chord_root in range(0, n_roots):
                for chord_type in range(0, n_chord_types):
                    key_index = key_mode * n_roots + key_root
                    chord_index = chord_type * n_roots + chord_root
                    chord_to_key = (chord_root - key_root) % n_roots #+ 1
                    prob = diatonic_chords[chord_type, key_mode, chord_to_key] * diatonic_prob

                    # we transform probability in a tuple and the compute the max
                    if chord_type == maj_chord_index:
                        prob = [prob, secondary_dominant[key_mode, chord_to_key] * secondary_dominant_probability]
                    else:
                        if chord_type == min_chord_index:
                            prob = [prob, secondary_subdominant[key_mode, chord_to_key] * secondary_subdominant_probability]

                key_to_chord[key_index, chord_index] = max(prob)

    # non ho riferimenti nel pdf ma aggiungo una colonna di probabilità che non ci sia alcun accordo
    no_chord_column = no_chord_prob * np.ones((n_keys, 1))
    key_to_chord_no_chord = np.append(arr=key_to_chord, values=no_chord_column, axis=1)

    # substitute all elements = 0 with epsilon
    sel = (key_to_chord_no_chord == 0)
    key_to_chord_no_chord[sel] = epsilon

    key_to_chord_prob = np.zeros(shape=(n_key_modes * n_roots, n_chord_types * n_roots + 1))

    # normalization for statistic row vectors
    for i in range(0, n_chord_types * n_roots + 1):
        key_to_chord_prob[i, :] = key_to_chord_no_chord[i, :] / np.sum(key_to_chord_no_chord[i, :])

    # in Matlab è previsto che possa essere operata una least square regression ma i parametri non la prevedono

    print('tuma')
    return key_to_chord_prob


def Prevchord_Nextchord_To_Bass():
    # da chordRecognition/chordDection/BassGivenChordChangeModel

    #params
    n_chords_and_no_chord = n_chords + 1
    no_chord_col = np.ones((n_roots, 1))
    bass_roots = n_roots
    chord_template = get_features.Get_Chord_Binary_Model()
    chord_template = np.append(arr=chord_template, values=no_chord_col, axis=1)
    bass_prob = np.array([0.8, 0.2])

    chord_to_bass = np.zeros((n_chords_and_no_chord, n_chords_and_no_chord, bass_roots))
    for bass_note in range(0, bass_roots):
        for this_chord in range(0, n_chords):               # in this_cord dimension stop a index before, last column(xbass) will be empty
            for prev_chord in range(0, n_chords_and_no_chord):
                n_chord_notes = sum(chord_template[:, this_chord])

                if this_chord == prev_chord:                # we are staying on same chord
                    if (this_chord % 12) == bass_note:      # bass chord is on root
                        chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[0] * 2 / (n_chord_notes + 1)
                    else:
                        if chord_template[bass_note, this_chord] == 1:  # bass on a chord tone
                            chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[0] * 1 / (n_chord_notes + 1)
                        else:
                            chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[1] * 1 / (13 - n_chord_notes )
                else:                                       # in case of chord changing
                    if (this_chord % 12) == bass_note:
                        chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[0]
                    else:
                        chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[1] / (12 -1)

    # add the no_chord to the last columns: in case there is no chord before all bass roots are equiprobable
    for prev_chord in range(0, n_chords_and_no_chord):
        chord_to_bass[prev_chord, n_chords_and_no_chord - 1, :] = np.ones((1, 1, bass_roots)) / bass_roots

    # normalization for statistic row vectors (CONTROLLA: NORMALIZZO L'ULTIMA DIMENSIONE cioè l'output (le righe nei casi precedenti)
    for prev_chord in range(0, n_chords_and_no_chord):
        for next_chord in range(0,n_chords_and_no_chord):
            chord_to_bass_prob = chord_to_bass[prev_chord, this_chord, :] / sum(chord_to_bass[prev_chord, this_chord, :])

    return chord_to_bass_prob


def Chord_To_Treble_Chromagram():
    # from chordRecognition/chordDetection/TrebleChromaGivenChordModel
    # without key dipendence
    # params
    maj_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    is_in_key = np.array(shape=(n_keys, n_roots), dtype=bool)
    for mode in range(0, n_keys):
        for key_root in range(0, n_root):
            is_in_key[mode * n_roots + key_root, :]

    #non capisco quale dipendenza sto modellando
    return


def Mode_To_Prevchord_Nextchord():


    n_chords_and_no_chord = n_chords + 1
    no_chord_col = np.ones((n_roots, 1))
    chord_template = get_features.Get_Binary_Model()
    chord_template = np.append(arr=chord_template, values=no_chord_col, axis=1)



if __name__=='__main__':
    # path = "testcorto.wav"
    # data, rate = librosa.load(path)
    # [step, chroma] = get_features.get_chromagram(data, rate)
    #
    # trans_prob = key_to_key(chroma)
    #print(trans_prob)

    Prevchord_Nextchord_To_Bass()
