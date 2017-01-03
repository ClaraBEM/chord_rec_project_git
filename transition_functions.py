import numpy as np
import librosa
import get_features


# CONVENZIONE : nelle matrici di transizione / matrici di sigma, l'ultima dimensione è l'ouput, la prima / le prime sono gli input

n_key_modes = 4         # maj mixolidian dorian minor
n_chord_types = 2       # maj min
n_roots = 12
n_keys = n_key_modes * n_roots
n_chords = n_chord_types * n_roots
n_chords_and_no_chord = n_chords + 1
maj_chord_index = 0
min_chord_index = 1
maj_key_index = 0
mix_key_index = 1
dor_key_index = 2
min_key_index = 3


def Prevkey_To_Nextkey():
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

    return key_to_chord_prob


def Prevchord_Nextchord_To_Bass():
    # da chordRecognition/chordDection/BassGivenChordChangeModel

    #params
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
    chord_to_bass_prob = np.zeros([n_chords_and_no_chord, n_chords_and_no_chord, bass_roots])
    for prev_chord in range(0, n_chords_and_no_chord):
        for next_chord in range(0,n_chords_and_no_chord):
            chord_to_bass_prob[prev_chord, next_chord, :] = chord_to_bass[prev_chord, this_chord, :] / sum(chord_to_bass[prev_chord, this_chord, :])

    return chord_to_bass_prob


def Chord_To_Treble_Chromagram():
    # from chordRecognition/chordDetection/TrebleChromaGivenChordModel
    # without key dipendence
    # this dependence will produce the one between chord salience and chord
    # I'm going to compute the parameters for a gaussian distribution
    # in this implementation defaul sigma and chord sigma are both = 0.2 (therefore it's not useful), still
    # it's possible to change manually the parameters

    # params
    treb_chrom_size = n_roots
    key_is_maj = np.ones((1, 12))
    key_is_maj = np.append(arr=key_is_maj, values=np.zeros((1, n_keys - 12)))
    key_is_mix = np.roll(key_is_maj, 12)
    key_is_dor = np.roll(key_is_mix, 12)
    key_is_min = np.roll(key_is_dor, 12)

    chord_is_maj = np.ones((1,12))
    chord_is_maj = np.append(arr=chord_is_maj, values=np.zeros((1, n_chords - 12)))
    chord_is_min = np.roll(chord_is_maj, 12)

    chord_sigmas = [0.2, 0.2, 0.2]
    default_sigma = 0.2

    chord_template = np.transpose(get_features.Get_Chord_Binary_Model())     # for simplicity I tranpose the matrix

    mu = np.zeros((n_chords + 1, treb_chrom_size))
    sigma = np.zeros((n_chords + 1, treb_chrom_size, treb_chrom_size))

    for chord_ind in range(0, n_chords):
        mu[chord_ind, :] = chord_template[chord_ind, :]
        diag = default_sigma * np.ones((1, 12))

        if chord_is_maj[chord_ind]:
            chord_tones_sel = np.array([0, 4, 7])        # accordo maggiore
        else:
            if chord_is_min[chord_ind]:
                chord_tones_sel = np.array([0, 3, 7])    # accordi minore

        sel = np.zeros((1, 12), dtype=bool)
        sel[:,(chord_ind + chord_tones_sel) % 12] = True

        diag[sel] = chord_sigmas
        sigma[chord_ind, : , :] = np.diag(diag)


    # add the no_chord row
    mu[n_chords_and_no_chord - 1, :] = np.ones(treb_chrom_size)
    sigma[n_chords_and_no_chord - 1, :, :] = np.identity(treb_chrom_size)

    return mu, sigma


def Mode_To_Prevchord_Nextchord():
    # from chordRecognition/ ChordDetection/ chordChangeGivenModeBak


    k1 = 10
    k2 = 15
    no_chord_col = np.ones((n_roots, 1))
    chord_template = get_features.Get_Chord_Binary_Model()
    chord_template = np.append(arr=chord_template, values=no_chord_col, axis=1)
    key_template = get_features.Get_Key_Binary_Model()
    mode_to_chord_change = np.zeros((n_key_modes, n_chords_and_no_chord, n_chords_and_no_chord))

    # assign lower weight to transition from tonic, higher weight for transition toward tonic
    # lower weight for transition without tonic

    # major key

    mode_to_chord_change[maj_key_index, 0, 5] = k1              # C -> F
    mode_to_chord_change[maj_key_index, 0, 7] = k1              # C -> G
    mode_to_chord_change[maj_key_index, 7, 0] = k2              # F -> C
    mode_to_chord_change[maj_key_index, 5, 0] = k2              # G -> C

    mode_to_chord_change[maj_key_index, 5, 7] = k1              # F -> C
    mode_to_chord_change[maj_key_index, 7, 5] = k1              # C -> F

    mode_to_chord_change[maj_key_index, 3 + 12 - 1, 7] = k1     # Dmin ( 12 octave + 3 second -1) -> G
    mode_to_chord_change[maj_key_index, 10 + 12 - 1 , 0] = k2   # Am -> C

    # mixolidian key

    mode_to_chord_change[mix_key_index, 0, 5] = k1              # C -> F
    mode_to_chord_change[mix_key_index, 0, 10] = k1             # C -> Bb
    mode_to_chord_change[mix_key_index, 5, 0] = k2              # F -> C
    mode_to_chord_change[mix_key_index, 10, 0] = k2             # Bb -> C

    mode_to_chord_change[mix_key_index, 10, 5] = k1             # Bb -> F
    mode_to_chord_change[mix_key_index, 5, 10] = k1             # F -> Bb
    mode_to_chord_change[maj_key_index, 0, 8 + 12 - 1] = k2     # C -> Gm
    mode_to_chord_change[maj_key_index, 8 + 12 - 1, 0] = k2     # Gm -> C


    #dorian key

    mode_to_chord_change[dor_key_index, 1 + 12 - 1, 5] = k1     # Cm -> F
    mode_to_chord_change[dor_key_index, 1 + 12 - 1, 3] = k1     # Cm -> Eb

    mode_to_chord_change[dor_key_index, 5, 1 + 12 - 1] = k2     # F -> Cm
    mode_to_chord_change[dor_key_index, 3, 1 + 12 - 1] = k2     # Eb -> Cm

    mode_to_chord_change[dor_key_index, 3, 5] = k1              # Eb -> F
    mode_to_chord_change[dor_key_index, 5, 3] = k1              # F -> Eb

    mode_to_chord_change[dor_key_index, 5, 10] = k1             # F -> Bb
    mode_to_chord_change[dor_key_index, 10, 5] = k1             # Bb -> F

    #minor key

    mode_to_chord_change[min_key_index, 1 + 12 - 1, 8] = k1     # Cm -> Ab
    mode_to_chord_change[min_key_index, 1 + 12 - 1, 10] = k1    # Cm -> Bb
    mode_to_chord_change[min_key_index, 8, 1 + 12 - 1] = k2     # Ab -> Cm
    mode_to_chord_change[min_key_index, 1 + 12 - 1, 10] = k2    # Bb -> Cm

    mode_to_chord_change[min_key_index, 8, 10] = k1             # Ab -> Bb
    mode_to_chord_change[min_key_index, 10, 8] = k1             # Bb -> Ab
    mode_to_chord_change[min_key_index, 6, 1 + 12 -1] = k2      # G -> Cm
    mode_to_chord_change[min_key_index, 1 + 12 - 1, 6] = k1     # Cm -> G

    return mode_to_chord_change


def Labels_To_Prevchord_NextchordMOD():
    # dobbiao modificarla in modo che l'output sia una matrice di num_labels* chord * chord
    # se il chord di partenza è uguale a quello di arrivo mettiamo la probabilità = 0?
    #  quando moltiplico per la probabilità data dalla chiave se è uguale a 0 mi mette a 0 tutto
    # se il chord di partenza è diverso da quello di arrivo avrà probabilità che cambia a seconda del label
    a = np.empty((12, 12, n_chords_and_no_chord, n_chords_and_no_chord))
    a[:] = np.NAN
    a[1, 0, :, :] = 0.684189684719655
    a[1, 1, :, :] = 0.158682988716859

    a[2, 0, :, :] = 0.551905308091156
    a[2, 1, :, :] = 0
    a[2, 2, :, :] = 0.0327184489628603
    a[3, 0, :, :] = 0.741551387544580
    a[3, 1, :, :] = 0.0223820305492633
    a[3, 2, :, :] = 0.216058967539676
    a[3, 3, :, :] = 0.0433649453617753


    a[11, :, :, :] = 0

    a[11, 0, :, :] = 0.854166666666667
    a[11, 3, :, :] = 0.0208333333333333
    a[11, 6, :, :] = 0.375000000000000

    for i in range(0, n_chords):
        a[:, :, i, i] = 1
    return a


def Labels_To_Prevchord_Nextchord():
    # dobbiao modificarla in modo che l'output sia una matrice di num_labels* chord * chord
    # se il chord di partenza è uguale a quello di arrivo mettiamo la probabilità = 0?
    #  quando moltiplico per la probabilità data dalla chiave se è uguale a 0 mi mette a 0 tutto
    # se il chord di partenza è diverso da quello di arrivo avrà probabilità che cambia a seconda del label
    a = np.empty((12, 12))
    a[:] = np.NAN
    a[1, 0] = 0.684189684719655
    a[1, 1] = 0.158682988716859
    a[2, 0] = 0.551905308091156
    a[2, 1] = 0
    a[2, 2] = 0.0327184489628603
    a[3, 0] = 0.741551387544580
    a[3, 1] = 0.0223820305492633
    a[3, 2] = 0.216058967539676
    a[3, 3] = 0.0433649453617753


    a[11, :] = 0

    a[11, 0] = 0.854166666666667
    a[11, 3] = 0.0208333333333333
    a[11, 6] = 0.375000000000000
    return a

# Da chiarire: perchè il bass chromagram ha 13 righe? (il treble chromagram ne ha 12)
def Bass_To_Bass_Chromagram():

    Bsize = 12
    BasCsize = 12
    notes = np.identity(Bsize)
    mu = np.zeros((Bsize, BasCsize), dtype=float)
    sigma = np.zeros((Bsize, BasCsize, BasCsize, ))

    for bassInd in range(0, Bsize):
        a = np.transpose(notes[bassInd, :])
        mu[bassInd, :] = np.array(a)
        sigma[bassInd, :, :] = 0.1 * np.identity(BasCsize)
    return mu, sigma


if __name__=='__main__':
    # path = "testcorto.wav"
    # data, rate = librosa.load(path)
    # [step, chroma] = get_features.get_chromagram(data, rate)
    #
    # trans_prob = key_to_key(chroma)
    #print(trans_prob)
    print(Prevchord_Nextchord_To_Bass())
