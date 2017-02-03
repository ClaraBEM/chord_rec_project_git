import numpy as np
import librosa
import get_features
import scipy.io as sio

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


# DA CONTROLLARE
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
    key_to_key = np.zeros((n_keys, n_keys))

    # we use for computations indexes+1 (as in Matlab), for indexing we shift them for Python
    for next_key_mode in range(1, n_key_modes+1):
        for next_key_root in range(1, n_roots+1):
            for prev_key_mode in range(1, n_key_modes+1):
                for prev_key_root in range(1, n_roots+1):

                    prev_key_index = (prev_key_mode - 1) * n_roots + prev_key_root
                    next_key_index = (next_key_mode -1) * n_roots + next_key_root
                    prev_maj_key_equivalent = (prev_key_root + 5 * (prev_key_mode - 1)) % n_roots + 1
                    next_maj_key_equivalent = (next_key_root + 5 * (next_key_mode - 1)) % n_roots + 1
                    next_eq_to_prev_eq = (next_maj_key_equivalent - prev_maj_key_equivalent) % n_roots

                    if prev_key_index == next_key_index:
                        key_to_key[prev_key_index - 1, next_key_index - 1] = same_key_prob
                    else:
                        key_to_key[prev_key_index - 1, next_key_index - 1] = np.power(gamma_c, (circle_fifth_distance[next_eq_to_prev_eq + 1 - 1] + 1))

                    if next_key_root == prev_key_root: #parallel case
                        key_to_key[prev_key_index - 1, next_key_index - 1] = key_to_key[prev_key_index - 1, next_key_index - 1] * parallel_key_bonus
                    else:
                        if (prev_maj_key_equivalent - next_maj_key_equivalent) % n_roots == 0:              #diatonic case
                            key_to_key[prev_key_index - 1, next_key_index - 1] = key_to_key[prev_key_index - 1, next_key_index - 1] * diatonic_key_malus

    # normalization for stochastic row vectors
    key_to_key_prob = np.zeros((n_key_modes*n_roots, n_key_modes*n_roots))
    for i in range(0, n_keys):
        key_to_key_prob[i, :] = key_to_key[i, :] / np.sum(key_to_key[i, :])

    # 2) now compute key root salience vector= correlation between averaged chromagram and circular shift of key profile


    return key_to_key_prob


# OK
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

    key_to_chord = np.zeros((n_keys, n_chords))

    for key_mode in range(1, n_key_modes + 1):
        for key_root in range(1, n_roots + 1):
            for chord_root in range(1, n_roots + 1):
                for chord_type in range(1, n_chord_types + 1):
                    prob = 0
                    key_index = (key_mode - 1) * n_roots + key_root
                    chord_index = (chord_type - 1) * n_roots + chord_root

                    chord_to_key = (chord_root - key_root) % n_roots + 1
                    prob = diatonic_chords[chord_type - 1, key_mode - 1, chord_to_key - 1] * diatonic_prob
                    # we transform probability in a tuple and the compute the max
                    if chord_type == (maj_chord_index + 1):
                        prob = [prob, secondary_dominant[key_mode - 1, chord_to_key - 1] * secondary_dominant_probability]
                    else:
                        if chord_type == (min_chord_index + 1):
                            prob = [prob, secondary_subdominant[key_mode - 1, chord_to_key - 1] * secondary_subdominant_probability]
                    key_to_chord[key_index - 1, chord_index - 1] = np.max(prob)

    # no chord probability
    no_chord_column = no_chord_prob * np.ones((n_keys, 1))
    key_to_chord_no_chord = np.append(arr=key_to_chord, values=no_chord_column, axis=1)

    # substitute all elements = 0 with epsilon
    sel = (key_to_chord_no_chord == 0)
    key_to_chord_no_chord[sel] = epsilon
    key_to_chord_prob = np.zeros([n_keys, n_chords_and_no_chord])
    # normalization for statistic row vectors
    for i in range(0, n_keys):
        key_to_chord_prob[i, :] = key_to_chord_no_chord[i, :] / np.sum(key_to_chord_no_chord[i, :])
    return key_to_chord_prob


# NO
def Prevchord_Nextchord_To_Bass():
    # da chordRecognition/chordDetection/BassGivenChordChangeModel

    #params
    no_chord_col = np.ones((1, n_roots))
    bass_roots = n_roots
    chord_template = get_features.Get_Chord_Binary_Model()
    chord_template = np.append(arr=chord_template, values=no_chord_col, axis=0)
    bass_prob = np.array([0.8, 0.2], dtype='float')

    chord_to_bass = np.zeros((n_chords_and_no_chord, n_chords_and_no_chord, bass_roots), dtype='float')
    for bass_note in range(0, bass_roots):
        for this_chord in range(0, n_chords):               # in this_cord dimension stop a index before, last column(xbass) will be empty
            for prev_chord in range(0, n_chords_and_no_chord):
                n_chord_notes = np.sum(chord_template[this_chord, :])

                if this_chord == prev_chord:                # we are staying on same chord
                    if (this_chord % 12) == bass_note:      # bass chord is on root
                        chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[0] * 2 / (n_chord_notes + 1)
                    elif chord_template[this_chord, bass_note] == 1:
                        chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[0] * 1 / (n_chord_notes + 1)
                    else:
                        chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[1] * 1 / (13 - n_chord_notes)
                else:                                       # in case of chord change
                    if (this_chord % 12) == bass_note:
                        chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[0]
                    else:
                        chord_to_bass[prev_chord, this_chord, bass_note] = bass_prob[1] / (12 - 1)

    # add the no_chord to the last columns: in case there is no chord before all bass roots are equiprobable
    for prev_chord in range(0, n_chords_and_no_chord):
        chord_to_bass[prev_chord, n_chords_and_no_chord - 1, :] = np.ones((1, 1, bass_roots)) / bass_roots

     #normalization for statistic row vectors (CONTROLLA: NORMALIZZO L'ULTIMA DIMENSIONE cioè l'output (le righe nei casi precedenti)
    chord_to_bass_prob = np.empty([n_chords_and_no_chord, n_chords_and_no_chord, bass_roots], dtype='float')
    for prev_chord in range(0, n_chords_and_no_chord):
        for next_chord in range(0, n_chords_and_no_chord):
            chord_to_bass_prob[prev_chord, next_chord, :] = chord_to_bass[prev_chord, this_chord, :] / np.sum(chord_to_bass[prev_chord, this_chord, :])

    return chord_to_bass


def Prevchord_Nextchord_to_Bass_MATLAB():
    matrix = sio.loadmat('MATLAB matrici/BassTransProb.mat')
    m = matrix['BassTransProb']
    return m


# def Chord_To_Treble_Chromagram():
#     # from chordRecognition/chordDetection/TrebleChromaGivenChordModel
#     # without key dipendence
#     # this dependence will produce the one between chord salience and chord
#     # I'm going to compute the parameters for a gaussian distribution
#     # in this implementation defaul sigma and chord sigma are both = 0.2 (therefore it's not useful), still
#     # it's possible to change manually the parameters
#
#     # params
#     treb_chrom_size = n_roots
#     key_is_maj = np.ones((1, 12))
#     key_is_maj = np.append(arr=key_is_maj, values=np.zeros((1, n_keys - 12)))
#     key_is_mix = np.roll(key_is_maj, 12)
#     key_is_dor = np.roll(key_is_mix, 12)
#     key_is_min = np.roll(key_is_dor, 12)
#
#     chord_is_maj = np.ones((1,12))
#     chord_is_maj = np.append(arr=chord_is_maj, values=np.zeros((1, n_chords - 12)))
#     chord_is_min = np.roll(chord_is_maj, 12)
#
#     chord_sigmas = [0.2, 0.2, 0.2]
#     default_sigma = 0.2
#
#     chord_template = np.transpose(get_features.Get_Chord_Binary_Model())     # for simplicity I tranpose the matrix
#
#     mu = np.zeros((n_chords + 1, treb_chrom_size))
#     sigma = np.zeros((n_chords + 1, treb_chrom_size, treb_chrom_size))
#
#     for chord_ind in range(0, n_chords):
#         mu[chord_ind, :] = chord_template[chord_ind, :]
#         diag = default_sigma * np.ones((1, 12))
#
#         if chord_is_maj[chord_ind]:
#             chord_tones_sel = np.array([0, 4, 7])        # accordo maggiore
#         else:
#             if chord_is_min[chord_ind]:
#                 chord_tones_sel = np.array([0, 3, 7])    # accordi minore
#
#         sel = np.zeros((1, 12), dtype=bool)
#         sel[:,(chord_ind + chord_tones_sel) % 12] = True
#
#         diag[sel] = chord_sigmas
#         sigma[chord_ind, : , :] = np.diag(diag)
#
#
#     # add the no_chord row
#     mu[n_chords_and_no_chord - 1, :] = np.ones(treb_chrom_size)
#     sigma[n_chords_and_no_chord - 1, :, :] = np.identity(treb_chrom_size)
#
#     return mu, sigma


def Mode_To_Prevchord_Nextchord():
    # from chordRecognition/ ChordDetection/ chordChangeGivenModeBak

    # max diversa la seconda riga
    # min ok
    # mix diversa la seconds riga
    # dor ok


    #k1 = 10
    #k2 = 15
    k1 = 5
    k2 = 10
    k3 = 1
    mode_to_chord_change = np.ones((n_key_modes, n_chords_and_no_chord, n_chords_and_no_chord))

    # assign lower weight to transition from tonic, higher weight for transition toward tonic
    # lower weight for transition without tonic

    #Tonic no change
    tonic_no_change = 10
    # mode_to_chord_change[maj_key_index, 1, 1] = tonic_no_change
    # mode_to_chord_change[mix_key_index, 1, 1] = tonic_no_change
    mode_to_chord_change[dor_key_index, 1 + 12 - 1, 1 + 12 - 1] = tonic_no_change
    mode_to_chord_change[min_key_index, 1 + 12 - 1, 1 + 12 - 1] = tonic_no_change

    # major key

    mode_to_chord_change[maj_key_index, 0, 5] = k1              # C -> F
    mode_to_chord_change[maj_key_index, 0, 7] = k1              # C -> G
    mode_to_chord_change[maj_key_index, 7, 0] = k2              # F -> C
    #mode_to_chord_change[maj_key_index, 5, 0] = k2              # G -> C
    mode_to_chord_change[maj_key_index, 5, 0] = k1  # G -> C

    mode_to_chord_change[maj_key_index, 5, 7] = k1              # F -> C
    mode_to_chord_change[maj_key_index, 7, 5] = k1              # C -> F

    #mode_to_chord_change[maj_key_index, 3 + 12 - 1, 7] = k1     # Dmin ( 12 octave + 3 second -1) -> G
    #mode_to_chord_change[maj_key_index, 10 + 12 - 1 , 0] = k2   # Am -> C
    mode_to_chord_change[maj_key_index, 2, 7] = k3
    mode_to_chord_change[maj_key_index, 4, 10 + 12 - 1] = k3
    mode_to_chord_change[maj_key_index, 10, 3 + 12 - 1] = k3


    # mixolidian key

    mode_to_chord_change[mix_key_index, 0, 5] = k1              # C -> F
    mode_to_chord_change[mix_key_index, 0, 10] = k1             # C -> Bb
    # mode_to_chord_change[mix_key_index, 5, 0] = k2              # F -> C
    mode_to_chord_change[mix_key_index, 5, 0] = k1
    mode_to_chord_change[mix_key_index, 10, 0] = k2             # Bb -> C

    mode_to_chord_change[mix_key_index, 10, 5] = k1             # Bb -> F
    mode_to_chord_change[mix_key_index, 5, 10] = k1             # F -> Bb

    # mode_to_chord_change[maj_key_index, 0, 8 + 12 - 1] = k2     # C -> Gm
    # mode_to_chord_change[maj_key_index, 8 + 12 - 1, 0] = k2     # Gm -> C


    #dorian key

    mode_to_chord_change[dor_key_index, 1 + 12 - 1, 5] = k1     # Cm -> F
    mode_to_chord_change[dor_key_index, 1 + 12 - 1, 3] = k1     # Cm -> Eb

    mode_to_chord_change[dor_key_index, 5, 1 + 12 - 1] = k2     # F -> Cm
    #mode_to_chord_change[dor_key_index, 3, 1 + 12 - 1] = k2     # Eb -> Cm
    mode_to_chord_change[dor_key_index, 3, 1 + 12 - 1] = k1

    mode_to_chord_change[dor_key_index, 3, 5] = k1              # Eb -> F
    mode_to_chord_change[dor_key_index, 5, 3] = k1              # F -> Eb

    # mode_to_chord_change[dor_key_index, 5, 10] = k1             # F -> Bb
    # mode_to_chord_change[dor_key_index, 10, 5] = k1             # Bb -> F

    #minor key

    mode_to_chord_change[min_key_index, 1 + 12 - 1, 8] = k1     # Cm -> Ab
    mode_to_chord_change[min_key_index, 1 + 12 - 1, 10] = k1    # Cm -> Bb
    mode_to_chord_change[min_key_index, 8, 1 + 12 - 1] = k2     # Ab -> Cm
    # mode_to_chord_change[min_key_index, 1 + 12 - 1, 10] = k2    # Bb -> Cm
    mode_to_chord_change[min_key_index, 10, 1 + 12 - 1] = k1

    mode_to_chord_change[min_key_index, 8, 10] = k1             # Ab -> Bb
    mode_to_chord_change[min_key_index, 10, 8] = k1             # Bb -> Ab

    #mode_to_chord_change[min_key_index, 6, 1 + 12 -1] = k2      # G -> Cm
    #mode_to_chord_change[min_key_index, 1 + 12 - 1, 6] = k1     # Cm -> G

    # make the matrix row statistich (normalization)
    mode_to_chord_change_prob = np.empty([n_key_modes, n_chords_and_no_chord, n_chords_and_no_chord], dtype='float')
    for mode in range(0, n_key_modes):
        for prev_chord in range(0, n_chords_and_no_chord):
            for next_chord in range(0, n_chords_and_no_chord):
                if (np.sum(mode_to_chord_change[mode, prev_chord, : ] != 0)):
                    mode_to_chord_change_prob[mode, prev_chord, :] = mode_to_chord_change[mode, prev_chord, :] / np.sum(mode_to_chord_change[mode, prev_chord, :])

   # symmetrize the matrix

    # mode_to_chord_change_prob[maj_key_index, :, :] = (mode_to_chord_change_prob[maj_key_index, :, :] + mode_to_chord_change_prob[maj_key_index, :, :].transpose()) / 2
    # mode_to_chord_change_prob[min_key_index, :, :] = (mode_to_chord_change_prob[min_key_index, :, :] + mode_to_chord_change_prob[min_key_index, :, :].transpose()) / 2
    # mode_to_chord_change_prob[mix_key_index, :, :] = (mode_to_chord_change_prob[mix_key_index, :, :] + mode_to_chord_change_prob[mix_key_index, :, :].transpose()) / 2
    # mode_to_chord_change_prob[dor_key_index, :, :] = (mode_to_chord_change_prob[dor_key_index, :, :] + mode_to_chord_change_prob[dor_key_index, :, :].transpose()) / 2



    return mode_to_chord_change_prob



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
        a[:, :, i, i] = 0
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


def Bass_To_Bass_Chromagram():

    bass_size = n_roots
    bass_c_size = n_roots
    notes = np.identity(bass_size)
    mu = np.zeros((bass_size, bass_c_size), dtype=float)
    sigma = np.zeros((bass_size, bass_c_size, bass_c_size))

    for bass_index in range(0, bass_size):
        a = np.transpose(notes[bass_index, :])
        mu[bass_index, :] = np.array(a)
        sigma[bass_index, :, :] = 0.1 * np.identity(bass_c_size)
    return mu, sigma


def Chord_To_ChordSalience():

    chord_salience_size = n_chords_and_no_chord
    mu = np.identity(chord_salience_size, dtype='float')
    sigma = np.zeros((n_chords_and_no_chord, chord_salience_size, chord_salience_size), dtype='float')
    for i in range(0, n_chords_and_no_chord):
        sigma[i, :, :] = np.identity(chord_salience_size, dtype='float')*0.2
    return [mu, sigma]


def Tot_To_Chord(max_label):

    tot_to_chord = np.zeros([max_label, n_keys, n_chords_and_no_chord, n_chords_and_no_chord])
    key_to_chord = Key_To_Chord()

    label_to_chord = Labels_To_Prevchord_Nextchord()[max_label - 1, :, :, :]

    for l in range(0, max_label):
        for c_next in range(0, n_chords_and_no_chord):
            for c_prev in range(0, n_chords_and_no_chord):
                for k in range(0, n_keys):
                    if c_prev == c_next:  # we use label to transition probability only if the chords are different
                        tot_to_chord[l, k, c_prev, c_next] = key_to_chord[k, c_next]
                    else:
                        tot_to_chord[l, k, c_prev, c_next] = key_to_chord[k, c_next] * label_to_chord[l, c_prev, c_next]

    for l in range(0, max_label):
        for k in range(0, n_keys):
            for c in range(0, n_chords_and_no_chord):
                if sum(tot_to_chord[l, k, c, :]) != 0:
                    tot_to_chord[l, k, c, :] = tot_to_chord[l, k, c, :] / sum(tot_to_chord[l, k, c, :])
                else:  # if the sum is 0
                    tot_to_chord[l, k, c, :] = 1 / n_chords_and_no_chord

    return tot_to_chord

def Tot_To_Chord_MOD(max_label):

    key_to_chord = Key_To_Chord()
    label_to_chord = Labels_To_Prevchord_Nextchord()[max_label - 1]
    type_from_chord = np.concatenate((np.ones(n_roots), 2*np.ones(n_roots)))
    tot_to_chord = np.zeros([n_chords_and_no_chord, max_label, n_keys, n_chords_and_no_chord])

    for this_chord in range(1, n_chords_and_no_chord+1):
        for key in range(1, n_keys+1):
            for label in range(1, max_label+1):
                for prev_chord in range(1, n_chords_and_no_chord+1):

                    key_root = (key - 1) % n_roots + 1
                    prev_chord_root = (prev_chord - 1) % n_roots + 1
                    this_chord_root = (this_chord - 1)% n_roots + 1
                    if prev_chord == n_chords_and_no_chord:
                        prev_chord_to_key = n_chords_and_no_chord
                    else:
                        prev_chord_type = type_from_chord[prev_chord - 1 ]
                        prev_chord_to_key = (prev_chord_root - key_root) % n_roots + 1 + (prev_chord_type - 1) * n_roots

                    if this_chord == n_chords_and_no_chord:
                        this_chord_to_key = n_chords_and_no_chord
                    else:
                        this_chord_type = type_from_chord[this_chord - 1]
                        this_chord_to_key = (this_chord_root - key_root) % n_roots + 1 + (this_chord_type - 1) * n_roots

                    tot_to_chord[prev_chord - 1, label - 1, key - 1, this_chord - 1] = key_to_chord[key - 1, this_chord - 1]

                    if prev_chord == this_chord:
                        tot_to_chord[prev_chord - 1, label - 1, key - 1, this_chord - 1] = tot_to_chord[prev_chord - 1, label - 1, key - 1, this_chord - 1] * (1 - label_to_chord[label - 1])
                    else:
                        tot_to_chord[prev_chord - 1, label - 1, key - 1, this_chord - 1] = tot_to_chord[prev_chord - 1, label - 1, key - 1, this_chord - 1] * (label_to_chord[label - 1])/(n_chords_and_no_chord - 1)

    for l in range(0, max_label):
        for k in range(0, n_keys):
            for c in range(0, n_chords_and_no_chord):
                tot_to_chord[c, l, k, :] = tot_to_chord[c, l, k, :] / sum(tot_to_chord[c, l, k, :])


    return tot_to_chord



#if __name__=='__main__':
    # path = "testcorto.wav"
    # data, rate = librosa.load(path)
    # [step, chroma] = get_features.get_chromagram(data, rate)
    # matrix = sio.loadmat('MATLAB matrici/ChChangeGivenMode.mat')
    # m = matrix['ChCh']
    #
    #
    #print(Mode_To_Prevchord_Nextchord()[mix_key_index, :, :] - m[:, :, 1])