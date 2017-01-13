import numpy as np
import numpy.matlib as matlib
import librosa
import get_features
import beat_synch

# pitch / key profile is obtained by a training dataset

pitch_profile = [0.2093, 0.0299, 0.0867, 0.0806, 0.0673, 0.0933, 0.0379, 0.1708, 0.0408, 0.0637, 0.0641, 0.0557]

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

# we are going to use wide chromagram instead treble chromagram for simplicity of implementation

def key_prior_probability(synchronized_chromagram):
    [pitch_num, frames_num] = synchronized_chromagram.shape
    key_prob_offset = 1.5

    root_feature = np.zeros([pitch_num, frames_num])

    for i in range(0, pitch_num):
        a = np.roll(pitch_profile, i)

        for j in range(0, frames_num):
            b = synchronized_chromagram[:, j]
            if np.std(b) != 0:
                root_feature[i, j] = np.sum((a - np.mean(a)) * (b - np.mean(b))) / ((12 -1) * np.std(a) * np.std(b))
            else:
                root_feature[i, j] = 0

    root_feature = np.sum(a=root_feature, axis=0)
    root_feature = root_feature + np.abs(min(root_feature))
    key_prob = root_feature / np.sum(root_feature)
    key_prob = (matlib.repmat(key_prob, 1, 4) + key_prob_offset)/np.sum((matlib.repmat(key_prob, 1, 4))+key_prob_offset)

    # NON MI TORNANO LE DIMENSIONI DELLA KEY PROBABILITY per ora uso una a priori generica


def Simple_Prior_Key_Prob():
    prob =  float(1/n_keys)
    key_prob = np.full([n_keys], prob)
    return key_prob


def Prior_Chord_Prob(n_label):
    prob = float(1/n_chords_and_no_chord)
    # I include the elements for each combination of keys: I obtain the probability value for each key/chord combination
    chord_prob = np.full([n_label, n_keys, n_chords_and_no_chord], prob)
    return chord_prob


def Prior_Label_Prob(n_labels):
    prob = float(1/n_labels)
    label_prob = np.full([n_labels], prob)
    return label_prob

def Prior_Bass_Prob():
    prob = float(1/n_roots)
    bass_prob = np.full([n_chords_and_no_chord, n_roots], prob)
    return bass_prob


if __name__=='__main__':
    path = "testcorto.wav"
    data, rate = librosa.load(path)
    beat = get_features.Get_Beat(data, rate)
    [step, chroma] = get_features.Get_Chromagram(data, rate)
    beat_chroma = beat_synch.Beat_Synchronization(chroma, beat, step)

    out = key_prior_probability(beat_chroma)
    print(out)
    # prob = key_probability(beat_chroma)
    # print(prob)