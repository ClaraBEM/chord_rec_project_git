import numpy as np
import librosa
import get_features
import beat_synch

# pitch / key profile is obtained by a training dataset

pitch_profile = [0.2093, 0.0299, 0.0867, 0.0806, 0.0673, 0.0933, 0.0379, 0.1708, 0.0408, 0.0637, 0.0641, 0.0557]

# we are going to use wide chromagram instead treble chromagram for simplicity of implementation

def key_prior_probability(synchronized_chromagram):
    [pitch_num, frames_num] = synchronized_chromagram.shape

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
    return key_prob


def label_prior_probability(labels_number):
    label_prior_prob = np.full(shape=[labels_number,1],fill_value=1/labels_number)
    return label_prior_prob


if __name__=='__main__':
    # path = "testcorto.wav"
    # data, rate = librosa.load(path)
    # beat = get_features.Get_Beat(data, rate)
    # [step, chroma] = get_features.Get_Chromagram(data, rate)
    # beat_chroma = beat_synch.Beat_Synchronization(chroma, beat, step)
    #
    # prob = key_probability(beat_chroma)
    # print(prob)

    prob = label_prior_probability(4)
    print(prob)








