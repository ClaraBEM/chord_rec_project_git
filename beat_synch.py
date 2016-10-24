import librosa
import numpy as np
import get_features
import chord_salience


def beat_synch(matrix, beat, step_size):

    [row, col] = np.shape(matrix)
    #matrix_timestamps = range(0, col) * step_size
    #if len(matrix_timestamps) <= len(beat):
    #    return matrix

    beat_support_1 = np.insert(beat, 0, 0).reshape(len(beat)+1,1)
    beat_support_2 = np.insert(beat, len(beat), 0).reshape(len(beat)+1,1)
    #beat_interval = np.subtract(beat_support_2, beat_support_1)
    #samples_per_beat = np.array(beat_interval / step_size)
    prev_beat = 0
    beat_interval= []
    [beat_interval.append(beat_support_2[i] - beat_support_1[i]) for i in range(0, len(beat))]

    return beat_interval

    #CRASHA e non ho idea del perchè
    

    # int = np.empty([])
    #
    # mat_synch = np.empty([])


#
#     for i in range(0, num_beat):
#             for h in range(0, m):
#                 if matrix(h) >= beat(i) & matrix(h)<= beat(i+1):
#                     int(h) = matrix(h)
#                 mat_synch[:,i] = numpy.median(interval[h])
#             interval = np.zeros([])
#
# #manca normalizzazione


if __name__=='__main__':
    path = "test.mp3"
    data, rate = librosa.load(path)
    beat_timestamp = get_features.get_beat(data, rate)
    [step, chord_salience] = chord_salience.get_chord_salience(data, rate)
    beat_interval = beat_synch(chord_salience, beat_timestamp, step)
    print(beat_interval)
