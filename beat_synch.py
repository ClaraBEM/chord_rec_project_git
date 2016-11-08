import librosa
import numpy as np
import get_features
import chord_salience


def beat_synch(matrix, beat, step_size):

    [row, col] = np.shape(matrix)
    #matrix_timestamps = range(0, col) * step_size
    #if len(matrix_timestamps) <= len(beat):
    #    return matrix

    synchronized_matrix = np.empty([row,len(beat)])

    beat_support_1 = np.insert(beat, 0, 0)                          # create 2 copies shifted of beat for computing the intervals
    beat_support_2 = np.insert(beat, len(beat), beat[len(beat)-1])
    beat_interval = np.subtract(beat_support_2, beat_support_1)
    beat_interval = np.delete(beat_interval, beat_interval[-1])     # delete the last interval (not significant)
    samples_per_beat = np.array(beat_interval / step_size)          # to obtain number of samples per beat
    samples_per_beat = np.round(samples_per_beat).astype(int)        # round to floor
    support = np.empty(12)
    prev = 0
    for i in range(1,len(beat)):
        selected_coloumns = range(prev, samples_per_beat[i]*i)
        submatrix = matrix[:,selected_coloumns]
        np.median(submatrix, axis=1, out=support)
        synchronized_matrix[:, i-1] = np.transpose(support)
        prev = samples_per_beat[i]*i

#    linfinite_norm = np.linalg.norm(synchronized_matrix.astype(float), ord=np.inf) # to compute norm I need float
    normalized_synch_matrix = synchronized_matrix #/ linfinite_norm

    return normalized_synch_matrix



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
    synch_chord_salience = beat_synch(chord_salience, beat_timestamp, step)
    print(synch_chord_salience)