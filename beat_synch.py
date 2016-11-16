#import librosa
import numpy as np


def Beat_Synchronization(matrix, beat, step_size):

    [row, col] = np.shape(matrix)
    matrix_timestamps = np.arange(0, col)*step_size

    synchronized_matrix = np.zeros([row,len(beat)])

    for i in range(0, len(beat)-1):
        sel = (np.logical_and(matrix_timestamps >= beat[i],matrix_timestamps <= beat[i+1])) #select the column's indeces in the beat intervaò
        support = matrix[:, sel]                                                            #select a submatrix
        for j in range(0, row):
            synchronized_matrix[j, i] = np.median(support[j, :])                            #compute median along rows

    linfinite_norm = np.linalg.norm(synchronized_matrix.astype(float), ord=np.inf)         # to compute norm I need float
    normalized_synch_matrix = synchronized_matrix/(linfinite_norm)

    return normalized_synch_matrix





# if __name__=='__main__':
#     path = "testcorto.wav"
#     data, rate = librosa.load(path)
#     beat_timestamp = get_features.get_beat(data, rate)
#
#