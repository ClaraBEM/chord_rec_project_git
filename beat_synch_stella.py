import librosa
import numpy as np
import get_features
import chord_salience
import numpy


def beat_synch(matrix, beat, step_size):

    [row, col] = np.shape(matrix)
    #matrix_timestamps = range(0, col) * step_size
    #if len(matrix_timestamps) <= len(beat):
    #    return matrix

    beat_support_1 = np.insert(beat, 0, 0)
    beat_support_2 = np.insert(beat, len(beat), beat[len(beat)-1])
    beat_interval = np.subtract(beat_support_2, beat_support_1)
    beat_interval = np.delete(beat_interval, beat_interval[-1])
    samples_per_beat = np.array(beat_interval / step_size)

    return samples_per_beat

#------------------------------------
def median(matrix, samples_per_beat):

    [row,col] = np.shape(matrix)
    round_spb = np.round(samples_per_beat)
    vect = numpy.zeros(round(col/max(round_spb))*12)

    col2 = round(col/max(round_spb))
    new_matrix = numpy.zeros((row,col2))
    for ri in range(0,row):
        for c in range(0,col2):
            new_matrix[ri,c] = matrix[ri,c]

    x=0
    i=0
    for r in range(0,row,round_spb(i)):
        patch = new_matrix[r,r+round_spb(i)]
        i +=1
        vect = np.insert(vect, x, np.median(patch))
        x += 1

    y=0
    for r in range(0,row+1):
            for c in range(0,col+1):
                new_matrix[r,c] = vect[y]
                y +=1

    return new_matrix



if __name__=='__main__':
    path = "test.mp3"
    data, rate = librosa.load(path)
    beat_timestamp = get_features.get_beat(data, rate)
    [step, chord_salience] = chord_salience.get_chord_salience(data, rate)
    beat_interval = beat_synch(chord_salience, beat_timestamp, step)
    #print(beat_interval)
    new_m = median(chord_salience, beat_interval)
    print(new_m)
