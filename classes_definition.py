import beat_synch
import get_features


class Key:
    def __init__(self, root, mode):
        if mode >= 0 and mode <= 3:
            Key.mode = mode
        if root >= 0 and root <=11:
            Key.root = root


class Chord:
    def __init__(self, triad, pitch):
        if triad >= 0 and triad <= 1:
            Chord.triad = triad
        if pitch >= 0 and pitch <= 11:
            Chord.pitch = pitch


class Beat:
    def __init__(self, data, rate):
        Beat.label = get_features.Get_Label(data, rate)
        Beat.beat = get_features.Get_Beat(data, rate)


class BassChromagram:
    def __init__(self, data, rate, beat):
        BassChromagram.step, bass_chromagram = get_features.Get_Bass_Chromagram(data, rate)
        BassChromagram.synch_bass_chromagram = beat_synch.Beat_Synchronization(bass_chromagram, beat, BassChromagram.step)

class Chromagram:
    def __init__(selfs, data, rate, beat):
        Chromagram.step, Chromagram.chromagram = get_features.Get_Chromagram(data, rate)
        Chromagram.synch_chromagram = beat_synch.Beat_Synchronization(Chromagram.chromagram, beat, Chromagram.step)



class ChordSalience:
    def __init__(self, chromagram, step,  beat):
        ChordSalience.step, ChordSalience.chord_salience_matrix = get_features.Get_Chord_Salience(step, chromagram)
        ChordSalience.synch_chord_salience = beat_synch.Beat_Synchronization(ChordSalience.chord_salience_matrix, beat, ChordSalience.step)

