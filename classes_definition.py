import beat_synch
import get_features


class KeyNode:
    def __init__(self, root, mode):
        if mode >= 0 and mode <= 3:
            KeyNode.mode = mode
        if root >= 0 and root <=11:
            KeyNode.root = root


class ChordNode:
    def __init__(self, triad, pitch):
        if triad >= 0 and triad <= 1:
            ChordNode.triad = triad
        if pitch >= 0 and pitch <= 11:
            ChordNode.pitch = pitch


class BeatLabelNode:
    def __init__(self, data, rate):
        BeatLabelNode.label = get_features.Get_Label(data, rate)
        BeatLabelNode.beat = get_features.Get_Beat(data, rate)


class BassChromagramNode:
    def __init__(self, data, rate, beat):
        BassChromagramNode.step, bass_chromagram = get_features.Get_Bass_Chromagram(data, rate)
        BassChromagramNode.synch_bass_chromagram = beat_synch.Beat_Synchronization(bass_chromagram, beat, BassChromagramNode.step)

class ChromagramNode:
    def __init__(selfs, data, rate, beat):
        ChromagramNode.step, chomagram = get_features.Get_Chromagram(data, rate)
        ChromagramNode.synch_chromagram = beat_synch.Beat_Synchronization(chomagram, beat, ChromagramNode.step)



class ChordSalienceNode:
    def __init__(self,data,rate,beat):
        ChordSalienceNode.step, chord_salience_matrix = get_features.Get_Chord_Salience(data, rate)
        ChordSalienceNode.synch_chord_salience = beat_synch.Beat_Synchronization(chord_salience_matrix, beat, ChordSalienceNode.step)

