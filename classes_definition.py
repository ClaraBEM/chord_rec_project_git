import beat_synch
import get_features
import chord_salience

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
        BeatLabelNode.label = get_features.get_label(data, rate)
        BeatLabelNode.beat = get_features.get_beat(data, rate)


class BassChromagramNode:
    def __init__(self, data, rate, beat):
        BassChromagramNode.step, bass_chromagram = get_features.get_bass_chromagram(data, rate)
        BassChromagramNode.synch_bass_chromagram = beat_synch.beat_synch(bass_chromagram, beat, BassChromagramNode.step)


class ChordSalienceNode:
    def __init__(self,data,rate,beat):
        ChordSalienceNode.step, chord_salience_matrix = chord_salience.get_chord_salience(data, rate)
        ChordSalienceNode.synch_chord_salience = beat_synch.beat_synch(chord_salience_matrix, beat, ChordSalienceNode.step)

