import get_features
import beat_synch
import librosa
import numpy as np

song = '/Users/Stella/Desktop/test.flac'
data, rate = librosa.load(song)

step_chroma, chromagram = get_features.get_chromagram(data, rate)
step_bass_chroma, bass_chromagram = get_features.get_bass_chromagram(data,rate)
beat = get_features.get_beat(data,rate)
labels = get_features.get_label(data,rate)
pitch_salience = get_features.pitch_salience(data,rate)

step_chord_salience, chord_salience = get_features.Get_Chord_Salience(data,rate)

synch_chroma = beat_synch.Beat_Synchronization(chromagram, beat, step_chroma)
synch_bass_chroma = beat_synch.Beat_Synchronization(bass_chromagram, beat, step_bass_chroma)
synch_chord_salience = beat_synch.Beat_Synchronization(chord_salience,beat,step_chord_salience)