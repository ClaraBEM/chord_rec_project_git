from jpype import *
import numpy as np

import transition_functions
import prior_probabilities

max_label = 4

n_key_modes = 4         # maj mixolidian dorian minor
n_chord_types = 2       # maj min
n_roots = 12
n_keys = n_key_modes * n_roots
n_chords = n_chord_types * n_roots

# PRIOR PROBABILITIES setup for network definition

# prior key probabilities

prior_key = prior_probabilities.simple_prior_key_prob()

# prior chord probabilities I replicate the array to fill each possible combination of the table chord|key label

# potrei modificare la funzione perchè in questo caso ogni elemento è uguale,
prior_chords = prior_probabilities.chord_prior_probability()

for i in range(0, (max_label * n_keys * n_chords)):
    prior_chords = np.append(prior_chords, prior_chords[0])

# TRANSITION PROBABILITIES

key_to_chord = transition_functions.Key_To_Chord()
bass_to_basschroma = transition_functions.Bass_To_Bass_Chromagram()
key_to_key = transition_functions.Prevkey_To_Nextkey()

# BAYES SERRVER LIBRARY SETUP

classpath = '/Users/Clara/Downloads/bayesserver-7.9/Java/bayesserver-7.9.jar'

startJVM('/Library/Java/JavaVirtualMachines/jdk1.7.0_51.jdk/Contents/Home/jre/lib/jli/libjli.dylib', "-Djava.class.path=%s" %classpath)

bayesServer = JPackage("com.bayesserver")

network = bayesServer.Network()


# Create LABEL NODE, depends on argument max_label

beat_states = []

for i in range(1, max_label + 1):
    label = str(i)
    beat_states.append(bayesServer.State(label))

# con questo costruttore il tipo della variabile è discreto per default (sto dando in ingresso un numero finito di stati)

label_variable = bayesServer.Variable('label', beat_states)

label_node = bayesServer.Node('label', [label_variable])


label_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)

# in questo modo dentro la varabile casuale sono salvati gli stati del nodo

# Create KEY NODE

key_labels = ['C:maj', 'Db:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'B:maj', 'Bb:maj',
             'C:mix', 'Db:mix', 'D:mix', 'Eb:mix', 'E:mix', 'F:mix', 'Gb:mix', 'G:mix', 'Ab:mix', 'A:mix', 'B:mix', 'Bb:mix',
             'C:dor', 'Db:dor', 'D:dor', 'Eb:dor', 'E:dor', 'F:dor', 'Gb:dor', 'G:dor', 'Ab:dor', 'A:dor', 'B:dor', 'Bb:dor',
             'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'B:min', 'Bb:min']

key_states = []
for i in range(0, 12*4):
    key_states.append(bayesServer.State(key_labels[i]))

key_variable = bayesServer.Variable('key', key_states)
key_node = bayesServer.Node('key', [key_variable])
key_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)

# Create CHORD NODE

chord_labels = ['C:maj', 'Db:maj', 'D:maj','Eb:maj', 'E:maj','F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'B:maj', 'Bb:maj',
                'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'B:min', 'Bb:min']

chord_states = []
for i in range(0, 12*2):
    chord_states.append(bayesServer.State(chord_labels[i]))

chord_variable = bayesServer.Variable('chord', chord_states)
chord_node = bayesServer.Node('chord', [chord_variable])
chord_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create BASS NODE

bass_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'G', 'G#', 'A', 'A#','B', 'B#']

bass_states = []
for i in range(0, 12):
    bass_states.append(bayesServer.State(bass_labels[i]))

bass_variable = bayesServer.Variable('bass', bass_states)
bass_node = bayesServer.Node('bass', [bass_variable])
bass_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create BASS CHROMAGRAM NODE

basschroma_variable = bayesServer.Variable('basschroma', bayesServer.VariableValueType.CONTINUOUS)
basschroma_node = bayesServer.Node('basschroma', [basschroma_variable])
basschroma_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create SALIENCE NODE

salience_variable = bayesServer.Variable('salience', bayesServer.VariableValueType.CONTINUOUS)
salience_node = bayesServer.Node('salience', [salience_variable])
salience_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)

# Add node to the network

network.getNodes().add(label_node)
network.getNodes().add(key_node)
network.getNodes().add(chord_node)
network.getNodes().add(bass_node)
network.getNodes().add(basschroma_node)
network.getNodes().add(salience_node)


# Create the links within the same time slice

network.getLinks().add(bayesServer.Link(label_node, chord_node))
network.getLinks().add(bayesServer.Link(key_node, chord_node))
network.getLinks().add(bayesServer.Link(chord_node, bass_node))

network.getLinks().add(bayesServer.Link(bass_node, basschroma_node))
network.getLinks().add(bayesServer.Link(chord_node, salience_node))

# Create the links between two consecutive time slice

network.getLinks().add(bayesServer.Link(key_node, key_node, 1))
network.getLinks().add(bayesServer.Link(chord_node, chord_node, 1))
network.getLinks().add(bayesServer.Link(chord_node, bass_node, 1))

#
#
# Set the distributions among the nodes


# time 0 for nodes that have a temporal link of order 1

# Key node distribution

key_table = key_node.newDistribution(0).getTable()
key_iterator = bayesServer.TableIterator(key_table, [key_variable], [java.lang.Integer(0)])
key_iterator.copyFrom(prior_key)
key_node.setDistribution(key_table)

# Chord node distribution
chord_table = chord_node.newDistribution(0).getTable()
chord_iterator = bayesServer.TableIterator(chord_table, [label_variable, key_variable, chord_variable], [java.lang.Integer(0), java.lang.Integer(0), java.lang.Integer(0)])
chord_iterator.copyFrom(prior_chords)
chord_node.setDistribution(chord_table)

# ____________________________________________________________________________________________#

# NB some relationships are between a node value and probability of transition of another: in particular
# label --> chord transition
# chord transition --> bass
# key(/mode) ---> chord transition
# non so come fare a definirle!

# distributions for the same time slice



# Bass chromagram node distribution

#table_bchroma = basschroma_node.newDistribution().getTable()
#iterator_bchroma = bayesServer.TableIterator(table_bchroma, [bass_node])
# iterator_bchroma.CopyFrom([bass_to_basschroma])
# basschroma_node.setDistribution(table_bchroma)

#distributions whithin two time slices

# Key node

key_trans_table = key_node.newDistribution(1).getTable()
key_trans_iterator = bayesServer.TableIterator(key_trans_table, [key_variable, key_variable], [java.lang.Integer(-1), java.lang.Integer(0)])

# astype to cast to float, ravel to linearize the matrix
key_trans_iterator.copyFrom(np.ravel(key_to_key.astype('float')))

# NON FUNZIONA
#key_node.setDistribution(key_trans_table)



print('tumaputtana')