from jpype import *
import numpy as np
import transition_functions
import get_features
import beat_synch
import librosa

key_prior_prob = np.array([ 0.02496951,  0.02973114,  0.01672377,  0.02044016,  0.01486557,  0.02229836,
  0.01486557,  0.,  0.02229836,  0.02044016,  0.00743279,  0.02229836,  0.01114918,  0.00371639,  0.00743279,
  0.01858196,  0.02044016,  0.01858196,
  0.01672377,  0.01114918,  0.01672377,  0.01858196,  0.01736252,  0.01114918,
  0.01718832,  0.,          0.01672377,  0.01022008,  0.01486557,  0.01114918,
  0.00929098,  0.02229836,  0.01347192,  0.01858196,  0.01951106,  0.02229836,
  0.01486557,  0.01858196,  0.01672377,  0.01858196,  0.01114918,  0.02229836,
  0.01858196,  0.00743279,  0.02044016,  0.02229836,  0.00371639,  0.00743279,
  0.01486557,  0.02601475,  0.02044016,  0.01765287,  0.02044016,  0.01858196,
  0.02415655,  0.02787295,  0.02206608,  0.01486557,  0.02880204,  0.01858196], 'double')

key_to_chord = transition_functions.Key_To_Chord()
bass_to_basschroma = transition_functions.Bass_To_Bass_Chromagram()



label_to_chord = np.zeros([12,12])

max_label = 4

classpath = '/Users/Clara/Downloads/bayesserver-7.9/Java/bayesserver-7.9.jar'

startJVM('/Library/Java/JavaVirtualMachines/jdk1.7.0_51.jdk/Contents/Home/jre/lib/jli/libjli.dylib', "-Djava.class.path=%s" %classpath)

bayesServer = JPackage("com.bayesserver")

network = bayesServer.Network()

# invece di costruire gli stati ---> nodo facciamo  stati ---> variabile casuale (discreta o continua)--->nodo

# Create Label Node, depends on argument max_label

beat_states = []

for i in range(1, max_label +1):
    label = str(i)
    beat_states.append(bayesServer.State(label))

# con questo costruttore il tipo della variabile è discreto per default (sto dando in ingresso un numero finito di stati)

label_variable = bayesServer.Variable('label', beat_states)

label_node = bayesServer.Node('label', [label_variable])

# PROVA
# label_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)

# in questo modo dentro la varabile casuale sono salvati gli stati del nodo

# Create Key Node

key_labels = ['C:maj', 'Db:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'B:maj', 'Bb:maj',
             'C:mix', 'Db:mix', 'D:mix', 'Eb:mix', 'E:mix', 'F:mix', 'Gb:mix', 'G:mix', 'Ab:mix', 'A:mix', 'B:mix', 'Bb:mix',
             'C:dor', 'Db:dor', 'D:dor', 'Eb:dor', 'E:dor', 'F:dor', 'Gb:dor', 'G:dor', 'Ab:dor', 'A:dor', 'B:dor', 'Bb:dor',
             'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'B:min', 'Bb:min']

key_states = []
for i in range(1, 12*4):
    key_states.append(bayesServer.State(key_labels[i]))

key_variable = bayesServer.Variable('key', key_states)
key_node = bayesServer.Node('key', [key_variable])
key_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)

# Create Chord Node

chord_labels = ['C:maj', 'Db:maj', 'D:maj','Eb:maj', 'E:maj','F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'B:maj', 'Bb:maj',
                'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'B:min', 'Bb:min']

chord_states = []
for i in range(1, 12*2):
    chord_states.append(bayesServer.State(chord_labels[i]))

chord_variable = bayesServer.Variable('chord', chord_states)
chord_node = bayesServer.Node('chord', [chord_variable])
chord_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create Bass Node

bass_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'G', 'G#', 'A', 'A#','B', 'B#']

bass_states = []
for i in range(1, 12):
    bass_states.append(bayesServer.State(bass_labels[i]))

bass_variable = bayesServer.Variable('bass', bass_states)
bass_node = bayesServer.Node('bass', [bass_variable])
bass_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create Bass Chromagram Node

basschroma_variable = bayesServer.Variable('basschroma', bayesServer.VariableValueType.CONTINUOUS)
basschroma_node = bayesServer.Node('basschroma', [basschroma_variable])
basschroma_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create Salience Node

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

# PROVA
# network.getLinks().add(bayesServer.Link(key_node, key_node, 1))
# network.getLinks().add(bayesServer.Link(chord_node, chord_node, 1))
# network.getLinks().add(bayesServer.Link(chord_node, bass_node, 1))

# Set the distributions among the nodes

#_____________________ PROVE PER I NODI CON LINK TEMPORALI________________________________
# time 0 for nodes that have a temporal link of order 1

# Key node

table = key_node.newDistribution(0).getTable()

#key_states_context = []
#state_context = bayesServer.StateContext(key_states[0], 1)

#for i in range(0, 48):
#    key_states_context.append(bayesServer.StateContext(key_states[0], 0))

#for item in key_states, i in range(0, 48):
#    table.set(key_prior_prob[i], item)

#


#iterator = bayesServer.TableIterator(table, [key_node], [0])
# ____________________________________________________________________________________________#

# NB some relationships are between a node value and probability of transistion of another: in particular
# label --> chord transition
# chord transition --> bass
# key(/mode) ---> chord transition
# non so come fare a definirle!

# distributions for the same time slice

# Chord distribution

table_chord = chord_node.newDistribution(0).getTable()
iterator_chord = bayesServer.TableIterator(table_chord, [key_node, chord_node])
iterator_chord.CopyFrom([key_to_chord])
chord_node.setDistribution(table_chord)

# Bass chromagram distribution

table_bchroma = basschroma_node.newDistribution().getTable()
iterator_bchroma = bayesServer.TableIterator(table, [bass_node, basschroma_node])
iterator_bchroma.CopyFrom([bass_to_basschroma])
basschroma_node.setDistribution(table_bchroma)



print('tumaputtana')

