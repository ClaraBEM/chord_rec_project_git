from jpype import *

max_label = 4
key_probability = []




classpath = '/Users/Clara/Downloads/bayesserver-7.9/Java/bayesserver-7.9.jar'

startJVM('/Library/Java/JavaVirtualMachines/jdk1.7.0_51.jdk/Contents/Home/jre/lib/jli/libjli.dylib', "-Djava.class.path=%s" %classpath)

bayesServer = JPackage("com.bayesserver")

network = bayesServer.Network()


# Create Label Node, depends on argument max_label

beat_states = []

for i in range(1, max_label +1):
    label = str(i)
    beat_states.append(bayesServer.State(label))

label_node = bayesServer.Node('label', beat_states)
label_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)

# Create Key Node

key_labels = ['C:maj', 'Db:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'B:maj', 'Bb:maj',
             'C:mix', 'Db:mix', 'D:mix', 'Eb:mix', 'E:mix', 'F:mix', 'Gb:mix', 'G:mix', 'Ab:mix', 'A:mix', 'B:mix', 'Bb:mix',
             'C:dor', 'Db:dor', 'D:dor', 'Eb:dor', 'E:dor', 'F:dor', 'Gb:dor', 'G:dor', 'Ab:dor', 'A:dor', 'B:dor', 'Bb:dor',
             'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'B:min', 'Bb:min']

key_states = []
for i in range(1, 12*4):
    key_states.append(bayesServer.State(key_labels[i]))

key_node = bayesServer.Node('key', key_states)
key_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create Chord Node

chord_labels = ['C:maj', 'Db:maj', 'D:maj','Eb:maj', 'E:maj','F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'B:maj', 'Bb:maj',
                'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'B:min', 'Bb:min']

chord_states = []
for i in range(1, 12*2):
    chord_states.append(bayesServer.State(chord_labels[i]))

chord_node = bayesServer.Node('chord', chord_states)
chord_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create Bass Node

bass_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'G', 'G#', 'A', 'A#','B', 'B#']

bass_states = []
for i in range(1, 12):
    bass_states.append(bayesServer.State(bass_labels[i]))

bass_node = bayesServer.Node('bass', bass_states)
bass_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create Bass Chromagram Node

basschroma_obs = bayesServer.Variable('bassobs', bayesServer.VariableValueType.CONTINUOUS)
basschroma_node = bayesServer.Node('basschroma', [basschroma_obs])
basschroma_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create Salience Node

salience_obs = bayesServer.Variable('salienceobs', bayesServer.VariableValueType.CONTINUOUS)
salience_node = bayesServer.Node('salience', [salience_obs])
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







print('tumaputtana')

