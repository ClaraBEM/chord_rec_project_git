from jpype import *
import numpy as np

import transition_functions
import prior_probabilities
import classes_definition
import librosa

max_label = 4

n_key_modes = 4         # maj mixolidian dorian minor
n_chord_types = 2       # maj min
n_roots = 12
n_keys = n_key_modes * n_roots
n_chords = n_chord_types * n_roots
n_chords_and_no_chord = n_chords + 1


# PRIOR PROBABILITIES setup for network definition

# prior key probabilities

prior_key = prior_probabilities.Simple_Prior_Key_Prob()

# prior chord probabilities
prior_chords = prior_probabilities.Prior_Chord_Prob(max_label)

# prior label prob

prior_label = prior_probabilities.Prior_Label_Prob(max_label)

# prior bass prob

prior_bass = prior_probabilities.Prior_Bass_Prob()


##### PROVA CON NUOVA RETE (da includere in prior probabilities)
#for i in range(0, (max_label * n_keys * n_chords)):
#    prior_chords = np.append(prior_chords, prior_chords[0])
#####


# TRANSITION PROBABILITIES

# for the chord transition probabilities I combine the transition functions and then normalize them
# forse potremmo spostare questo in prior_probabilities.py

key_to_chord = transition_functions.Key_To_Chord()

label_to_chord = transition_functions.Labels_To_Prevchord_NextchordMOD()[max_label - 1, :, :, :]

tot_to_chord = np.zeros([max_label, n_keys, n_chords_and_no_chord, n_chords_and_no_chord])

for l in range(0, max_label):
    for c_next in range(0, n_chords_and_no_chord):
        for c_prev in range(0, n_chords_and_no_chord):
            for k in range(0, n_keys):
                if c_prev == c_next:        # we use label to transition probability only if the chords are different
                    tot_to_chord[l, k, c_prev, c_next] = key_to_chord[k, c_next]
                else:
                    tot_to_chord[l, k, c_prev, c_next] = key_to_chord[k, c_next] * label_to_chord[l, c_prev, c_next]

for l in range(0, max_label):
    for k in range(0, n_keys):
        for c in range(0, n_chords_and_no_chord):
            if (sum(tot_to_chord[l, k, c, :]) != 0):
                tot_to_chord[l, k, c, :] = tot_to_chord[l, k, c, :] / sum(tot_to_chord[l, k, c, :])
            else:                                   # if the sum is 0
                tot_to_chord[l, k, c, :] = 1/n_chords_and_no_chord

[bass_to_basschroma_mu, bass_to_basschroma_sigma] = transition_functions.Bass_To_Bass_Chromagram()
[chord_to_chordsalience_mu, chord_to_chordsalience_sigma] = transition_functions.Chord_To_ChordSalience()

key_to_key = transition_functions.Prevkey_To_Nextkey()

chordtransition_to_bass = transition_functions.Prevchord_Nextchord_To_Bass()


# BAYES SERVER LIBRARY SETUP

classpath = '/Users/Clara/Downloads/bayesserver-7.10/Java/bayesserver-7.10.jar'

startJVM('/Library/Java/JavaVirtualMachines/jdk1.7.0_51.jdk/Contents/Home/jre/lib/jli/libjli.dylib', "-Djava.class.path=%s" %classpath)

bayesServer = JPackage("com.bayesserver")

network = bayesServer.Network()


# Create LABEL NODE, depends on argument max_label

beat_states = []
for i in range(1, max_label + 1):
    label = str(i)
    beat_states.append(bayesServer.State(label))
label_variable = bayesServer.Variable('label', beat_states)
label_node = bayesServer.Node('label', [label_variable])
label_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create KEY NODE

key_labels = ['C:maj', 'Db:maj', 'D:maj', 'Eb:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'B:maj', 'Bb:maj',
            'C:mix', 'Db:mix', 'D:mix', 'Eb:mix', 'E:mix', 'F:mix', 'Gb:mix', 'G:mix', 'Ab:mix', 'A:mix', 'B:mix', 'Bb:mix',
            'C:dor', 'Db:dor', 'D:dor', 'Eb:dor', 'E:dor', 'F:dor', 'Gb:dor', 'G:dor', 'Ab:dor', 'A:dor', 'B:dor', 'Bb:dor',
            'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'B:min', 'Bb:min']
key_states = []
for i in range(0, n_keys):
    key_states.append(bayesServer.State(key_labels[i]))
key_variable = bayesServer.Variable('key', key_states)
key_node = bayesServer.Node('key', [key_variable])
key_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create CHORD NODE

chord_labels = ['C:maj', 'Db:maj', 'D:maj','Eb:maj', 'E:maj','F:maj', 'F#:maj', 'G:maj', 'Ab:maj', 'A:maj', 'B:maj', 'Bb:maj',
                'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'B:min', 'Bb:min', 'no_chord']
chord_states = []
for i in range(0, n_chords_and_no_chord):
    chord_states.append(bayesServer.State(chord_labels[i]))
chord_variable = bayesServer.Variable('chord', chord_states)
chord_node = bayesServer.Node('chord', [chord_variable])
chord_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create BASS NODE

bass_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'G', 'G#', 'A', 'A#','B', 'B#']
bass_states = []
for i in range(0, n_roots):
    bass_states.append(bayesServer.State(bass_labels[i]))
bass_variable = bayesServer.Variable('bass', bass_states)
bass_node = bayesServer.Node('bass', [bass_variable])
bass_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create BASS CHROMAGRAM NODE
# create a continue variable for each row of the chromagram

basschroma_rows = 12
basschroma_labels = bass_labels
basschroma_variables = []
for i in range(0, basschroma_rows):
    basschroma_variables.append(bayesServer.Variable(basschroma_labels[i], bayesServer.VariableValueType.CONTINUOUS))
basschroma_node = bayesServer.Node('basschroma', basschroma_variables)
basschroma_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


# Create SALIENCE NODE

salience_rows = n_chords_and_no_chord
salience_labels = chord_labels
salience_variables = []
for i in range(0, salience_rows):
    salience_variables.append( bayesServer.Variable(salience_labels[i], bayesServer.VariableValueType.CONTINUOUS))
salience_node = bayesServer.Node('salience', salience_variables)
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
network.getLinks().add(bayesServer.Link(chord_node, bass_node))
network.getLinks().add(bayesServer.Link(key_node, chord_node))
network.getLinks().add(bayesServer.Link(bass_node, basschroma_node))
network.getLinks().add(bayesServer.Link(chord_node, salience_node))

# Create the links between two consecutive time slice

network.getLinks().add(bayesServer.Link(key_node, key_node, 1))
network.getLinks().add(bayesServer.Link(chord_node, chord_node, 1))
network.getLinks().add(bayesServer.Link(chord_node, bass_node, 1))



### Set the distributions among the nodes


# DISTRIBUTION AT TIME 0

# Key node distribution

key_table = key_node.newDistribution(0).getTable()
key_iterator = bayesServer.TableIterator(key_table, [key_variable], [java.lang.Integer(0)])
key_iterator.copyFrom(prior_key)
key_node.setDistribution(key_table)

# Chord node distribution
chord_table = chord_node.newDistribution(0).getTable()
chord_iterator = bayesServer.TableIterator(chord_table, [label_variable , key_variable, chord_variable], [java.lang.Integer(0), java.lang.Integer(0), java.lang.Integer(0)])
chord_iterator.copyFrom(np.ravel(prior_chords))
chord_node.setDistribution(chord_table)

# Label node distribution
label_table = label_node.newDistribution(0).getTable()

label_iterator = bayesServer.TableIterator(label_table, [label_variable], [java.lang.Integer(0)])
label_iterator.copyFrom(prior_label)
label_node.setDistribution(label_table)

# bass node distribution

bass_table = bass_node.newDistribution(0).getTable()
bass_iterator = bayesServer.TableIterator(bass_table, [chord_variable, bass_variable], [java.lang.Integer(0), java.lang.Integer(0)])
bass_iterator.copyFrom(np.ravel(prior_bass))
bass_node.setDistribution(bass_table)


# DISTIRIBUTIONS WITHIN SAME TIME SLICE

# Basschroma node distribution

basschroma_distr = bayesServer.CLGaussian(basschroma_node.newDistribution())
#basschroma_table = basschroma_distr.getTable()
#basschroma_iterator = bayesServer.TableIterator(basschroma_table, [bass_variable], [java.lang.Integer(0)])

for i in range(0, n_roots):
    for j in range(0, basschroma_rows):
        basschroma_distr.setMean(i, j, float(bass_to_basschroma_mu[i, j]))
for i in range(0, n_roots):
    for j in range(0, basschroma_rows):
        for k in range(0, basschroma_rows):
            basschroma_distr.setCovariance(i, j, k, float(bass_to_basschroma_sigma[i, j, k]))
basschroma_node.setDistribution(basschroma_distr)


# decidere cast a float dentro funzioni o fuori

# Chord Salience distribution node

chordsalience_distr = bayesServer.CLGaussian(salience_node.newDistribution())

for i in range(0, n_chords_and_no_chord):
    for j in range(0, salience_rows):
        chordsalience_distr.setMean(i, j, chord_to_chordsalience_mu[i, j])

for i in range(0, n_chords_and_no_chord):
    for j in range(0, salience_rows):
        for k in range(0, salience_rows):
            chordsalience_distr.setCovariance(i, j, k, chord_to_chordsalience_sigma[i, j, k])
salience_node.setDistribution(chordsalience_distr)



# DISTRIBUTIONS WITHIN TWO TIME SLICES

# Key node

key_trans_table = key_node.newDistribution(1).getTable()
key_trans_iterator = bayesServer.TableIterator(key_trans_table, [key_variable, key_variable], [java.lang.Integer(-1), java.lang.Integer(0)])
key_trans_iterator.copyFrom(np.ravel(key_to_key.astype('float')))
key_node.getDistributions().set(1, key_trans_table)

# Chord node

chord_trans_table = chord_node.newDistribution(1).getTable()
chord_trans_iterator = bayesServer.TableIterator(chord_trans_table, [label_variable, key_variable, chord_variable, chord_variable], [java.lang.Integer(0), java.lang.Integer(0), java.lang.Integer(-1), java.lang.Integer(0)])
chord_trans_iterator.copyFrom(np.ravel(tot_to_chord.astype('float')))
chord_node.getDistributions().set(1, chord_trans_table)


# Bass node

bass_trans_table = bass_node.newDistribution(1).getTable()
bass_trans_iterator = bayesServer.TableIterator(bass_trans_table, [chord_variable, chord_variable, bass_variable], [java.lang.Integer(-1), java.lang.Integer(0), java.lang.Integer(0)])
bass_trans_iterator.copyFrom(np.ravel(chordtransition_to_bass.astype('float')))
bass_node.getDistributions().set(1, bass_trans_table)


#### Validation

network.validate(bayesServer.ValidationOptions())


### INFERENCE

path = "test_corto_2.wav"
[data, rate] = librosa.load(path)

Beat = classes_definition.BeatLabelNode(data, rate)
BassChromagram = classes_definition.BassChromagramNode(data, rate, Beat.beat)
ChordSalience = classes_definition.ChordSalienceNode(data, rate, Beat.beat)
end_time = len(Beat.beat)


inference_factory = bayesServer.inference.VariableEliminationInferenceFactory()
inference = inference_factory.createInferenceEngine(network)
query_options = inference_factory.createQueryOptions()
query_output = inference_factory.createQueryOutput()

#query_options.setLogLikelihood(True)


# SET EVIDENCES
# label evidence


for i in range(0, len(Beat.beat)):
    inference.getEvidence().setState(beat_states[int(Beat.label[i])-1], java.lang.Integer(i))

# basschroma evidence

for i in range(0, len(basschroma_variables)):
    for j in range(0, len(Beat.beat)):
        basschroma_value = java.lang.Double(BassChromagram.synch_bass_chromagram[i, j])
        inference.getEvidence().set(basschroma_variables[i], basschroma_value, java.lang.Integer(j))



# salience evidence

chordsalience_and_no_chord = np.r_[ChordSalience.synch_chord_salience, np.zeros([1, len(Beat.beat)])]

for i in range(0, len(salience_variables)):
    for j in range(0, len(Beat.beat)):
        chordsalience_value = java.lang.Double(chordsalience_and_no_chord[i, j])
        inference.getEvidence().set(salience_variables[i], chordsalience_value, java.lang.Integer(j))


# SET QUERY
chord_queries = []
for i in range(0, len(Beat.beat)):
    chord_queries.append(bayesServer.Table(chord_variable, java.lang.Integer(i)))
    inference.getQueryDistributions().add(bayesServer.inference.QueryDistribution(chord_queries[i]))

inference.query(query_options, query_output)

for i in range(0, len(Beat.beat)):
    chord_prob = np.zeros([n_chords_and_no_chord])
    for j in range(0, len(chord_states)):
        state_context = bayesServer.StateContext(chord_states[j], java.lang.Integer(i))
        chord_prob[j] = (chord_queries[i].get([state_context]))
    index = np.argmax(chord_prob)
    value = np.max(chord_prob)
    print(chord_states[index], value)


print('tuma')
