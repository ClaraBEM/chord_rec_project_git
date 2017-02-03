from jpype import *
import numpy as np

classpath = '/Users/Clara/Downloads/bayesserver-7.9/Java/bayesserver-7.9.jar'

startJVM('/Library/Java/JavaVirtualMachines/jdk1.7.0_51.jdk/Contents/Home/jre/lib/jli/libjli.dylib', "-Djava.class.path=%s" %classpath)

bayesServer = JPackage("com.bayesserver")
bayesServerInference = bayesServer.inference

network = bayesServer.Network()

continuous_variable = bayesServer.Variable('continuos', bayesServer.VariableValueType.CONTINUOUS)
continuous_node = bayesServer.Node('continuous', [continuous_variable])
continuous_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)


N = 2
discrete_states = []
for i in range(0, N):
    label = 'state'+str(i)
    discrete_states.append(bayesServer.State(label))
discrete_variable = bayesServer.Variable('discrete', discrete_states)
discrete_node = bayesServer.Node('discrete', [discrete_variable])
discrete_node.setTemporalType(bayesServer.TemporalType.TEMPORAL)

network.getNodes().add(continuous_node)
network.getNodes().add(discrete_node)

network.getLinks().add(bayesServer.Link(discrete_node, continuous_node))
network.getLinks().add(bayesServer.Link(discrete_node, discrete_node, 1))

# set distributions
# discrete apriori distribution

discrete_table = discrete_node.newDistribution(0).getTable()
apriori = np.random.rand(N)
apriori = apriori/np.sum(apriori)
discrete_iterator = bayesServer.TableIterator(discrete_table, [discrete_variable], [java.lang.Integer(0)])
discrete_iterator.copyFrom(apriori)
discrete_node.setDistribution(discrete_table)

# p(continuous | discrete)
continuous_distr = bayesServer.CLGaussian(continuous_node.newDistribution())
continuous_context = bayesServer.VariableContext(continuous_variable, java.lang.Integer(0), bayesServer.HeadTail.HEAD)

variance = 0.1
discrete_context = []
for i in range(0, N):
    discrete_context.append(bayesServer.StateContext(discrete_states[i], java.lang.Integer(0)))

for i in range(0, N):
    continuous_distr.setMean(continuous_context, float(i), [discrete_context[i]])
    continuous_distr.setVariance(continuous_context, float(variance), [discrete_context[i]])

continuous_node.setDistribution(continuous_distr)



# p(discrete t+1 | discrete t)
trans_prob = np.array([[0.1, 0.9],[0.4, 0.6]])



trans_table = discrete_node.newDistribution(1).getTable()
trans_iterator = bayesServer.TableIterator(trans_table, [discrete_variable, discrete_variable], [java.lang.Integer(-1), java.lang.Integer(0)])
trans_iterator.copyFrom(np.ravel(trans_prob.astype('float')))
discrete_node.getDistributions().set(1, trans_table)



network.validate(bayesServer.ValidationOptions())

end_time = 10000

state = np.zeros(end_time, dtype=np.int)
state[0] = 1
choice = [0, 1]
for i in range(1, end_time):
    state[i] = np.random.choice(choice, p=trans_prob[state[i-1], :])

continuous_samples = np.zeros(end_time)
for i in range(0, end_time):
    continuous_samples[i] = np.sqrt(variance)*np.random.randn(1) + state[i]

inference_factory = bayesServer.inference.RelevanceTreeInferenceFactory()
inference = inference_factory.createInferenceEngine(network)
query_options = inference_factory.createQueryOptions()
query_output = inference_factory.createQueryOutput()
query_options.setPropagation(bayesServer.PropagationMethod.MAX)



for i in range(0, end_time):
    value = java.lang.Double(float(continuous_samples[i]))
    inference.getEvidence().set(continuous_variable, value, java.lang.Integer(i))

discrete_queries = []
for i in range(0, end_time):
    discrete_queries.append(bayesServer.Table(discrete_variable, java.lang.Integer(i)))
    inference.getQueryDistributions().add(bayesServer.inference.QueryDistribution(discrete_queries[i]))

inference.query(query_options, query_output)

discrete_prob = np.zeros((end_time, N))
predicted_state = np.zeros(end_time, dtype=np.int)
for i in range(0, end_time):
    for j in range(0, len(discrete_states)):
        state_context = bayesServer.StateContext(discrete_states[j], java.lang.Integer(i))
        discrete_prob[i, j] = (discrete_queries[i].get([state_context]))
    index = np.argmax(discrete_prob[i,:])
    predicted_state[i] = index

print(np.sum(predicted_state == state) / end_time)

    #print(index, discrete_states[index], value)



