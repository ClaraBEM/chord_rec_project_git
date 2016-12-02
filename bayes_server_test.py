from math import sqrt
from jpype import *

classpath = '/Users/Clara/Downloads/bayesserver-7.9/Java/bayesserver-7.9.jar'

startJVM('/Library/Java/JavaVirtualMachines/jdk1.7.0_51.jdk/Contents/Home/jre/lib/jli/libjli.dylib', "-Djava.class.path=%s" %classpath)

bayesServer = JPackage("com.bayesserver")
bayesServerInference = bayesServer.inference
#
network = bayesServer.Network()

aTrue = bayesServer.State("True")
aFalse = bayesServer.State("False")
a = bayesServer.Node('a', [aTrue, aFalse])

network.getNodes().add(a)

bTrue = bayesServer.State("True")
bFalse = bayesServer.State("False")
b = bayesServer.Node('b', [bTrue, bFalse])
network.getNodes().add(b)

cTrue = bayesServer.State("True")
cFalse = bayesServer.State("False")
c = bayesServer.Node('c', [cTrue, cFalse])
network.getNodes().add(c)

network.getLinks().add(bayesServer.Link(a, b))
network.getLinks().add(bayesServer.Link(a, c))
network.getLinks().add(bayesServer.Link(b, c))

tableA = a.newDistribution().getTable()

tableA.set(0.1, [aTrue])
tableA.set(0.9, [aFalse])

a.setDistribution(tableA)

tableB = b.newDistribution().getTable()
tableB.set(0.2, [aTrue, bTrue])
tableB.set(0.8, [aTrue, bFalse])
tableB.set(0.15, [aFalse, bTrue])
tableB.set(0.85, [aFalse, bFalse])

b.setDistribution(tableB)

# tableC = c.newDistribution().getTable()
# tableC.set(0.3, [aTrue, cTrue])
# tableC.set(0.7, [aTrue, cFalse])
# tableC.set(0.4, [aFalse, cTrue])
# tableC.set(0.6, [aFalse,cTrue])
#
# c.setDistribution(tableC)

tableC = c.newDistribution().getTable()



iteratorC = bayesServer.TableIterator(tableC, [a, b, c])
iteratorC.copyFrom([0.4, 0.6, 0.55, 0.45, 0.32, 0.68, 0.01, 0.99])
c.setDistribution(tableC)