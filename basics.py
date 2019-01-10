import numpy as np

class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        for node in input_nodes:
            node.output_nodes.append(self)
        _default_graph.operations.append(self)
    def compute():
        pass

class Add(Operation):
    def __init__(self, x, y):
        super().__init__([x,y])
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var

class Multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x,y])
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var

class MatMul(Operation):
    def __init__(self, x, y):
        super().__init__([x,y])
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)

class PlaceHolder():
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)

class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)

class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
    def set_as_default(self):
        global  _default_graph
        _default_graph = self

def traverse_postorder(operation):
    """
    To make sure the computational is done in the correct order.
    Ax first, then Ax+b
    """
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation): # a classe Operation
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(operation)
    return nodes_postorder

class Session():
    def run (self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if type(node) == PlaceHolder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else :
                #Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs) #Asterics for independency of size of input_node
            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output

g = Graph()
g.set_as_default()
A = Variable([[10,20], [30,40]])
b = Variable([1,2,])
x = PlaceHolder()
y = MatMul(A, x)
z = Add(y, b)
####
sess = Session()
result = sess.run(operation=z, feed_dict={x:10})
print('Result: {}' .format(result))

#Starting Classification
print('')
print('Starting Classification!')
import matplotlib.pyplot as plt
# %matplotlib inline #only works on jupyter notebook
def sigmoid(z):
    return 1 / (1+np.exp(-z))

sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)

# plt.plot(sample_z, sample_a) # Using on jupyter notebook

class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])
    def compute(self, z_val):
        return 1 / (1+np.exp(-z_val))

from sklearn.datasets import make_blobs
data = make_blobs(n_samples=50, n_features=2,centers=2,random_state=75)
features = data[0]
labels = data[1]
# plt.scatter(features[:,0],features[:,1],c=labels,cmap='coolwarm')
plt.scatter(features[:,0], features[:,1],cmap='coolwarm')

x = np.linspace(0,11,10)
y = -x + 5
plt.scatter(features[:,0],features[:,1],c=labels,cmap='coolwarm')
plt.plot(x,y)
# plt.show()
# (1,1) * FeatureMatrix -5 = 0 #Equation of the previous line
var = np.array([1,1]).dot(np.array([ [8],[10] ]) ) - 5
print('Se var eh positivo esta acima da linha ou abaixo se for negativo')
print('var eh: {}' .format(var))

g = Graph()
g.set_as_default()
x = PlaceHolder()
w = Variable([1,1])
b = Variable(-5)
z = Add(MatMul(w, x), b)
a = Sigmoid(z)
####
sess = Session()
result = sess.run(operation=a, feed_dict={x:[8,10]})
print('')
print('Certainty that it belongs to group B: {}' .format(result))
