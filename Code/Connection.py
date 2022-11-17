# Connection Class

from Neuron import Neuron

class Connection():
    def __init__(self, weight):
        self.sourceNeuron = Neuron()     # Source Neuron
        self.destNeuron = Neuron()       # Destination Neuron
        self.weight = weight             # Initialize Connection's Weight While Creating The Network  
        self.errorDerivative = 0         # Error Derivative With Respect To This Weight
        self.totalErrorDerivative = 0    # Accumulated Error Derivative Since Last Update
        self.numErrorDerivatives = 0     # Total Number Of Error Derivatives