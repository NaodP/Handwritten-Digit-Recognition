import numpy as np
from Neuron import Neuron
from Connection import Connection

# Neural Network Class
class NeuralNetwork():

    # Initialize The Whole Network
    def __init__(self,structure):
        self.structure = [int(x) for x in structure]
        self.layers = []

        # Initialize All Neurons
        for num in self.structure:
            layer = []
            for i in range(num):
                layer.append(Neuron())
            self.layers.append(layer)

        # Initialize All Connections
        for i in range(0,len(self.layers)-1):
            for firstNode in self.layers[i]:
                for secondNode in self.layers[i+1]:
                    connection = Connection(np.random.randn())      # New Connection With Random Weight
                    connection.sourceNeuron = firstNode             # Add First Node To Connection's Source
                    connection.destNeuron = secondNode              # Add Second Node To Connection's Destination
                    firstNode.outputConnections.append(connection)  # Add Connection To First Node's Output Connections
                    secondNode.inputConnections.append(connection)  # Add Connection To Second Node's Input Connections

    # Sigmoid Activation Function
    def sigmoidActivation(self, z):
        return (1.0 / (1.0 + np.exp(-z)))

    # Sigmoid Activation Function Derivative
    def sigmoidDerivative(self, z):
        z = self.sigmoidActivation(z)
        return z * (1-z)

    # Error Derivative (Sum Of Squared Residuals)
    def errorDerivative(self, output, expected):
        return output - expected

    def rand(self, x):
        return 1

    # Forward Propagation
    def forwardProp(self, inputs):
        outputValues = []

        # Set Input Layer Node Outputs
        for input, node in zip(inputs, self.layers[0]):
            node.totalOutput = input

        # Hidden Layers + Output Layer
        for i in range(1, len(self.layers)):                # For Each Layer
            for node in self.layers[i]:                     # For Each Node In The Layer
                node.totalInput = node.bias
                for connection in node.inputConnections:    # For Each Node's Input Connections
                    node.totalInput += connection.weight * connection.sourceNeuron.totalOutput
                
                # Put The Input Through The Activation Function
                node.totalOutput = self.sigmoidActivation(node.totalInput)

        # Get All Output Values
        for node in self.layers[-1]:
            outputValues.append(node.totalOutput)

        return outputValues
    
    # Backward Propagaion
    def backwardProp(self, outputData):
        
        # Output Layer
        for node, output in zip(self.layers[-1], outputData):
            node.outputDerivative = self.errorDerivative(node.totalOutput, output)   # Compute Error Derivative For Output Layer

        # Start With Output Layer And Go Backwards Through Hidden Layers 
        for i in range(len(self.layers)-1,-1,-1):

            # Compute Error Derivative Of Each Node With Respect To Its Total Input and Each Input Weight
            for node in self.layers[i]: 
                node.inputDerivative = node.outputDerivative * self.sigmoidDerivative(node.totalOutput)
                node.totalInputDerivative += node.inputDerivative
                node.numInputDerivatives += 1

            # Compute Error Derivative With Respect To Each Input Connection Weight
            for node in self.layers[i]:
                for connection in node.inputConnections:
                    connection.errorDerivative = node.inputDerivative * connection.sourceNeuron.totalOutput
                    connection.totalErrorDerivative += connection.errorDerivative
                    connection.numErrorDerivatives += 1

            # If We Are At The Input Layer
            if i == 0: continue

            # Compute Error Derivative With Respect To Each Node's Output
            for node in self.layers[i-1]: # For Each Node In The Previous Layer
                node.outputDerivative = 0
                for connection in node.outputConnections:
                    node.outputDerivative += connection.weight * connection.destNeuron.inputDerivative

    # Update New Weights And Biases
    def updateValues(self, alpha):
        for layer in self.layers:
            for i in range(0,len(layer)):
                node = layer[i]
                # Update Bias
                if node.numInputDerivatives > 0:
                    node.bias -= alpha * node.totalInputDerivative / node.numInputDerivatives
                    node.totalInputDerivative = 0
                    node.numInputDerivatives = 0

                # Update Input Connections' Weights
                for connection in node.inputConnections:
                    if connection.numErrorDerivatives > 0:
                        connection.weight -= alpha / connection.numErrorDerivatives * connection.totalErrorDerivative
                        connection.numErrorDerivatives = 0
                        connection.totalErrorDerivative = 0

    # Calculate Loss
    def calculateLoss(self, outputData, outputValues):
        loss = 0
        for D, V in zip(outputData, outputValues):
            loss += ((D - V)**2)                          # Sum Of Squared Residuals
        loss *= 0.5                                       # Divide Sum By Two

        return loss

    # Train The Network With The Given Data
    def train(self, inputData, outputData, alpha):

        # Put The Network Through Forward Propagation
        outputValues = self.forwardProp(inputData)

        # Put The Network Through Backward Propagation
        self.backwardProp(outputData)

        # Update The Values In The Network
        self.updateValues(alpha)

        # Calculate Loss
        loss = self.calculateLoss(outputData, outputValues)
        
        return loss

    # Test The Network And Return The Output
    def test(self, inputData):
        outputValue = self.forwardProp(inputData)
        return outputValue














    # Print Bias And Weights In Network (For Debugging Purposes)
    def __str__(self):
        output = []
        for i in range(0,len(self.layers)):
            layer = []
            for node in self.layers[i]:
                nodeWeights = [node.bias]
                for connection in node.outputConnections:
                    nodeWeights.append(connection.weight)
                layer.append(nodeWeights)
            output.append(layer)

        return str(output)