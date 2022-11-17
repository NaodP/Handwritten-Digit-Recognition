# Neuron Class

class Neuron():
    def __init__(self):
        self.bias = 0.1                  # Bias Value - Can Be Initialized As 0 Or 0.1
        self.totalInput = 0              # Total Input Value
        self.totalOutput = 0             # Total Output Value
        self.inputConnections = []       # All Input Connections As Part Of A List
        self.outputConnections = []      # All Output Connections As Part Of A List
        self.inputDerivative = 0         # Error Derivative With Respect To The Input
        self.outputDerivative = 0        # Error Derivative With Respect To The Output
        self.totalInputDerivative = 0    # Accumulated Error Derivative With Respect To The Total Input
        self.numInputDerivatives = 0     # Number Of Accumulated Error Derivatives With Respect To The Total Input
