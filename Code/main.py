from NeuralNetwork import NeuralNetwork 
import random
import pickle
import sys; sys.setrecursionlimit(4000) # To Pickle The Network
import csv
import time

# Soft Max Function
def softMax(values):
    output = []
    totalVal = 0
    for val in values: totalVal += val
    for val in values: output.append(f'{round((val/totalVal)*100,2)}%')

    return output

# Get Training Data
def getTrainingData():
    file = open('Final Digit Detection/Data/mnist_train.csv', mode ='r')
    csvFile = csv.reader(file)

    finalInput = []

    # LIMIT INPUTS
    counter = 0
    numOfInputs = 40000

    for line in csvFile:
        input = []
        output = []

        # Correct Output
        if line[0] == '0': output = [1,0,0,0,0,0,0,0,0,0]
        if line[0] == '1': output = [0,1,0,0,0,0,0,0,0,0]
        if line[0] == '2': output = [0,0,1,0,0,0,0,0,0,0]
        if line[0] == '3': output = [0,0,0,1,0,0,0,0,0,0]
        if line[0] == '4': output = [0,0,0,0,1,0,0,0,0,0]
        if line[0] == '5': output = [0,0,0,0,0,1,0,0,0,0]
        if line[0] == '6': output = [0,0,0,0,0,0,1,0,0,0]
        if line[0] == '7': output = [0,0,0,0,0,0,0,1,0,0]
        if line[0] == '8': output = [0,0,0,0,0,0,0,0,1,0]
        if line[0] == '9': output = [0,0,0,0,0,0,0,0,0,1]

        # Correct Input
        for i in range(1,len(line)):
            input.append(float(line[i])/255)

        # Final Input
        finalInput.append((input,output))

        # LIMIT INPUTS
        counter += 1
        if counter == numOfInputs: break

    return finalInput

# Get Testing Data
def getTestingData():
    file = open('Final Digit Detection/Data/mnist_test.csv', mode ='r')
    csvFile = csv.reader(file)

    input = []
    output = []

    # LIMIT INPUTS
    counter = 0
    numOfInputs = 5000

    for line in csvFile:
        # Correct Output
        if line[0] == '0': output.append([1,0,0,0,0,0,0,0,0,0])
        if line[0] == '1': output.append([0,1,0,0,0,0,0,0,0,0])
        if line[0] == '2': output.append([0,0,1,0,0,0,0,0,0,0])
        if line[0] == '3': output.append([0,0,0,1,0,0,0,0,0,0])
        if line[0] == '4': output.append([0,0,0,0,1,0,0,0,0,0])
        if line[0] == '5': output.append([0,0,0,0,0,1,0,0,0,0])
        if line[0] == '6': output.append([0,0,0,0,0,0,1,0,0,0])
        if line[0] == '7': output.append([0,0,0,0,0,0,0,1,0,0])
        if line[0] == '8': output.append([0,0,0,0,0,0,0,0,1,0])
        if line[0] == '9': output.append([0,0,0,0,0,0,0,0,0,1])

        # Correct Input
        pixels = []
        for i in range(1,len(line)):
            pixels.append(float(line[i])/255)
        input.append(pixels)

        # LIMIT INPUTS
        counter += 1
        if counter == numOfInputs: break

    return input, output

# Test The Network
def testNetwork(Network: NeuralNetwork, testingInput, testingOutput, accuracy=0):
    for i, inputData, outputData in zip(range(1,len(testingInput)+1), testingInput, testingOutput):
        outputValues = Network.test(inputData)

        # Print The Results
        softExpected = softMax(outputData)
        softActual = softMax(outputValues)
        # print(f'Input {i}')
        # [print(f'{x:^6} | ', end='') for x in softExpected]
        # print()
        # [print(f'{x:^6} | ', end='') for x in softActual]
        # print('\n')

        # Compute Accuracy
        max = 0; index = 0
        for i in range(0,len(outputValues)):
            val = outputValues[i]
            if val > max:
                max = val
                index = i
        
        if(outputData[index] == 1): accuracy += 1
        
    accuracy /= len(testingInput)

    return accuracy

# Main Function
def main():

    # Create Network
    Epochs = 15                          # Num Of Training Cycles
    Structure = [784,100,10]             # Network Structure
    Alpha = 0.09                         # Learning Rate
    Deviance = 0.1                       # Highest Deviance Accepted
    Accuracy = 0                         # Final Accuracy Based On Deviance
    Loss = 0                             # Loss After Each Epoch
    BatchRate = 10                       # Percentage Of Total Input To Feed Forward Each Epoch
    Network = NeuralNetwork(Structure)

    # Get Training Input
    totalTrainingInput = getTrainingData()

    # Get Testing Data
    testingInput, testingOutput = getTestingData()

    # Timing
    totalTime = time.time()

    # Train The Network
    for epoch in range(1,Epochs+1):
        Loss = 0
        BatchSize = int(len(totalTrainingInput)*(BatchRate/100.0))

        # Timing
        start = time.time()

        # Get A Random Batch Of The Data
        tempTrainingInput = random.sample(totalTrainingInput, BatchSize)

        # Put The Batch Through The Network
        for trainingInput, i in zip(tempTrainingInput, range(1, len(tempTrainingInput)+1)):
            IN, OUT = trainingInput
            Loss += Network.train(IN, OUT, Alpha)

            # TEMPORARY - print % of epoch finished
            j = (i+1)/(len(tempTrainingInput)+1)
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()
            sys.stdout.write('\r')

        Loss /= BatchSize
        end = time.time()

        # Test The Network
        #if epoch % 10 == 0: 
        Accuracy = testNetwork(Network, testingInput, testingOutput)
        
        # Print Loss
        print(f'| Epoch {epoch} | Loss {round(Loss,5)} | Accuracy {round(Accuracy*100, 2)}% | Minutes {round((end - start) / 60, 2)} |')

        # Save Neural Network Object
        file = open('Final Digit Detection/Samples/Saved_Pickle_Network.txt', 'wb')
        pickle.dump(Network, file, pickle.HIGHEST_PROTOCOL)

    # Final Test
    Accuracy = testNetwork(Network, testingInput, testingOutput)
    
    # Final Report
    print(f'------------------------------ Final Report ------------------------------')
    print(f'Network Structure {Network.structure} | Learning Rate {Alpha} | Epochs {Epochs}')
    print(f'Final Loss {Loss} | Total Accuracy {round(Accuracy*100, 2)}% | Time Taken {round((time.time() - totalTime)/3600, 2)} Hours')
        
# Start
main()