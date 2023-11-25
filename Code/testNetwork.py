import csv
import os
import pickle
import sys; sys.setrecursionlimit(4000) # To Pickle The Network
import numpy as np
from PIL import Image
import random
from NeuralNetwork import NeuralNetwork
import tkinter
from tkinter import *
from PIL import Image, ImageTk

path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir_data = dir.replace('Code', 'Data')
dir_samples = dir.replace('Code', 'Samples')

# Soft Max Function
def softMax(values):
    output = []
    totalVal = 0
    for val in values: totalVal += val
    for val in values: output.append(round((val/totalVal)*100,2))

    return output

def loadNetwork():
    os.chdir(dir_samples)
    with open('Saved_Pickle_Network.txt', 'rb') as file:
        Network = pickle.load(file)
        return Network

def getTestingData():
    os.chdir(dir_data)
    file = open('mnist_test.csv', mode ='r')
    csvFile = csv.reader(file)

    finalInput = []

    # LIMIT INPUTS
    counter = 0
    numOfInputs = 1000

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
            input.append(int(line[i]))
        
        finalInput.append((input,output))

        # LIMIT INPUTS
        counter += 1
        if counter == numOfInputs: break

    return finalInput

def main():
    Network = loadNetwork()
    testSize = 100

    totalTestInput = getTestingData()
    tempTestInput = random.sample(totalTestInput, testSize)
    count = 0
    

    for IN, output in tempTestInput:
        outputValues = Network.test(IN)
        softOutputValues = softMax(outputValues)

        sortedValues = []
        for i in range(0, len(softOutputValues)):
            sortedValues.append((softOutputValues[i], i))
        
        sortedValues = sorted(sortedValues, reverse=True)

        max = 0; index = 0; index2 = 0 
        for i in range(0,len(outputValues)):
            val = outputValues[i]
            if val > max:
                max = val
                index = i
            if output[i] == 1: index2 = i

        index = sortedValues[0][1]
        indexPercentage = sortedValues[0][0]

        indexSecond = sortedValues[1][1]
        indexSecondPercentage = sortedValues[1][0]

        indexThird = sortedValues[2][1]
        indexThirdPercentage = sortedValues[2][0]


        # Show Prediction
        print(f'Predicted: {index} | Actual: {index2}')
        if index == index2: count += 1

        convertedPixels = np.reshape(IN, (-1,28))
        pic = Image.fromarray(convertedPixels)

        # Display
        root = Tk()
        root.geometry("600x800")

        pic = pic.resize((500,500), Image.Resampling.LANCZOS)
        testPhoto = ImageTk.PhotoImage(pic)

        label = Label(root, anchor=N, width=100, height=0, text = f'{index} | {indexPercentage}%\n{indexSecond} | {indexSecondPercentage}%\n{indexThird} | {indexThirdPercentage}%', font=("Arial", 20))
        label.pack(side=TOP, pady=20)

        canvas = Canvas(root, width=500, height=500)
        canvas.create_image(0,0, anchor=NW, image=testPhoto)
        canvas.pack(side=TOP)

        label2 = Label(root, anchor=N, width=100, height=0, text=f'Actual = {index2}', font=("Arial", 25))
        label2.pack(side=TOP, pady=0)

        root.after(5000,lambda:root.destroy())
        root.mainloop() 

    # Print Final Accuracy
    print(f'Accuracy {round((count/testSize)*100,2)}')       

# Start
main()