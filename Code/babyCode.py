import pickle
import sys

import tkinter
from tkinter import *
from PIL import Image, ImageTk

# sys.setrecursionlimit(4000)
# from NeuralNetwork import NeuralNetwork

# test = NeuralNetwork([784,200,50,25,10])

# with open('test.pickle', 'wb') as file:
#     pickle.dump(test, file, pickle.HIGHEST_PROTOCOL)

# with open('test.pickle', 'rb') as file:
#     test = pickle.load(file)
#     print(test)

root = Tk()
root.geometry("500x500")

test = Image.open('Final Digit Detection/Data/img_10004.jpg')
test = test.resize((500,500), Image.Resampling.LANCZOS)
testPhoto = ImageTk.PhotoImage(test)

canvas = Canvas(root, width=500, height=500)
canvas.pack()

canvas.create_image(10,10, anchor=NW, image=testPhoto)

root.mainloop()