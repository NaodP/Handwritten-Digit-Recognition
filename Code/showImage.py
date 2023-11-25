from PIL import Image
import numpy as np
import csv

def show(input):

    convertedPixels = np.reshape(input, (-1,28))
    pic = Image.fromarray(convertedPixels)
    pic.show()

            
    