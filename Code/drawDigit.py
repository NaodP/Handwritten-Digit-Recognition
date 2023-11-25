from turtle import *
from tkinter import *
import turtle

def getCoords(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def draw(event):
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y), 
                      fill='white', 
                      width=2)
    lastx, lasty = event.x, event.y

root = Tk()
root.geometry('1000x1000')

canvas = Canvas(root, width=1000, height=1000, bg='black')
canvas.pack(anchor=NW)

canvas.bind('<Button-1>', getCoords)
canvas.bind('<B1-Motion>', draw)

root.mainloop()

# screen = turtle.Screen()

# screen.onclick(write)