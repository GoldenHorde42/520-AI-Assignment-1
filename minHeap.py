from array import *
from re import S
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from queue import PriorityQueue
import pickle
import copy
import time
import logging
from datetime import datetime
import imageio
import os
import main

class minHeap():
    
    def __init__(self) -> None:
        self.heap = []
        self.length = -1
    def parent(self,child):
        return ((child - 1)//2)
    def shiftDown(self,min):
        last = min
        left = ((2*min)+1)
        right = ((2*min)+2)
        if(left<=self.length and self.heap[left] < self.heap[last]):
            last = left
        if(right<=self.length and self.heap[right] < self.heap[last]):
            last = right
        if(min != last):
            temp = self.heap[min]
            self.heap[min] = self.heap[last]
            self.heap[last] = temp
            self.shiftDown(last)
    def shiftUp(self):
        length = self.length
        while(length > 0 and self.heap[self.parent(length)] > self.heap[length]):
            temp = self.heap[self.parent(length)]
            self.heap[self.parent(length)] = self.heap[length]
            self.heap[length] = temp
            length = self.parent(length)
    def put(self,cell):
        self.length += 1
        self.heap.append(cell)
        self.shiftUp()
    def get(self):
        if (self.length < 0):
            self.heap = []
            print("here")
            return
        cell = self.heap[0]
        self.heap[0] = self.heap[self.length]
        self.length-=1
        self.shiftDown(0)
        return cell
    def empty(self):
        if self.length < 0:
            return True
        return False

if __name__ == "__main__":
    heap = minHeap()
    start = (0,0)
    startcell = main.Cell(start, None)
    startcell.fval = startcell.gval = 1001
    heap.put(startcell)
    i = 1000
    newCell = startcell
    while(i>0):
        newCell = main.Cell(start, newCell)
        newCell.fval = newCell.gval = i
        heap.put(newCell)
        i-=1
    while(not heap.empty()):
        cell = heap.get()
        print(cell.fval)

    



