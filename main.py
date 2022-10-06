from array import *
from re import S
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from queue import PriorityQueue
class Cell:

    def __init__(self, coordinates=None, parentCell=None):
        
        self.gval = 0
        self.hval = 0
        self.fval = 0

        self.parentCell = parentCell
        self.coordinates = coordinates
    
    def __eq__(self,other):
        if (self.coordinates == other.coordinates):
            return 1
        return 0

    def __lt__(self,other):
        if (self.fval == other.fval):
            return 1
        return 0


class Maze:
    
    
    def __init__(self, rows, columns):
 
        self.rows = rows
        self.columns = columns
        self.steps = 0
        self.dRow = [0, 1, 0, -1]
        self.dCol = [-1, 0, 1, 0]
    def validity(self, r, c):
        
        if (r < 0 or r >= self.rows): #row bounds
            return 0

        if (c < 0 or c >= self.columns): #column bounds
            return 0
        
        if (self.visited[r][c]): #already visited
            return 0
        if (self.maze[r][c] == 2): #cell part of dfs path
            return 1
        if (self.maze[r][c]): #cell blocked
            return 0
        return 1

        

    
    def generate_maze(self):
        
        self.maze = [[0] * self.rows for i in range(self.columns)]
        self.visited = [[0] * self.rows for i in range(self.columns)]
        self.stack = []
        self.stack.append([0,0])
        
        while (len(self.stack) > 0):
            current = self.stack[len(self.stack) - 1]
            self.stack.remove(self.stack[len(self.stack) - 1])
            r = current[0]
            c = current[1]

            if (self.validity(r,c) == 0):
                continue
            self.visited[r][c] = 1
            blocked = np.random.choice(np.arange(0, 2), p=[0.70,0.30])
            if (r == 0 and c == 0):
                self.maze[r][c]  = 0
            elif (r == 0 and c == 1):
                self.maze[r][c]  = 0
            elif (r == 1 and c == 0):
                self.maze[r][c]  = 0
            elif (r == self.rows - 1 and c == self.columns - 1):
                self.maze[r][c] = 0
            
            else:
                self.maze[r][c] = blocked
            for i in range(4):
                neighbour_x = r + self.dRow[i]
                neighbour_y = c + self.dCol[i]
                self.stack.append([neighbour_x, neighbour_y])
    def dfsolver(self):

        self.visited = [[0] * self.rows for i in range(self.columns)]
        self.stack = []
        self.stack.append([0,0])
        while (len(self.stack) > 0):
            current = self.stack[len(self.stack) - 1]
            self.stack.remove(self.stack[len(self.stack) - 1])
            r = current[0]
            c = current[1]

            if (self.validity(r,c) == 0):
                continue
            self.visited[r][c] = 1
            self.maze[r][c] = 2
            self.steps+=1
            print(r,c)
            if (r == self.rows - 1 and c == self.columns - 1):
                print("Reached the goal in", self.steps , "steps")
                break
            for i in range(4):
                neighbour_x = r + self.dRow[i]
                neighbour_y = c + self.dCol[i]
                self.stack.append([neighbour_x, neighbour_y])
    def bfsolver(self):

        self.visited = [[0] * self.rows for i in range(self.columns)]
        self.stack = []
        self.stack.append([0,0])
        while (len(self.stack) > 0):
            current = self.stack[0]
            self.stack.pop(0)
            r = current[0]
            c = current[1]

            if (self.validity(r,c) == 0):
                continue
            self.visited[r][c] = 1
            self.maze[r][c] = 2
            self.steps+=1
            print(r,c)
            if (r == self.rows - 1 and c == self.columns - 1):
                print("Reached the goal in", self.steps , "steps")
                break
            for i in range(4):
                neighbour_x = r + self.dRow[i]
                neighbour_y = c + self.dCol[i]
                self.stack.append([neighbour_x, neighbour_y])
    def visualize_maze(self):
        m = 0
        n = 0
        for i in range(self.rows):
            for j in range(self.columns):
                print(self.maze[i][j], end = " ")
                if self.maze[i][j] == 1:
                    m+=1
                else:
                    n+=1

            print("\n")
        print(m/(m+n))
        plt.imshow(self.maze,cmap='gray_r')
        plt.show()
        #print(self.maze)
        #print(self.visited)

def AstarSearch(start, goal, maze : Maze):

    startCell = Cell(start, None)
    goalCell = Cell(goal, None)

    #initialize all values as 0 for start and end nodes

    startCell.gval = startCell.hval = startCell.fval = 0
    goalCell.gval = goalCell.hval = goalCell.fval = 0
    
    #create a priority queue for the open list

    openList = PriorityQueue()
    closedList = []
    openList.put(startCell)
    
    AdjacentRowIndex = [0, 1, 0, -1]
    AdjacentColumnIndex = [-1, 0, 1, 0]

    while(openList):
        currentCell = openList.get()
        if currentCell == goalCell:
            return "Path Found" #A function to return the actual path
        
        closedList.append(currentCell)

        neighbourCells = [] #To explore the neighbours of the current cell
        for i in range(4):
            neighbour_x = currentCell.coordinates[0] + AdjacentRowIndex[i]
            neighbour_y = currentCell.coordinates[0] + AdjacentColumnIndex[i]
            if not maze.validity(neighbour_x,neighbour_y):
                continue
            neighbourCords = (neighbour_x,neighbour_y)
            neighbour = Cell(neighbourCords,currentCell)
            neighbourCells.append(neighbour)
        for neighbour in neighbourCells:
            visited = 0
            weakerNeighbour = 0
            for visitedNeighbour in closedList:
                if visitedNeighbour == neighbour:
                    visited = 1
                    break
            if visited:
                continue

            neighbour.gval = currentCell.gval + 1
            neighbour.hval = (goalCell.coordinates[0] - currentCell.coordinates[0]) + (goalCell.coordinates[1] - currentCell.coordinates[1])
            neighbour.fval = neighbour.gval + neighbour.hval

            for openNeighbour in openList:
                if neighbour.coordinates == openNeighbour.coordinates and neighbour.gval >= openNeighbour.coordinates:
                    weakerNeighbour = 1
                    break
            if weakerNeighbour:
                continue
            openList.put(neighbour)
            

    



if __name__ == "__main__":
    maze1 = Maze(50,50)
    maze1.generate_maze()
    maze1.visualize_maze()
    maze3 = maze2 = maze1
    maze2.dfsolver()
    maze2.visualize_maze()
    maze3.bfsolver()
    maze3.visualize_maze()