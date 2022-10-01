from array import *
import numpy as np
import matplotlib.pyplot as plt
class Maze:
    
    
    def __init__(self, rows, columns):
 
        self.rows = rows
        self.columns = columns
        self.dRow = [0, 1, 0, -1]
        self.dCol = [-1, 0, 1, 0]
    def validity(self, r, c):
        
        if (r < 0 or r >= self.rows): #row bounds
            return 0

        if (c < 0 or c >= self.columns): #column bounds
            return 0
        
        if (self.visited[r][c]): #already visited
            return 0
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
            blocked = np.random.choice(np.arange(0, 2), p=[0.7,0.3])
            if (r == 0 and c == 0):
                self.maze[r][c]  = 0
            else:
                self.maze[r][c] = blocked
            for i in range(4):
                adjx = r + self.dRow[i]
                adjy = c + self.dCol[i]
                self.stack.append([adjx, adjy])
    
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

if __name__ == "__main__":
    maze1 = Maze(50,50)
    maze1.generate_maze()
    maze1.visualize_maze()