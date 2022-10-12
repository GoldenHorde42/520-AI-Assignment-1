from array import *
from re import S
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from queue import PriorityQueue
import pickle
import copy
class Cell:

    def __init__(self, coordinates=None, parentCell=None):
        
        self.gval = 0
        self.hval = 0
        self.fval = 0

        self.parentCell = parentCell
        self.coordinates = coordinates
        self.childrenCells = []
    
    def __eq__(self,other):
        if (self.coordinates == other.coordinates):
            return 1
        return 0

    def __lt__(self,other):
        if (self.fval < other.fval):
            return 1
        elif(self.fval == other.fval):
            if (self.gval > other.gval):
                return 1
        return 0


class Maze:
    
    
    def __init__(self, rows, columns):
 
        self.rows = rows
        self.columns = columns
        self.steps = 0
        self.dRow = [0, 1, 0, -1]
        self.dCol = [-1, 0, 1, 0]
        self.visited = [[0] * self.rows for i in range(self.columns)]
        self.solution = []
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
    def generateAgentMaze(self):
        self.maze = [[0] * self.rows for i in range(self.columns)]
    def clearVisitedArray(self):
        self.visited = [[0] * self.rows for i in range(self.columns)]
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
        k = 0
        colormaze = copy.deepcopy(self.maze)
        for i in range(self.rows):
            for j in range(self.columns):
                
                if colormaze[i][j] == 1:
                    m+=1
                # elif self.maze[i][j] == 2:
                #     k+=1
                elif (i,j) in self.solution:
                    colormaze[i][j] = 2
                    k+=1
                else:
                    n+=1
                #print(self.maze[i][j], end = " ")
            #print("\n")
        #print(m/(m+n))
        if k == 0:
            gridcolors = colors.ListedColormap(["white","black"])
        else:
            gridcolors = colors.ListedColormap(["white","black","red"])
        
        plt.imshow(colormaze,cmap=gridcolors)
        plt.show()
        #print(self.maze)
        #print(self.visited)
def tracePath(cell : Cell, maze : Maze, notGoal=0):
    pathTaken = []
    maze.solution = []
    steps = 0
    while cell is not None:
        pathTaken.append(cell.coordinates)
        #maze.maze[cell.coordinates[0]][cell.coordinates[1]] = 2
        maze.solution.append(cell.coordinates)
        cell = cell.parentCell
        steps+=1
    return pathTaken,steps,notGoal
def AstarSearch(start, goal, maze : Maze):

    startCell = Cell(start, None)
    goalCell = Cell(goal, None)

    #initialize all values as 0 for start and end nodes

    startCell.gval = startCell.hval = startCell.fval = 0
    goalCell.gval = goalCell.hval = goalCell.fval = 0
    #print(startCell.coordinates, goalCell.coordinates)
    #create a priority queue for the open list

    openList = PriorityQueue()
    closedList = []
    openList.put(startCell)
    
    AdjacentRowIndex = [0, 1, 0, -1]
    AdjacentColumnIndex = [-1, 0, 1, 0]

    while(not openList.empty()):
        currentCell = openList.get()
        #print(currentCell.coordinates)
        if currentCell == goalCell:
            return tracePath(currentCell,maze) #A function to return the actual path
        #print("here1")
        closedList.append(currentCell)

        neighbourCells = [] #To explore the neighbours of the current cell
        for i in range(4):
            neighbour_x = currentCell.coordinates[0] + AdjacentRowIndex[i]
            neighbour_y = currentCell.coordinates[1] + AdjacentColumnIndex[i]
            if not maze.validity(neighbour_x,neighbour_y):
                #print("oops")
                continue
            neighbourCords = (neighbour_x,neighbour_y)
            neighbour = Cell(neighbourCords,currentCell)
            #print(neighbour.coordinates)
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
            neighbour.hval = abs(goalCell.coordinates[0] - neighbour.coordinates[0]) + abs(goalCell.coordinates[1] - neighbour.coordinates[1])
            neighbour.fval = neighbour.gval + neighbour.hval
            for openNeighbour in openList.queue:
                if neighbour.coordinates == openNeighbour.coordinates and neighbour.gval >= openNeighbour.gval:
                    weakerNeighbour = 1
                    break
            if weakerNeighbour:
                continue
            openList.put(neighbour)
    return tracePath(currentCell,maze,1)       



  
        

            


class Agent:

    def __init__(self,gridworld,goal):
        self.gridworld = gridworld
        self.goal = goal
        self.position = (0,0)
        self.moves = 0
        self.locations = {}
    def findForwardPath(self):
        path,steps,notGoal = AstarSearch(self.position,self.goal,self.gridworld)
        if notGoal:
            return notGoal
        return path
    def findBackwardPath(self):
        path,steps,notGoal = AstarSearch(self.goal,self.position,self.gridworld)
        if notGoal:
            return notGoal
        return path
    def findAdaptivePath(self):
        path,steps,notGoal,hvals = AdaptiveAstarSearch(self.position,self.goal,self.gridworld,self.locations)
        #print (hvals)
        for i in hvals:
                self.locations[i] = hvals[i]
        if notGoal:
            return notGoal
        return path

    def makeAdaptiveMoves(self, maze : Maze, path):
        for i in path[::-1]:
            if maze.maze[i[0]][i[1]] == 0:
                self.position = i
                self.moves+=1
            else:
                self.gridworld.maze[i[0]][i[1]] = 1
                break  
    def makeMoves(self, maze : Maze, path):
        for i in path[::-1]:
            if maze.maze[i[0]][i[1]] == 0:
                self.position = i
                self.moves+=1
            else:
                self.gridworld.maze[i[0]][i[1]] = 1
                break

def adaptiveTracePath(cell : Cell, maze : Maze, notGoal=0):
    pathTaken = []
    maze.solution = []
    steps = 0
    goalgval = cell.gval
    hvals = {}
    while cell is not None:
        pathTaken.append(cell.coordinates)
        hvals[cell.coordinates] = goalgval - cell.gval
        #maze.maze[cell.coordinates[0]][cell.coordinates[1]] = 2
        maze.solution.append(cell.coordinates)
        cell = cell.parentCell
        steps+=1
    return pathTaken,steps,notGoal,hvals

def AdaptiveAstarSearch(start, goal, maze : Maze, agentlocations):
    startCell = Cell(start, None)
    goalCell = Cell(goal, None)

    startCell.gval = startCell.hval = startCell.fval = 0
    goalCell.gval = goalCell.hval = goalCell.fval = 0

    openList = PriorityQueue()
    closedList = []
    openList.put(startCell)
    
    AdjacentRowIndex = [0, 1, 0, -1]
    AdjacentColumnIndex = [-1, 0, 1, 0]

    while(not openList.empty()):
        currentCell = openList.get()
        if currentCell == goalCell:
            return adaptiveTracePath(currentCell,maze)
        closedList.append(currentCell)
        neighbourCells = [] #To explore the neighbours of the current cell
        for i in range(4):
            neighbour_x = currentCell.coordinates[0] + AdjacentRowIndex[i]
            neighbour_y = currentCell.coordinates[1] + AdjacentColumnIndex[i]
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
            if neighbour.coordinates in agentlocations:
                neighbour.hval = agentlocations[neighbour.coordinates]
            else: 
                neighbour.hval = abs(goalCell.coordinates[0] - neighbour.coordinates[0]) + abs(goalCell.coordinates[1] - neighbour.coordinates[1])

            neighbour.fval = neighbour.gval + neighbour.hval
            for openNeighbour in openList.queue:
                if neighbour.coordinates == openNeighbour.coordinates and neighbour.gval >= openNeighbour.gval:
                    weakerNeighbour = 1
                    break
            if weakerNeighbour:
                continue
            openList.put(neighbour)
    return adaptiveTracePath(currentCell,maze,1)

        
        
        




        



if __name__ == "__main__":
    maze1 = Maze(50,50)
    maze1.generate_maze()
    maze1.visualize_maze()
    maze1.clearVisitedArray()
    # mazefile = open("mazefile.obj", 'wb')
    # pickle.dump(maze1,mazefile)
    # mazefilereader = open("mazefile.obj",'rb')
    # storedmaze = pickle.load(mazefilereader)
    # storedmaze.visualize_maze()
    start = (0,0)
    goal = (49,49)
    path,steps,notGoal = AstarSearch(start,goal,maze1)
    if(not notGoal):
        print(path,"in", steps," steps" )
    else:
        print("Goal is blocked by cells")
    # print(storedmaze.solution)
    maze1.visualize_maze()
    emptyworld = Maze(50,50)
    emptyworld.generateAgentMaze()
    agent1 = Agent(emptyworld,goal)
    while(agent1.position != goal):
        agentPath = agent1.findForwardPath()
        #agent1.gridworld.visualize_maze()
        #print(agentPath)
        #print(agent1.position)
        if agentPath == 1:
            print("Goal unreachable after ", agent1.moves," moves")
            break
        reversedAgentPath = agentPath[::-1]
        agent1.makeMoves(maze1,agentPath)
    agent1.gridworld.visualize_maze()
    if agentPath != 1:
        print("solved the maze in -", agent1.moves," moves with fog of war using Repeated Forward A*")
    emptyworld1 = Maze(50,50)
    emptyworld1.generateAgentMaze()
    agent2 = Agent(emptyworld1,goal)
    while(agent2.position != goal):
        agentPath = agent2.findAdaptivePath()
        #agent2.gridworld.visualize_maze()
        if agentPath == 1:
            print("Goal unreachable after ", agent2.moves," moves")
            break
        agent2.makeAdaptiveMoves(maze1,agentPath)
    if agentPath != 1:
        print("solved the maze in -", agent2.moves," moves with fog of war using Adaptive A*")



    agent1moves = 0
    agent2moves = 0
    solved = 0
    blocked = 0
    for i in range(10):
        maze = Maze(50,50)
        maze.generate_maze()
        maze.clearVisitedArray()
        path,steps,notGoal = AstarSearch(start,goal,maze)
        if(not notGoal):
            print ("Maze ",i," can solved with the path: in", steps," steps without fog of war" )
        else:
            print("Goal is blocked by cells")
        emptyworld = Maze(50,50)
        emptyworld.generateAgentMaze()
        agent1 = Agent(emptyworld,goal)
        while(agent1.position != goal):
            agentPath = agent1.findForwardPath()
            #print(agentPath)
            if agentPath == 1:
                print("Goal unreachable after ", agent1.moves," moves")
                blocked+=1
                break
            reversedAgentPath = agentPath[::-1]
            agent1.makeMoves(maze,agentPath)
        if agentPath != 1:
            print("solved the maze in -", agent1.moves," moves with fog of war using Repeated Forward A*")
            agent1moves += agent1.moves
            solved+=1
        
        #adaptive
        emptyworld1 = Maze(50,50)
        emptyworld1.generateAgentMaze()
        agent2 = Agent(emptyworld1,goal)
        while(agent2.position != goal):
            agentPath = agent2.findAdaptivePath()
            if agentPath == 1:
                print("Goal unreachable after ", agent2.moves," moves")
                break
            agent2.makeAdaptiveMoves(maze,agentPath)
        if agentPath != 1:
            print("solved the maze in -", agent2.moves," moves with fog of war using Adaptive A*")
            agent2moves += agent2.moves
    print("I solved ",solved," mazes and the other ",blocked," were blocked :) thanks to Goutham")
    print("agent 1 took ", agent1moves/100, " moves on average")
    print("agent 2 took ", agent2moves/100, " moves on average")
    # maze3 = maze2 = maze1
    # maze2.dfsolver()
    # maze2.visualize_maze()
    # maze3.bfsolver()
    # maze3.visualize_maze()
    #maze2 = Maze(5,5)
    #maze2.generateAgentMaze()
    #maze2.visualize_maze()
    # maze1 = maze
    # maze1.dfsolver()
    # maze1.visualize_maze()
