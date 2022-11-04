#By Goutham Swaminathan, Simran Singh and Faraz Ahmed
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
import minHeap
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
        elif(self.fval == other.fval): #breaking ties with greater g values given preference. This results in fewer expanded cells

            if (self.gval > other.gval):
                return 1
        return 0
    def __gt__(self,other):
        if (self.fval > other.fval):
            return 1



class Maze:
    
    
    def __init__(self, rows, columns):
 
        self.rows = rows
        self.columns = columns
        self.steps = 0
        self.AdjacentRowIndex = [0, 1, 0, -1]
        self.AdjacentColumnIndex = [-1, 0, 1, 0]
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
                neighbour_x = r + self.AdjacentRowIndex[i]
                neighbour_y = c + self.AdjacentColumnIndex[i]
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
                neighbour_x = r + self.AdjacentRowIndex[i]
                neighbour_y = c + self.AdjacentColumnIndex[i]
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
                neighbour_x = r + self.AdjacentRowIndex[i]
                neighbour_y = c + self.AdjacentColumnIndex[i]
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
                elif (i,j) in self.solution:
                    colormaze[i][j] = 2
                    k+=1
                else:
                    n+=1
        #print(m/(m+n))
        if k == 0:
            gridcolors = colors.ListedColormap(["white","black"])
        else:
            gridcolors = colors.ListedColormap(["white","black","red"])
        plot = plt.imshow(colormaze,cmap=gridcolors)
        plot.figure.savefig("plots/Agent" + datetime.utcnow().strftime("%d%H%M%S%f") + ".png")
        #plt.show()
def tracePath(cell : Cell, maze : Maze,expandedCells, notGoal=0):
    pathTaken = []
    maze.solution = []
    steps = 0
    while cell is not None:
        pathTaken.append(cell.coordinates)
        maze.solution.append(cell.coordinates)
        cell = cell.parentCell
        steps+=1
    return pathTaken,steps,notGoal,expandedCells
def AstarSearch(start, goal, maze : Maze):
    expandedCells = 0
    startCell = Cell(start, None)
    goalCell = Cell(goal, None)

    #initialize all values as 0 for start and end nodes

    startCell.gval = startCell.hval = startCell.fval = 0
    goalCell.gval = goalCell.hval = goalCell.fval = 0
    #create a priority queue for the open list

    openList = PriorityQueue()
    # openList = minHeap.minHeap()
    closedList = []
    openList.put(startCell)
    AdjacentRowIndex = [0, 1, 0, -1]
    AdjacentColumnIndex = [-1, 0, 1, 0]
    consistent = 1
    while(not openList.empty()):
        currentCell = openList.get()
        expandedCells+=1
        if currentCell == goalCell:
            # print(consistent,expandedCells)
            # if consistent == expandedCells:
            #     print("All cells in this A* search had consistent manhattan values")
            return tracePath(currentCell,maze,expandedCells) #A function to return the actual path
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
        neighbours = 0
        consistentneighbours = 0
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
            # if(neighbour.coordinates[0] > 10 and neighbour.coordinates[0] < 20 and neighbour.coordinates[1] < 40):
            #     neighbour.gval = currentCell.gval + 5
            # if(neighbour.coordinates[0] > 30 and neighbour.coordinates[0] < 40 and neighbour.coordinates[1] > 10):
            #     neighbour.gval = currentCell.gval + 5
            neighbour.hval = abs(goalCell.coordinates[0] - neighbour.coordinates[0]) + abs(goalCell.coordinates[1] - neighbour.coordinates[1])
            neighbour.fval = neighbour.gval + neighbour.hval
            for openNeighbour in openList.queue:
            # for openNeighbour in openList.heap:
                if neighbour.coordinates == openNeighbour.coordinates and neighbour.gval >= openNeighbour.gval:
                    weakerNeighbour = 1
                    break
            if weakerNeighbour:
                continue
            neighbours += 1
            if(currentCell.hval <= 1 + neighbour.hval):
                consistentneighbours += 1
            openList.put(neighbour)
        if (neighbours == consistentneighbours):
            consistent += 1
    return tracePath(currentCell,maze,expandedCells,1)       

class Agent:

    def __init__(self,gridworld,start,goal):
        self.gridworld = gridworld
        self.goal = goal
        self.position = start
        self.moves = 0
        self.locations = {}
    def findForwardPath(self):
        path,steps,notGoal,expandedCells = AstarSearch(self.position,self.goal,self.gridworld)
        if notGoal:
            return notGoal,expandedCells
        return path,expandedCells
    def findBackwardPath(self):
        path,steps,notGoal,expandedCells = AstarSearch(self.goal,self.position,self.gridworld)
        if notGoal:
            return notGoal, expandedCells
        return path, expandedCells
    def findAdaptivePath(self):
        path,steps,notGoal,hvals,expandedCells = AdaptiveAstarSearch(self.position,self.goal,self.gridworld,self.locations)
        for i in hvals:
                self.locations[i] = hvals[i]
        if notGoal:
            return notGoal,expandedCells
        return path,expandedCells

    def makeAdaptiveMoves(self, maze : Maze, path):
        for i in path[::-1]:
            if maze.maze[i[0]][i[1]] == 0:
                if (self.position != i):
                    self.moves+=1
                self.position = i
            else:
                self.gridworld.maze[i[0]][i[1]] = 1
                break  
    def makeMoves(self, maze : Maze, path):
        for i in path[::-1]:
            if maze.maze[i[0]][i[1]] == 0:
                if (self.position != i):
                    self.moves+=1
                self.position = i
            else:
                self.gridworld.maze[i[0]][i[1]] = 1
                break

def adaptiveTracePath(cell : Cell, maze : Maze, closedList,expandedCells, notGoal=0):
    pathTaken = []
    maze.solution = []
    steps = 0
    goalgval = cell.gval
    hvals = {}
    for node in closedList:
        hvals[node.coordinates] = goalgval - node.gval
    while cell is not None:
        pathTaken.append(cell.coordinates)
        maze.solution.append(cell.coordinates)
        cell = cell.parentCell
        steps+=1
    return pathTaken,steps,notGoal,hvals,expandedCells

def AdaptiveAstarSearch(start, goal, maze : Maze, agentlocations):
    expandedCells = 0
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
        expandedCells+=1
        if currentCell == goalCell:
            return adaptiveTracePath(currentCell,maze,closedList,expandedCells)
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
            # if(neighbour.coordinates[0] > 10 and neighbour.coordinates[0] < 20 and neighbour.coordinates[1] < 40):
            #     neighbour.gval = currentCell.gval + 5
            # if(neighbour.coordinates[0] > 30 and neighbour.coordinates[0] < 40 and neighbour.coordinates[1] > 10):
            #     neighbour.gval = currentCell.gval + 5
            
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
    return adaptiveTracePath(currentCell,maze,closedList,expandedCells, 1)

def ConvertToGif(stringval): 
    directory='plots'
    images=os.listdir(directory)
    filtered_images=[file for file in images if file.endswith('.png')]
    with imageio.get_writer('plots/' + stringval + '.gif', mode='I') as writer:
        for filename in filtered_images:
            image=imageio.imread(directory+'/'+filename)
            writer.append_data(image)
    for filename in filtered_images:
        filepath=os.path.join(directory, filename)
        if not filename[-3:] == "gif":
            os.remove(filepath)  
        
        
if __name__ == "__main__":
    logging.basicConfig(filename='ForwardvsBackward_.txt', level=logging.INFO, format='')
    # maze1 = Maze(50,50)
    # maze1.generate_maze()
    # maze1.clearVisitedArray()
    # maze1.maze[24][24] = 0
    # maze1.maze = [[0,0,0,0,0],[0,0,1,0,0],[0,0,1,1,0],[0,0,1,1,0],[0,0,0,1,0]]
    #mazefile = open("mazefile1.obj", 'wb')
    #pickle.dump(maze1,mazefile)
    mazefilereader = open("mazefile1.obj",'rb')
    maze1 = pickle.load(mazefilereader)
    maze1.visualize_maze()
    start = (0,0)
    goal = (49,49)

    path,steps,notGoal,expandedCells = AstarSearch(start,goal,maze1)
    if(not notGoal):
        print(path,"in", steps," steps" )
    else:
        print("Goal is blocked by cells")

    maze1.visualize_maze()

    
    #Agent for solving using repeated forward A*, for Repeated backward A* replace findForwardPath() with findBackwardPath()
    emptyworld = Maze(50,50)
    emptyworld.generateAgentMaze()
    agent1 = Agent(emptyworld,start,goal)
    TotalExpandedCells = 0
    iterations = 0
    starttime = time.time()
    while(agent1.position != goal):
        iterations += 1
        agentPath,expandedCells = agent1.findBackwardPath()
        agent1.gridworld.visualize_maze()
        TotalExpandedCells += expandedCells
        if agentPath == 1:
            print("Goal unreachable after ", agent1.moves," moves")
            logging.info(f"Goal unreachable after {agent1.moves} moves")
            break
        reversedAgentPath = agentPath[::-1]
        #use reversed agent path for Backward A*
        agent1.makeMoves(maze1,reversedAgentPath)

    endtime = time.time()
    ConvertToGif("repeatedForward")

    if agentPath != 1:
        print("solved the maze in -", agent1.moves," moves with fog of war using Repeated Forward A*")
        logging.info(f"solved the maze in - {agent1.moves} moves with fog of war using Repeated Forward A*")
    print(TotalExpandedCells)
    logging.info(TotalExpandedCells)
    logging.info(endtime - starttime)

    
    #Agent solves using Adaptive A*
    emptyworld1 = Maze(50,50)
    emptyworld1.generateAgentMaze()
    agent2 = Agent(emptyworld1,start,goal)
    TotalAexpandedCells = 0
    starttime = time.time()
    while(agent2.position != goal):
        agentPath,expandedCells= agent2.findAdaptivePath()
        TotalAexpandedCells += expandedCells
        agent2.gridworld.visualize_maze()
        if agentPath == 1:
            print("Goal unreachable after ", agent2.moves," moves")
            logging.info(f"Goal unreachable after {agent2.moves} moves")
            break
        agent2.makeAdaptiveMoves(maze1,agentPath)

    endtime = time.time()
    ConvertToGif("repeatedAdaptive")
    if agentPath != 1:
        print("solved the maze in -", agent2.moves," moves with fog of war using Adaptive A*")
        logging.info(f"solved the maze in - {agent2.moves} moves with fog of war using Adaptive A*")
    print(TotalAexpandedCells)
    logging.info(TotalAexpandedCells)
    logging.info(endtime - starttime)
    


    #Uncomment the code below and comment the code above to test adaptive and repeated forward A* for 500 random mazes




    # start = (0,0)
    # goal = (49,49)
    # agent1moves = 0
    # agent2moves = 0
    # solved = 0
    # blocked = 0
    # TotalForwardExpandedCells = 0
    # TotalAdaptiveExpandedCells = 0
    # timeTakenForward = 0
    # timeTakenAdaptive = 0
    # logging.basicConfig(filename='AdaptiveAstar_2.txt', level=logging.DEBUG, format='')
    # ub = 0
    # for i in range(500):
        
    #     maze = Maze(50,50)
    #     maze.generate_maze()
    #     m = 0
    #     n = 0
    #     for i in range(maze.rows):
    #         for j in range(maze.columns):
    #             if maze.maze[i][j] == 0:
    #                 m+=1
    #             else:
    #                 n+=1
    #     logging.info(f"Ratio of unblocked cells to total cells = {m/(m+n)}")
    #     logging.info(f"Number of unblocked cells = {m}")
    #     maze.clearVisitedArray()
    #     path,steps,notGoal,expandedCells = AstarSearch(start,goal,maze)
    #     if(not notGoal):
    #         logging.info(f"Maze {i} can solved with the path: in {steps} steps without fog of war" )
    #     else:
    #         logging.info("Goal is blocked by cells")
    #     emptyworld = Maze(50,50)
    #     emptyworld.generateAgentMaze()
    #     agent1 = Agent(emptyworld,start,goal)
    #     TotalExpandedCells = 0
    #     iterations = 0
    #     starttime = time.time()
    #     while(agent1.position != goal):
    #         iterations += 1
    #         agentPath,expandedCells = agent1.findForwardPath()
    #         TotalExpandedCells = TotalExpandedCells + expandedCells
    #         #print(agentPath)
    #         if agentPath == 1:
    #             logging.info(f"Goal unreachable after {agent1.moves} moves")
    #             if agent1.moves <= (m*m):
    #                 logging.info(f"Number of agent moves is = {agent1.moves} and this is less than the number off unblocked cells squared, which is = {m*m}")
    #             else:
    #                 ub+=1
    #             blocked+=1
    #             break
    #         reversedAgentPath = agentPath[::-1]
    #         agent1.makeMoves(maze,agentPath)
    #     endtime = time.time()
    #     timeTakenForward += (endtime - starttime)
    #     if agentPath != 1:
    #         logging.info(f"solved the maze in - {agent1.moves} moves with fog of war using Repeated Forward A*")
    #         agent1moves += agent1.moves
    #         solved+=1
    #     print(TotalExpandedCells)
    #     #adaptive
    #     TotalAexpandedCells = 0
    #     emptyworld1 = Maze(50,50)
    #     emptyworld1.generateAgentMaze()
    #     iterations = 0
    #     agent2 = Agent(emptyworld1,start,goal)
    #     starttime = time.time()
    #     while(agent2.position != goal):
    #         iterations += 1
    #         agentPath,expandedCells = agent2.findAdaptivePath()
    #         TotalAexpandedCells += expandedCells
    #         if agentPath == 1:
    #             logging.info(f"Goal unreachable after {agent2.moves} moves")
    #             break
    #         agent2.makeAdaptiveMoves(maze,agentPath)
    #     endtime = time.time()
    #     timeTakenAdaptive += (endtime - starttime)
    #     if agentPath != 1:
    #         logging.info(f"solved the maze in - {agent2.moves} moves with fog of war using Adaptive A*")
            
    #         agent2moves += agent2.moves
    #     print(TotalAexpandedCells)
    #     TotalForwardExpandedCells += TotalExpandedCells
    #     TotalAdaptiveExpandedCells += TotalAexpandedCells
    # logging.info(f"I solved {solved} mazes and the other {blocked} were blocked")
    # logging.info(f"agent 1 took {agent1moves/100} moves on average")
    # logging.info(f"Of the {blocked} blocked mazes, the number of moves by agent 1 were greater than the number of unblocked cells squared {ub} times.")
    # logging.info(f"agent 2 took {agent2moves/100} moves on average")
    # i = (1 - (TotalAdaptiveExpandedCells/TotalForwardExpandedCells)) * 100
    # logging.info(f"Adaptive A* expanded {i} percent less cells than forward A* on average")
    # j = (1 - (timeTakenAdaptive/timeTakenForward)) * 100
    # logging.info(f"Adaptive A* took {j} percent less time than forward A* on average")
    
    
