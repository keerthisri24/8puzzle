import sys
import numpy as np

class Node:
    def __init__(self, state, parent, depth):
        self.state = state
        self.parent = parent
        self.depth = depth

class NodeA:
    def __init__(self, state, parent, depth, distance, heuristic):
        self.state = state
        self.parent = parent
        self.depth = depth
        self.distance = distance
        self.heuristic = heuristic

class Stack:
    def __init__(self):
        self.nodeStack = []

    def add(self, node):
        self.nodeStack.append(node)

    def length(self):
        return len(self.nodeStack)

    def empty(self):
        return len(self.nodeStack) == 0

    def contains_state(self, state):
        for node in self.nodeStack:
            return (node.state[0] == state[0]).all()

    def pop(self):
        if self.empty():
            raise Exception("Empty Frontier")
        else:
            node = self.nodeStack[0]
            self.nodeStack = self.nodeStack[1:]
            return node

class PriorityQueue:
    def __init__(self):
        self.nodeQueue = []

    def add(self, node):
        self.nodeQueue.append(node)

    def length(self):
        return len(self.nodeQueue)

    def empty(self):
        return len(self.nodeQueue) == 0

    def contains_state(self, state):
        for node in self.nodeQueue:
            return (node.state[0] == state[0]).all()

    def pop(self):
        min_idx = 0
        for i in range(len(self.nodeQueue)):
            if (self.nodeQueue[i].distance + self.nodeQueue[i].heuristic) < (self.nodeQueue[min_idx].distance + self.nodeQueue[min_idx].heuristic):
                min_idx = i
        item = self.nodeQueue[min_idx]
        del self.nodeQueue[min_idx]
        return item

    def delete(self, state):
        for i in range(len(self.nodeQueue)):
            if self.nodeQueue[i].state[0] == state[0]:
                del self.nodeQueue[i]
    
    def checkDistance(self, state, distance):
        for i in range(len(self.nodeQueue)):
            if self.nodeQueue[i].state[0] == state[0]:
                if self.nodeQueue[i].distance < distance:
                    return False
                else:
                    return True
     
class DFS: #**********************************************************************************
    def __init__(self, start, goal):
        startIndex = (np.where(start == 0)[0][0], np.where(start == 0)[1][0])
        goalIndex = (np.where(goal == 0)[0][0], np.where(goal == 0)[1][0])
        self.start = [start, startIndex]
        self.goal = [goal, goalIndex] 
        self.solution = None

    def findNeighbors(self, state):
        mat, (row, col) = state
        neighbors = []

        # Up Neighbor
        if row > 0:
            matCopy = mat.copy()
            temp = matCopy[row-1][col]
            matCopy[row][col] = temp
            matCopy[row-1][col] = 0
        
            neighbors.append(([matCopy, (row - 1, col)]))

        # Down Neighbor
        if row < 2:
            matCopy = mat.copy()
            temp = matCopy[row+1][col]
            matCopy[row][col] = temp
            matCopy[row+1][col] = 0
        
            neighbors.append(([matCopy, (row + 1, col)]))

        # Right Neighbor
        if col < 2:
            matCopy = mat.copy()
            temp = matCopy[row][col + 1]
            matCopy[row][col] = temp
            matCopy[row][col+1] = 0
        
            neighbors.append(([matCopy, (row, col+1)]))

        # Left Neighbor
        if col > 0:
            matCopy = mat.copy()
            temp = matCopy[row][col-1]
            matCopy[row][col] = temp
            matCopy[row][col-1] = 0
        
            neighbors.append(([matCopy, (row, col-1)]))

        return neighbors

    def contains_markedState(self, state):
        for i in self.marked:
            if (i[0] == state[0]).all():
                return False
        return True

    def solve(self):
        limit = 10
        self.enqueued = 0

        start = Node(state=self.start, parent=None, depth=0)
        nodeStack = Stack()
        nodeStack.add(start)

        self.marked = [] 

        while True:
              if nodeStack.empty():
                self.solution = None
                self.print()
                return 
                
              node = nodeStack.pop()

              if (node.state[0] == self.goal[0]).all():
                cells = []
                while node.parent is not None:
                    cells.append(node.state)
                    node = node.parent
                cells.reverse()
                self.solution = cells
                self.print()
                return

              self.marked.append(node.state)

              if node.depth < limit:
                for state in self.findNeighbors(node.state):
                  if not nodeStack.contains_state(state) and self.contains_markedState(state):
                      child = Node(state=state, parent=node, depth=node.depth+1)
                      nodeStack.add(child)
                      self.enqueued +=1

    def print(self):
        if self.solution is not None:
            solution = self.solution 
            print("DFS Solution Found\n")
            print("Number of States Enqueued: ", self.enqueued, "\n")
            print("Number of Moves: ", len(solution), "\n")
            print("Solution:\n ")
            tempStart = self.start[0].copy()
            tempStart = tempStart.tolist()
            for i in range(0,3):
                for j in range(0,3):
                    if tempStart[i][j] == 0:
                        tempStart[i][j] = '*'
            tempStart = np.array(tempStart)
            print(tempStart, "\n")
            for cell in zip(solution):
                temp = cell[0][0].copy()
                temp = temp.tolist()
                for i in range(0,3):
                    for j in range(0,3):
                        if temp[i][j] == 0:
                            temp[i][j] = '*'
                temp = np.array(temp)
                print(temp, "\n")
        else: 
            print("DFS No Solution Found within Depth 10\n")

class IDS: #**********************************************************************************
    def __init__(self, start, goal):
        startIndex = (np.where(start == 0)[0][0], np.where(start == 0)[1][0])
        goalIndex = (np.where(goal == 0)[0][0], np.where(goal == 0)[1][0])
        self.start = [start, startIndex]
        self.goal = [goal, goalIndex] 
        self.solution = None

    def findNeighbors(self, state):
        mat, (row, col) = state
        neighbors = []

        # Up Neighbor
        if row > 0:
            matCopy = mat.copy()
            temp = matCopy[row-1][col]
            matCopy[row][col] = temp
            matCopy[row-1][col] = 0
        
            neighbors.append(([matCopy, (row - 1, col)]))

        # Down Neighbor
        if row < 2:
            matCopy = mat.copy()
            temp = matCopy[row+1][col]
            matCopy[row][col] = temp
            matCopy[row+1][col] = 0
        
            neighbors.append(([matCopy, (row + 1, col)]))

        # Right Neighbor
        if col < 2:
            matCopy = mat.copy()
            temp = matCopy[row][col + 1]
            matCopy[row][col] = temp
            matCopy[row][col+1] = 0
        
            neighbors.append(([matCopy, (row, col+1)]))

        # Left Neighbor
        if col > 0:
            matCopy = mat.copy()
            temp = matCopy[row][col-1]
            matCopy[row][col] = temp
            matCopy[row][col-1] = 0
        
            neighbors.append(([matCopy, (row, col-1)]))

        return neighbors

    def contains_markedState(self, state):
        for i in self.marked:
            if (i[0] == state[0]).all():
                return False
        return True

    def idsUtil(self, limit):
        self.enqueued = 0

        start = Node(state=self.start, parent=None, depth=0)
        nodeStack = Stack()
        nodeStack.add(start)

        self.marked = [] 

        while True:
            if nodeStack.empty():
                self.solution = None
                self.print(limit)
                return

            node = nodeStack.pop()

            if (node.state[0] == self.goal[0]).all():
                cells = []
                while node.parent is not None:
                    cells.append(node.state)
                    node = node.parent
                cells.reverse()
                self.solution = cells
                self.print(limit)
                return True

            self.marked.append(node.state)

            if node.depth < limit:
                for state in self.findNeighbors(node.state):
                    if not nodeStack.contains_state(state) and self.contains_markedState(state):
                        child = Node(state=state, parent=node, depth=node.depth+1)
                        nodeStack.add(child)
                        self.enqueued += 1
                
    def solve(self):
        for x in range(1, 11):
            if self.idsUtil(x):
                break

    def print(self, limit):
      if self.solution is not None:
        solution = self.solution 
        print("IDS Depth ", limit, ": Solution Found\n")
        print("Number of States Enqueued: ", self.enqueued, "\n")
        print("Number of Moves: ", len(solution), "\n")
        print("Solution:\n ")
        tempStart = self.start[0].copy()
        tempStart = tempStart.tolist()
        for i in range(0,3):
            for j in range(0,3):
                if tempStart[i][j] == 0:
                    tempStart[i][j] = '*'
        tempStart = np.array(tempStart)
        print(tempStart, "\n")
        for cell in zip(solution):
            temp = cell[0][0].copy()
            temp = temp.tolist()
            for i in range(0,3):
                for j in range(0,3):
                    if temp[i][j] == 0:
                        temp[i][j] = '*'
            temp = np.array(temp)
            print(temp, "\n")
      else: 
        print("IDS Depth ", limit, ": No Solution Found\n")

class AStar1: #**********************************************************************************
    def __init__(self, start, goal):
        startIndex = (np.where(start == 0)[0][0], np.where(start == 0)[1][0])
        goalIndex = (np.where(goal == 0)[0][0], np.where(goal == 0)[1][0])
        self.start = [start, startIndex]
        self.goal = [goal, goalIndex] 
        self.solution = None

    def findNeighbors(self, state):
        mat, (row, col) = state
        neighbors = []

        # Up Neighbor
        if row > 0:
            matCopy = mat.copy()
            temp = matCopy[row-1][col]
            matCopy[row][col] = temp
            matCopy[row-1][col] = 0
        
            neighbors.append(([matCopy, (row - 1, col)]))

        # Down Neighbor
        if row < 2:
            matCopy = mat.copy()
            temp = matCopy[row+1][col]
            matCopy[row][col] = temp
            matCopy[row+1][col] = 0
        
            neighbors.append(([matCopy, (row + 1, col)]))

        # Right Neighbor
        if col < 2:
            matCopy = mat.copy()
            temp = matCopy[row][col + 1]
            matCopy[row][col] = temp
            matCopy[row][col+1] = 0
        
            neighbors.append(([matCopy, (row, col+1)]))

        # Left Neighbor
        if col > 0:
            matCopy = mat.copy()
            temp = matCopy[row][col-1]
            matCopy[row][col] = temp
            matCopy[row][col-1] = 0
        
            neighbors.append(([matCopy, (row, col-1)]))

        return neighbors

    def contains_markedState(self, state):
        for i in self.marked:
            if (i[0] == state[0]).all():
                return False
        return True

    def solve(self):
        self.enqueued = 0
        limit = 10

        start_h1 = self.heuristic_check_wrong_tiles(self.start)
        start = NodeA(state=self.start, parent=None, depth=0, distance=0, heuristic=start_h1)
        nodeQueue = PriorityQueue()
        nodeQueue.add(start)

        self.marked = [] 

        while True:
            if nodeQueue.empty():
                self.solution = None
                self.print(limit)
                return

            node = nodeQueue.pop()

            if (node.state[0] == self.goal[0]).all():
                cells = []
                while node.parent is not None:
                    cells.append(node.state)
                    node = node.parent
                cells.reverse()
                self.solution = cells
                self.print(limit)
                return True

            self.marked.append(node.state)

            if node.depth < limit:
                for state in self.findNeighbors(node.state):
                    if not nodeQueue.contains_state(state) and self.contains_markedState(state):
                        h1 = self.heuristic_check_wrong_tiles(state)
                        child = NodeA(state=state, parent=node, depth=node.depth+1, 
                            distance=node.distance+1, heuristic = h1)
                        nodeQueue.add(child)
                        self.enqueued +=1
                    if nodeQueue.contains_state(state):
                        if not self.contains_markedState(state):
                            if nodeQueue.checkDistance(state, node.distance+1):
                                nodeQueue.delete(state)
                                h1 = self.heuristic_check_wrong_tiles(state)
                                child = NodeA(state=state, parent=node, depth=node.depth+1, 
                                    distance=node.distance+1, heuristic=h1)
                                nodeQueue.add(child)
                                self.enqueued +=1

    def heuristic_check_wrong_tiles(self, state):
        incorrect = 0
        for i in range(0,3):
            for j in range(0,3):
                if (self.goal[0][i][j] == 0):
                    continue
                if (self.goal[0][i][j] != state[0][i][j]).all():
                    incorrect +=1
        return incorrect

    def print(self, limit):
      if self.solution is not None:
        solution = self.solution 
        print("AStar1 Solution Found\n")
        print("Number of States Enqueued: ", self.enqueued, "\n")
        print("Number of Moves: ", len(solution), "\n")
        print("Solution:\n ")
        tempStart = self.start[0].copy()
        tempStart = tempStart.tolist()
        for i in range(0,3):
            for j in range(0,3):
                if tempStart[i][j] == 0:
                    tempStart[i][j] = '*'
        tempStart = np.array(tempStart)
        print(tempStart, "\n")
        for cell in zip(solution):
            temp = cell[0][0].copy()
            temp = temp.tolist()
            for i in range(0,3):
                for j in range(0,3):
                    if temp[i][j] == 0:
                        temp[i][j] = '*'
            temp = np.array(temp)
            print(temp, "\n")
      else: 
        print("AStar1 No Solution Found within Depth 10\n")

class AStar2: #**********************************************************************************
    def __init__(self, start, goal):
        startIndex = (np.where(start == 0)[0][0], np.where(start == 0)[1][0])
        goalIndex = (np.where(goal == 0)[0][0], np.where(goal == 0)[1][0])
        self.start = [start, startIndex]
        self.goal = [goal, goalIndex] 
        self.solution = None

    def findNeighbors(self, state):
        mat, (row, col) = state
        neighbors = []

        # Up Neighbor
        if row > 0:
            matCopy = mat.copy()
            temp = matCopy[row-1][col]
            matCopy[row][col] = temp
            matCopy[row-1][col] = 0
        
            neighbors.append(([matCopy, (row - 1, col)]))

        # Down Neighbor
        if row < 2:
            matCopy = mat.copy()
            temp = matCopy[row+1][col]
            matCopy[row][col] = temp
            matCopy[row+1][col] = 0
        
            neighbors.append(([matCopy, (row + 1, col)]))

        # Right Neighbor
        if col < 2:
            matCopy = mat.copy()
            temp = matCopy[row][col + 1]
            matCopy[row][col] = temp
            matCopy[row][col+1] = 0
        
            neighbors.append(([matCopy, (row, col+1)]))

        # Left Neighbor
        if col > 0:
            matCopy = mat.copy()
            temp = matCopy[row][col-1]
            matCopy[row][col] = temp
            matCopy[row][col-1] = 0
        
            neighbors.append(([matCopy, (row, col-1)]))

        return neighbors


    def contains_markedState(self, state):
        for i in self.marked:
            if (i[0] == state[0]).all():
                return False
        return True

    def solve(self):
        self.enqueued = 0
        limit = 10

        start_h2 = self.heuristic_manhattan(self.start)
        start = NodeA(state=self.start, parent=None, depth=0, distance=0, heuristic=start_h2)
        nodeQueue = PriorityQueue()
        nodeQueue.add(start)

        self.marked = [] 

        while True:
            if nodeQueue.empty():
                self.solution = None
                self.print(limit)
                return

            node = nodeQueue.pop()

            if (node.state[0] == self.goal[0]).all():
                cells = []
                while node.parent is not None:
                    cells.append(node.state)
                    node = node.parent
                cells.reverse()
                self.solution = cells
                self.print(limit)
                return True

            self.marked.append(node.state)

            if node.depth < limit:
                for state in self.findNeighbors(node.state):
                    if not nodeQueue.contains_state(state) and self.contains_markedState(state):
                        h2 = self.heuristic_manhattan(state)
                        child = NodeA(state=state, parent=node, depth=node.depth+1, 
                            distance=node.distance+1, heuristic = h2)
                        nodeQueue.add(child)
                        self.enqueued +=1
                    if nodeQueue.contains_state(state):
                        if not self.contains_markedState(state):
                            if nodeQueue.checkDistance(state, node.distance+1):
                                nodeQueue.delete(state)
                                h2 = self.heuristic_manhattan(state)
                                child = NodeA(state=state, parent=node, depth=node.depth+1, 
                                    distance=node.distance+1, heuristic=h2)
                                nodeQueue.add(child)
                                self.enqueued +=1

    def heuristic_manhattan(self, state):
        goalVal = []
        stateVal = []
        total = 0
        for i in range(0,3):
            for j in range(0,3):
                goalVal.append(self.goal[0][i][j])
                stateVal.append(state[0][i][j])
        total = sum(abs(b%3 - g%3) + abs(b//3 - g//3)for b, g in ((stateVal.index(i), goalVal.index(i)) for i in range(1, 9)))
        return total
                       
    def print(self, limit):
      if self.solution is not None:
        solution = self.solution 
        print("AStar2 Solution Found\n")
        print("Number of States Enqueued: ", self.enqueued, "\n")
        print("Number of Moves: ", len(solution), "\n")
        print("Solution:\n ")
        tempStart = self.start[0].copy()
        tempStart = tempStart.tolist()
        for i in range(0,3):
            for j in range(0,3):
                if tempStart[i][j] == 0:
                    tempStart[i][j] = '*'
        tempStart = np.array(tempStart)
        print(tempStart, "\n")
        for cell in zip(solution):
            temp = cell[0][0].copy()
            temp = temp.tolist()
            for i in range(0,3):
                for j in range(0,3):
                    if temp[i][j] == 0:
                        temp[i][j] = '*'
            temp = np.array(temp)
            print(temp, "\n")
      else: 
        print("AStar2 No Solution Found within Depth 10\n")

# start = np.array([[6,7,1], [8, 2,0], [5,4,3]])
# goal = np.array([[7, 8, 1], [6, 0, 2], [5, 4, 3]])

#start = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
#goal = np.array([[2, 8, 1], [0, 4, 3], [7, 6, 5]])

start = []
line1 = []
with open(sys.argv[2], 'r') as f:
    for line in f:
        line = line.split() 
        if line:            
            #line = [int(i) for i in line]
            for i in line:
                if i == '*':
                    line1.append(0)
                else:
                    line1.append(int(i))
            start.append(line1)
    start = start[0]

start1=[]
start2 = []
start3 = []
for x in range(len(start)):
    if x <= 2:
        start1.append(start[x])
    if x > 2 and x <= 5:
        start2.append(start[x])
    if x > 5 and x <= 8:
        start3.append(start[x])
start = []
start.append(start1)
start.append(start2)
start.append(start3)

start = np.array(start)
goal = np.array([[7, 8, 1], [6, 0, 2], [5, 4, 3]])

if sys.argv[1] == "dfs":
    dfs = DFS(start, goal)
    dfs.solve()
if sys.argv[1] == "ids":
    ids = IDS(start, goal)
    ids.solve()
if sys.argv[1] == "astar1":
    astar1 = AStar1(start, goal)
    astar1.solve()
if sys.argv[1] == "astar2":
    astar2 = AStar2(start, goal)
    astar2.solve()