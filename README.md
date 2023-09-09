# 8puzzle

8 Puzzle is a game that consists of 3x3 numbered tiles from 1-8 where 1 tile is missing. The object of the game is to rearrange the start state to the goal state- which are a 
sequence of tiles you want the end result to look like. You can only slide squares into a spot that has a missing tile. For this project I used 3 different algorithms to solve 
this problem: Depth First Search (DFS), Iterative Deepening Search (IDS) and A*. For the A* algorithm, I used two different heuristics. The first one being the number of tiles in 
the wrong position and the second being the sum of manhattan distances. Through this project, I wanted to succesfully solve the puzzle while also determining which algorithm
can complete it in the fewest amount of steps and fewest number of states enqueued. 

Ultimately the A* algorithm with the sum of manhattan distances heuristic was the most efficient as it completed the puzzle in 5 steps and 11 states enqueued (note that these numbers will 
change depending on the start and goal state you choose to use). Please look at my Report for additional information and analysis on each of the algorithms and results.

This program runs through the command line. Here are the instructions to run the code:

1. Save 8puzzle.py. This python file contains all the code for the assignment.
2. Save input_file.txt. This text file contains the input for the program provided in the
instruction document. The content of the file looks like this: 6 7 1 8 2 * 5 4 3. These
numbers can be changed to provide different results.
3. Open command line. Navigate into the folder containing 8puzzle.py.
4. The format of the command you can type in is as follows:
     python 8puzzle.py <algorithm name> <input_file path>
Everything should be separated by a single space in the command. The algorithm name can take
on the following values without the quotes: “dfs”, “ids”, “astar1”, “astar2”. After typing in
the algorithm name, type in the path to where input_file.txt is saved.
Here is an example command: python homework1.py dfs /Users/name/input_file.txt
6. After typing the command and entering, the results will appear for that specific algorithm.
7. A screenshort of a sample output is below:
<img width="542" alt="Screen Shot 2023-09-09 at 3 45 29 PM" src="https://github.com/keerthisri24/8puzzle/assets/64601701/f6463261-1da7-4ec1-9f9e-aeca85cc008b">
