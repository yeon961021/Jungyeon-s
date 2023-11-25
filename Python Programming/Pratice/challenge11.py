# Solution for the simple random maze search 1 problem

import random

def setup_maze():
    """
    Setup an 8x8 maze.
    0 is a square we can move through, 1 is a rock
    """
    return \
        [ [0, 1, 0, 0, 1, 1, 0, 1], \
          [0, 1, 1, 0, 0, 1, 0, 1], \
          [0, 0, 0, 0, 0, 1, 0, 1], \
          [0, 0, 1, 0, 0, 1, 0, 1], \
          [1, 0, 1, 0, 0, 0, 0, 0], \
          [1, 0, 0, 0, 0, 1, 1, 0], \
          [0, 0, 1, 0, 0, 1, 0, 0], \
          [1, 1, 1, 1, 0, 0, 0, 0], \
        ]

def find_path(maze, dirction1):
    game = 1
    going = 1 # 1 = east 2 = west 3 = south 4 = north
    i = 0
    j = 0
    a = 0
    direction2 = [(0,0)]
    direction = []
    while game == 1:
        if going == 1:
            if j < len(maze) - 1 and maze[i][j+1] == 0:
                direction.append("Right")
                j += 1
                direction2.append((i,j))
            else:
                my_list = [2, 3, 4]
                a = int(random.choice(my_list))
                if a == 2:
                    going = 2
                elif a == 3:
                    going = 3
                elif a == 4:
                    going = 4
        elif going == 2:
            if j > 0 and maze[i][j-1] == 0:
                direction.append("Left")
                j -= 1
                direction2.append((i,j))
            else:
                my_list = [1, 3, 4]
                a = int(random.choice(my_list))
                if a == 1:
                    going = 1
                elif a == 3:
                    going = 3
                elif a == 4:
                    going = 4
        elif going == 3:
            if i < len(maze) - 1 and maze[i+1][j] == 0:
                direction.append("Down")
                i += 1
                direction2.append((i,j))
            else:
                my_list = [2, 4, 1]
                a = int(random.choice(my_list))
                if a == 2:
                    going = 2
                elif a == 4:
                    going = 4
                elif a == 1:
                    going = 1
        elif going == 4:
            if i > 0 and maze[i-1][j] == 0:
                direction.append("Up")
                i -= 1
                direction2.append((i,j))
            else:
                my_list = [2, 3, 1]
                a = int(random.choice(my_list))
                if a == 2:
                    going = 2
                elif a == 3:
                    going = 3
                elif a == 1:
                    going = 1
        if i == len(maze) - 1 and j == len(maze) - 1:
            game = 0
            print("We found the Exit!")
            
    direction1 = direction2
    return direction1

def display_path(path):
    count = len(path)
    print(f"{count} times search: ",path)


def main():
    maze = setup_maze()  # maze is a 2D list
    initial_location = 0, 0  # 2-tuple to hold current location as (row, column)

    path = find_path(maze)
    display_path(path)

if __name__ == "__main__":
    main()
