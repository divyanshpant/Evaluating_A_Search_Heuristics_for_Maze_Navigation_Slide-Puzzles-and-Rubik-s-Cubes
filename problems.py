from algorithms import MazeAStar
import random
import numpy as np
import cv2
from algorithms import PuzzleAStar
from termcolor import colored
import time
import copy
from comparisons import comparison
from algorithms import rubiks_asearch

class Maze:
    def __init__(self, dim):
        self.dim = dim
        self.maze = self._generate_maze()  # Generate the maze grid
        self.rows, self.cols = self.maze.shape
        self.comp = comparison()  # Object to log heuristic comparisons
        cv2.namedWindow("Maze Search", cv2.WINDOW_NORMAL)  # Set up a resizable window for visualization

    def _generate_maze(self):
        # Create a grid of size (2n+1) to allow walls between each cell
        maze = np.zeros((self.dim * 2 + 1, self.dim * 2 + 1), dtype=np.uint8)
        x, y = (0, 0)
        maze[2 * x + 1, 2 * y + 1] = 1  # Mark the starting cell as a path
        stack = [(x, y)]  # DFS stack

        # Generate the maze using recursive backtracking (DFS)
        while stack:
            x, y = stack[-1]
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)  # Shuffle directions to ensure random maze structure
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # Check bounds and if the target cell is still unvisited
                if (0 <= nx < self.dim and 0 <= ny < self.dim and 
                    maze[2 * nx + 1, 2 * ny + 1] == 0):
                    # Carve the path to the new cell
                    maze[2 * nx + 1, 2 * ny + 1] = 1
                    maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 1
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()

        # Define entry and exit points
        maze[1, 0] = 1
        maze[-2, -1] = 1
        return maze

    def display_search_state(self, cell, move, heuristic_name, depth, heuristic_value, delay=10):
        # This function visualizes each step of the search process in the maze
        scale = 30
        scaled_rows, scaled_cols = self.rows * scale, self.cols * scale
        img = self.maze * 255
        img = np.stack([img] * 3, axis=-1).astype(np.uint8)
        img = cv2.resize(img, (scaled_cols, scaled_rows), interpolation=cv2.INTER_NEAREST)

        # Highlight current cell in red
        x, y = cell
        cv2.circle(img, (y * scale + scale // 2, x * scale + scale // 2), scale // 4, (0, 0, 255), -1)
        # Start (green) and goal (blue) markers
        cv2.circle(img, (0 * scale + scale // 2, 1 * scale + scale // 2), scale // 3, (0, 255, 0), -1)
        cv2.circle(img, ((self.cols - 1) * scale + scale // 2, (self.rows - 2) * scale + scale // 2), 
                   scale // 3, (255, 0, 0), -1)

        # Display heuristic name, current depth, and heuristic value on the top-right
        text1 = f"Heuristic: {heuristic_name}"
        text2 = f"Depth: {depth}"
        text3 = f"Heuristic Value: {heuristic_value:.2f}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1
        color = (255, 255, 255)
        margin = 10
        text_x = scaled_cols - max(
            cv2.getTextSize(text1, font, font_scale, thickness)[0][0],
            cv2.getTextSize(text2, font, font_scale, thickness)[0][0],
            cv2.getTextSize(text3, font, font_scale, thickness)[0][0]
        ) - margin
        text1_y = margin + cv2.getTextSize(text1, font, font_scale, thickness)[0][1]
        text2_y = text1_y + cv2.getTextSize(text2, font, font_scale, thickness)[0][1] + 10
        text3_y = text2_y + cv2.getTextSize(text3, font, font_scale, thickness)[0][1] + 10

        # Put text on image
        cv2.putText(img, text1, (text_x, text1_y), font, font_scale, color, thickness)
        cv2.putText(img, text2, (text_x, text2_y), font, font_scale, color, thickness)
        cv2.putText(img, text3, (text_x, text3_y), font, font_scale, color, thickness)

        # Show the image and wait for delay
        cv2.startWindowThread()
        cv2.imshow("Maze Search", img)
        cv2.resizeWindow("Maze Search", scaled_cols, scaled_rows)
        key = cv2.waitKey(delay)
        if key == 27:  # ESC key pressed to interrupt
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Interrupted by ESC")

    def solve(self):
        # Defines start and goal coordinates for solving the maze
        start = (1, 0)
        goal = (self.rows - 2, self.cols - 1)
        astar = MazeAStar(self.maze, visualizer=self)

        try:
            # First heuristic: Manhattan
            print(colored("\nSolving with Manhattan Heuristic:", "blue"))
            start_time = time.time()
            heuristic_manhattan = astar.manhattan(start, goal)
            path_manhattan, depth_manhattan, _, nodes_manhattan = astar.a_star(start, goal, heuristic_name="manhattan")
            cv2.destroyWindow("Maze Search")

            if path_manhattan:
                print(colored("Solution found with Manhattan Heuristic:", "green"))
                solution_path = self.visualize_solution(path_manhattan, "Manhattan", delay=1000)
            else:
                print(colored("No solution found with Manhattan Heuristic.", "red"))
                solution_path = []

            time_manhattan = time.time() - start_time
            self.comp.add_comparison({
                "heuristic_name": "Manhattan",
                "time_taken": time_manhattan,
                "steps": depth_manhattan,
                "solution": solution_path,
                "heuristic_value": heuristic_manhattan,
                "nodes_evaluated": nodes_manhattan
            })

            # Second heuristic: Dead-End Avoidance
            print(colored("\nSolving with Dead-End Avoidance Heuristic:", "blue"))
            start_time = time.time()
            heuristic_deadend = astar.dead_end_avoidance_heuristic(start, goal)
            path_deadend, depth_deadend, _, nodes_deadend = astar.a_star(start, goal, heuristic_name="dead_end_avoidance")
            cv2.destroyWindow("Maze Search")

            if path_deadend:
                print(colored("Solution found with Dead-End Avoidance Heuristic:", "green"))
                solution_path = self.visualize_solution(path_deadend, "Dead-End Avoidance", delay=0)
            else:
                print(colored("No solution found with Dead-End Avoidance Heuristic.", "red"))
                solution_path = []

            time_deadend = time.time() - start_time
            self.comp.add_comparison({
                "heuristic_name": "Dead-End Avoidance",
                "time_taken": time_deadend,
                "steps": depth_deadend,
                "solution": solution_path,
                "heuristic_value": heuristic_deadend,
                "nodes_evaluated": nodes_deadend
            })

            # Display comparison summary of both heuristics
            self.comp.display_comparisons()

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            cv2.destroyAllWindows()

    def visualize_solution(self, path, heuristic_name, delay=1000):
        # Draws the final solution path between visited nodes
        scale = 30
        scaled_rows, scaled_cols = self.rows * scale, self.cols * scale
        img = self.maze * 255
        img = np.stack([img] * 3, axis=-1).astype(np.uint8)
        img = cv2.resize(img, (scaled_cols, scaled_rows), interpolation=cv2.INTER_NEAREST)
        solution_path = []

        # Draw lines between parent-child nodes in the path
        for start_cell, end_cell in path.items():
            cv2.line(img, 
                     (start_cell[1] * scale + scale // 2, start_cell[0] * scale + scale // 2), 
                     (end_cell[1] * scale + scale // 2, end_cell[0] * scale + scale // 2), 
                     (0, 0, 255), 5)
            solution_path.append((start_cell, end_cell))

        # Highlight start and goal points
        cv2.circle(img, (0 * scale + scale // 2, 1 * scale + scale // 2), 
                   scale // 3, (0, 255, 0), -1)
        cv2.circle(img, ((self.cols - 1) * scale + scale // 2, (self.rows - 2) * scale + scale // 2), 
                   scale // 3, (255, 0, 0), -1)

        # Display the heuristic used in the visualization
        text = f"Heuristic: {heuristic_name}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1
        color = (255, 255, 255)
        margin = 10
        text_x = scaled_cols - cv2.getTextSize(text, font, font_scale, thickness)[0][0] - margin
        text_y = margin + cv2.getTextSize(text, font, font_scale, thickness)[0][1]
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

        # Display and close window after delay
        cv2.namedWindow("Maze Solution", cv2.WINDOW_NORMAL)
        cv2.imshow("Maze Solution", img)
        cv2.resizeWindow("Maze Solution", scaled_cols, scaled_rows)
        cv2.waitKey(delay)
        cv2.destroyWindow("Maze Solution")

        return solution_path

class Puzzle:
    def __init__(self, size):
        self.size = size  # Puzzle size (e.g., 3 for 3x3)
        self.goal_state = list(range(1, self.size * self.size)) + [0]  # Goal configuration with blank tile (0) at the end
        self.initial_state = self.scramble()  # Randomized starting state
        self.comp = comparison()  # Object to store and display heuristic performance comparisons

    def scramble(self, steps=100):
        # Adjust number of scrambling steps based on puzzle size
        if self.size == 3:
            steps = 100
        elif self.size == 4:
            steps = 80
        else:
            steps = 50

        state = self.goal_state.copy()  # Start from the solved state
        blank_pos = len(state) - 1  # Initial blank tile position
        moves = ['U', 'D', 'L', 'R']  # Possible directions

        for _ in range(steps):
            valid_moves = []
            # Determine valid moves based on current blank position
            if blank_pos >= self.size:
                valid_moves.append('U')
            if blank_pos < len(state) - self.size:
                valid_moves.append('D')
            if blank_pos % self.size != 0:
                valid_moves.append('L')
            if blank_pos % self.size != self.size - 1:
                valid_moves.append('R')
            move = random.choice(valid_moves)  # Pick a random valid move
            new_blank_pos = blank_pos
            # Calculate new blank position based on move
            if move == 'U':
                new_blank_pos -= self.size
            elif move == 'D':
                new_blank_pos += self.size
            elif move == 'L':
                new_blank_pos -= 1
            elif move == 'R':
                new_blank_pos += 1
            # Swap tiles
            state[blank_pos], state[new_blank_pos] = state[new_blank_pos], state[blank_pos]
            blank_pos = new_blank_pos
        return state

    def print_board(self, board):
        # Print the board state in a readable format using ASCII characters
        print("+" + "---+" * self.size)
        for row in range(0, self.size * self.size, self.size):
            row_visual = "|"
            for tile in board[row:row + self.size]:
                if tile == 0:
                    row_visual += f" {colored(' ', 'cyan')} |"
                else:
                    row_visual += f" {colored(str(tile).rjust(2), 'yellow')} |"
            print(row_visual)
            print("+" + "---+" * self.size)

    def display(self, board, move=None, heuristic_name=None, delay=1000):
        # Visual representation of the puzzle board using OpenCV
        cell_size = 100
        puzzle_width = cell_size * self.size
        text_area_width = 300
        img_width = puzzle_width + text_area_width
        img_height = cell_size * self.size
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2 if self.size <= 4 else 1.5
        thickness = 3 if self.size <= 4 else 2

        # Draw puzzle tiles
        for i in range(self.size):
            for j in range(self.size):
                tile = board[i * self.size + j]
                x, y = j * cell_size, i * cell_size
                color = (0, 255, 255) if tile == 0 else (200, 200, 200)
                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), (0, 0, 0), 1)
                cv2.rectangle(img, (x + 2, y + 2), (x + cell_size - 2, y + cell_size - 2), color, -1)
                if tile != 0:
                    text = str(tile)
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = x + (cell_size - text_width) // 2
                    text_y = y + (cell_size + text_height) // 2
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        # Optional: Display move and heuristic name
        if heuristic_name:
            text1 = f"Heuristic: {heuristic_name}"
            text2 = f"Move: {move}" if move else "Initial State"
            font_scale_small = 0.6
            thickness_small = 1
            color = (255, 255, 255)
            margin = 10
            text_x = puzzle_width + margin
            (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale_small, thickness_small)
            (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale_small, thickness_small)
            text1_y = margin + text1_height
            text2_y = text1_y + text2_height + 10
            cv2.putText(img, text1, (text_x, text1_y), font, font_scale_small, color, thickness_small)
            cv2.putText(img, text2, (text_x, text2_y), font, font_scale_small, color, thickness_small)

        # Show image in resizable window
        cv2.namedWindow(f"{self.size}x{self.size} Puzzle", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{self.size}x{self.size} Puzzle", img)
        cv2.resizeWindow(f"{self.size}x{self.size} Puzzle", img_width, img_height)
        key = cv2.waitKey(delay)
        if key == 27:  # ESC key to interrupt
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Interrupted by ESC")

    def display_search_state(self, board, move, heuristic_name, depth, heuristic_value, delay=10):
        cell_size = 100
        puzzle_width = cell_size * self.size
        text_area_width = 300
        img_width = puzzle_width + text_area_width
        img_height = cell_size * self.size
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2 if self.size <= 4 else 1.5
        thickness = 3 if self.size <= 4 else 2
        for i in range(self.size):
            for j in range(self.size):
                tile = board[i * self.size + j]
                x, y = j * cell_size, i * cell_size
                color = (0, 255, 255) if tile == 0 else (200, 200, 200)
                cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), (0, 0, 0), 1)
                cv2.rectangle(img, (x + 2, y + 2), (x + cell_size - 2, y + cell_size - 2), color, -1)
                if tile != 0:
                    text = str(tile)
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = x + (cell_size - text_width) // 2
                    text_y = y + (cell_size + text_height) // 2
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        text1 = f"Heuristic: {heuristic_name}"
        text2 = f"Move: {move}" if move else "Initial State"
        text3 = f"Depth: {depth}"
        text4 = f"Heuristic Value: {heuristic_value:.2f}"
        font_scale_small = 0.6
        thickness_small = 1
        color = (255, 255, 255)
        margin = 10
        text_x = puzzle_width + margin
        (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale_small, thickness_small)
        (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale_small, thickness_small)
        (text3_width, text3_height), _ = cv2.getTextSize(text3, font, font_scale_small, thickness_small)
        (text4_width, text4_height), _ = cv2.getTextSize(text4, font, font_scale_small, thickness_small)
        text1_y = margin + text1_height
        text2_y = text1_y + text2_height + 10
        text3_y = text2_y + text3_height + 10
        text4_y = text3_y + text4_height + 10
        cv2.putText(img, text1, (text_x, text1_y), font, font_scale_small, color, thickness_small)
        cv2.putText(img, text2, (text_x, text2_y), font, font_scale_small, color, thickness_small)
        cv2.putText(img, text3, (text_x, text3_y), font, font_scale_small, color, thickness_small)
        cv2.putText(img, text4, (text_x, text4_y), font, font_scale_small, color, thickness_small)
        cv2.namedWindow(f"{self.size}x{self.size} Puzzle", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{self.size}x{self.size} Puzzle", img)
        cv2.resizeWindow(f"{self.size}x{self.size} Puzzle", img_width, img_height)
        key = cv2.waitKey(delay)
        if key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Interrupted by ESC")

    def solve(self):
        # Run A* search with two heuristics and compare results
        solver = PuzzleAStar(self.size, visualizer=self)
        try:
            print(colored("Initial State:", "blue"))
            self.print_board(self.initial_state)
            self.display(self.initial_state, heuristic_name="Initial")

            # Run A* using the Misplaced Tile heuristic
            print(colored("\nSolving with Misplaced Tile Heuristic:", "blue"))
            start_time = time.time()
            heuristic_misplaced = solver.misplaced_tile_heuristic(self.initial_state, self.goal_state)
            solution_misplaced, depth_misplaced, _, nodes_misplaced = solver.a_star(
                self.initial_state, self.goal_state, heuristic_name="Misplaced Tile"
            )
            if solution_misplaced:
                print(colored("Solution found with Misplaced Tile Heuristic:", "green"))
                solution_path = self.visualize_solution(solution_misplaced, "Misplaced Tile")
            else:
                print(colored("No solution found with Misplaced Tile Heuristic.", "red"))
                solution_path = None
            time_misplaced = time.time() - start_time

            self.comp.add_comparison({
                "heuristic_name": "Misplaced Tile",
                "time_taken": time_misplaced,
                "steps": depth_misplaced,
                "solution": solution_path,
                "heuristic_value": heuristic_misplaced,
                "nodes_evaluated": nodes_misplaced
            })

            # Run A* using the Dynamic Weighted heuristic
            print(colored("\nSolving with Dynamic Weighted Heuristic:", "blue"))
            start_time = time.time()
            heuristic_dynamic = solver.dynamic_weighted_heuristic(self.initial_state, self.goal_state)
            solution_dynamic, depth_dynamic, _, nodes_dynamic = solver.a_star(
                self.initial_state, self.goal_state, heuristic_name="Dynamic Weighted"
            )
            if solution_dynamic:
                print(colored("Solution found with Dynamic Weighted Heuristic:", "green"))
                solution_path = self.visualize_solution(solution_dynamic, "Dynamic Weighted")
            else:
                print(colored("No solution found with Dynamic Weighted Heuristic.", "red"))
                solution_path = None
            time_dynamic = time.time() - start_time

            self.comp.add_comparison({
                "heuristic_name": "Dynamic Weighted",
                "time_taken": time_dynamic,
                "steps": depth_dynamic,
                "solution": solution_path,
                "heuristic_value": heuristic_dynamic,
                "nodes_evaluated": nodes_dynamic
            })

            self.comp.display_comparisons()  # Show a summary of both runs

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_solution_path(self, solution):
        # Reconstruct the path from the goal node back to the start
        path = []
        current = solution
        while current:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def print_solution(self, solution):
        # Print each step in the solution path
        path = self.get_solution_path(solution)
        for step in path:
            if step.move:
                print(f"Move: {step.move}")
                self.print_board(step.board)

    def visualize_solution(self, solution, heuristic_name):
        # Visually walk through each move of the solution
        path = self.get_solution_path(solution)
        solution_path = []
        for step in path:
            if step.move:
                print(f"Move: {step.move}")
                self.print_board(step.board)
                self.display(step.board, move=step.move, heuristic_name=heuristic_name)
                solution_path.append(step.move)
        return solution_path


class Cube:
    def __init__(self, size):
        # Initialize a solved cube of the given size (e.g., 3x3, 4x4)
        # Each face is filled with a unique number from 0 to 5
        self.size = size
        self.state = [[i] * (size * size) for i in range(6)]

    def is_solved(self):
        # Returns True if every face has all identical stickers
        return all(all(sticker == face[0] for sticker in face) for face in self.state)

    def copy(self):
        # Create a deep copy of the cube (used for search algorithms)
        return copy.deepcopy(self)

    def rotate_face(self, face):
        # Rotates a single face 90 degrees clockwise
        n = self.size
        s = self.state[face]
        rotated = [0] * (n * n)
        for i in range(n):
            for j in range(n):
                rotated[j * n + (n - 1 - i)] = s[i * n + j]
        self.state[face] = rotated

    def move(self, move):
        # Performs a move on the cube (U, D, L, R, F, B with optional primes for counterclockwise)
        # Counterclockwise moves are handled by repeating clockwise move 3 times
        if move == 'U': self._move_U()                          # Clockwise up side
        elif move == "U'": [self._move_U() for _ in range(3)]   # Counterclockwise up side
        elif move == 'D': self._move_D()                        # Clockwise down side
        elif move == "D'": [self._move_D() for _ in range(3)]   # Counterclockwise down side
        elif move == 'L': self._move_L()                        # Clockwise left side
        elif move == "L'": [self._move_L() for _ in range(3)]   # Counterclockwise left side
        elif move == 'R': self._move_R()                        # Clockwise right side
        elif move == "R'": [self._move_R() for _ in range(3)]   # Counterclockwise right side
        elif move == 'F': self._move_F()                        # Clockwise front side
        elif move == "F'": [self._move_F() for _ in range(3)]   # Counterclockwise front side
        elif move == 'B': self._move_B()                        # Clockwise back side
        elif move == "B'": [self._move_B() for _ in range(3)]   # Counterclockwise back side

    # The following methods apply clockwise rotations and update the adjacent faces accordingly

    def _move_U(self):
        n = self.size
        self.rotate_face(0)
        s = self.state
        temp = s[1][0:n]
        s[1][0:n] = s[2][0:n]
        s[2][0:n] = s[3][0:n]
        s[3][0:n] = s[4][0:n]
        s[4][0:n] = temp

    def _move_D(self):
        n = self.size
        self.rotate_face(5)
        s = self.state
        temp = s[1][n*(n-1):n*n]
        s[1][n*(n-1):n*n] = s[4][n*(n-1):n*n]
        s[4][n*(n-1):n*n] = s[3][n*(n-1):n*n]
        s[3][n*(n-1):n*n] = s[2][n*(n-1):n*n]
        s[2][n*(n-1):n*n] = temp

    def _move_L(self):
        n = self.size
        self.rotate_face(1)
        s = self.state
        temp = [s[0][i*n] for i in range(n)]
        for i in range(n):
            s[0][i*n] = s[2][i*n]
            s[2][i*n] = s[5][i*n]
            s[5][i*n] = s[4][n*n - 1 - i*n]
            s[4][n*n - 1 - i*n] = temp[i]

    def _move_R(self):
        n = self.size
        self.rotate_face(3)
        s = self.state
        temp = [s[0][i*n + n - 1] for i in range(n)]
        for i in range(n):
            s[0][i*n + n - 1] = s[4][n*n - 1 - (i*n + n - 1)]
            s[4][n*n - 1 - (i*n + n - 1)] = s[5][i*n + n - 1]
            s[5][i*n + n - 1] = s[2][i*n + n - 1]
            s[2][i*n + n - 1] = temp[i]

    def _move_F(self):
        n = self.size
        self.rotate_face(2)
        s = self.state
        temp = [s[0][n*(n-1) + i] for i in range(n)]
        for i in range(n):
            s[0][n*(n-1) + i] = s[1][n*n - 1 - i*n]
            s[1][n*n - 1 - i*n] = s[5][i]
            s[5][i] = s[3][i*n]
            s[3][i*n] = temp[i]

    def _move_B(self):
        n = self.size
        self.rotate_face(4)
        s = self.state
        temp = [s[0][i] for i in range(n)]
        for i in range(n):
            s[0][i] = s[3][n*n - 1 - i*n]
            s[3][n*n - 1 - i*n] = s[5][n*n - 1 - i]
            s[5][n*n - 1 - i] = s[1][i*n]
            s[1][i*n] = temp[i]

    # Hashing and equality functions to support sets, dictionaries, and priority queues

    def __hash__(self):
        return hash(str(self.state))

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return str(self.state) < str(other.state)

    def scramble(self, steps=8):
        # Scrambles the cube with a number of random moves based on cube size
        if self.size == 3:
            steps = 6
        elif self.size == 4:
            steps = 5
        else:
            steps = 2
        print(f"Scrambling cube size {self.size} with {steps} steps")
        moves = ['U', "U'", 'D', "D'", 'L', "L'", 'R', "R'", 'F', "F'", 'B', "B'"]
        for _ in range(steps):
            self.move(random.choice(moves))

    def display(self, delay=1000, heuristic_name=None, move=None):
        # Draws the cube on screen using OpenCV
        # Optionally includes move and heuristic info for visualization during search
        color_map = {
            0: (255, 255, 255),  # White (U)
            1: (0, 0, 255),      # Blue (L)
            2: (0, 255, 0),      # Green (F)
            3: (0, 165, 255),    # Orange (R)
            4: (0, 255, 255),    # Yellow (B)
            5: (255, 0, 0)       # Red (D)
        }
        n = self.size
        face_positions = [
            (n, 0),   # U
            (0, n),   # L
            (n, n),   # F
            (2*n, n), # R
            (3*n, n), # B
            (n, 2*n)  # D
        ]
        img_height = 3 * n * 60
        img_width = 4 * n * 60
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 0
        for face_id, face in enumerate(self.state):
            x_off, y_off = face_positions[face_id]
            for i in range(n):
                for j in range(n):
                    color = color_map[face[i*n+j]]
                    cv2.rectangle(img, ((x_off+j)*60, (y_off+i)*60), ((x_off+j+1)*60, (y_off+i+1)*60), color, -1)
                    cv2.rectangle(img, ((x_off+j)*60, (y_off+i)*60), ((x_off+j+1)*60, (y_off+i+1)*60), (0,0,0), 1)

        if heuristic_name is not None:
            # Draw heuristic and move info if provided
            text1 = f"Heuristic: {heuristic_name}"
            text2 = f"Move: {move}" if move else "Initial State"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness = 1
            color = (255, 255, 255)
            margin = 20
            text_x = img_width - max(cv2.getTextSize(text1, font, font_scale, thickness)[0][0], 
                                    cv2.getTextSize(text2, font, font_scale, thickness)[0][0]) - margin
            text1_y = margin + cv2.getTextSize(text1, font, font_scale, thickness)[0][1]
            text2_y = text1_y + cv2.getTextSize(text2, font, font_scale, thickness)[0][1] + 10
            cv2.putText(img, text1, (text_x, text1_y), font, font_scale, color, thickness)
            cv2.putText(img, text2, (text_x, text2_y), font, font_scale, color, thickness)

        # Display the image and handle ESC key to interrupt
        cv2.namedWindow(f"{self.size}x{self.size} Cube", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{self.size}x{self.size} Cube", img)
        cv2.resizeWindow(f"{self.size}x{self.size} Cube", img_width, img_height)
        key = cv2.waitKey(delay)
        if key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Interrupted by ESC")

    def display_search_state(self, cube, move, heuristic_name, depth, heuristic_value, delay=10):
        # Visualizes cube state during A* or any other search
        # Adds more detailed info like depth and heuristic value for step-by-step debugging
        color_map = {
            0: (255, 255, 255),  # White (U)
            1: (0, 0, 255),      # Blue (L)
            2: (0, 255, 0),      # Green (F)
            3: (0, 165, 255),    # Orange (R)
            4: (0, 255, 255),    # Yellow (B)
            5: (255, 0, 0)       # Red (D)
        }
        n = self.size
        face_positions = [
            (n, 0),   # U
            (0, n),   # L
            (n, n),   # F
            (2*n, n), # R
            (3*n, n), # B
            (n, 2*n)  # D
        ]
        img_height = 3 * n * 60
        img_width = 4 * n * 60
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 0
        for face_id, face in enumerate(cube.state):
            x_off, y_off = face_positions[face_id]
            for i in range(n):
                for j in range(n):
                    color = color_map[face[i*n+j]]
                    cv2.rectangle(img, ((x_off+j)*60, (y_off+i)*60), ((x_off+j+1)*60, (y_off+i+1)*60), color, -1)
                    cv2.rectangle(img, ((x_off+j)*60, (y_off+i)*60), ((x_off+j+1)*60, (y_off+i+1)*60), (0,0,0), 1)

        # Annotate heuristic, depth, and current move
        text1 = f"Heuristic: {heuristic_name}"
        text2 = f"Move: {move}" if move else "Initial State"
        text3 = f"Depth: {depth}"
        text4 = f"Heuristic Value: {heuristic_value:.2f}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1
        color = (255, 255, 255)
        margin = 20
        text_x = img_width - max(
            cv2.getTextSize(text1, font, font_scale, thickness)[0][0],
            cv2.getTextSize(text2, font, font_scale, thickness)[0][0],
            cv2.getTextSize(text3, font, font_scale, thickness)[0][0],
            cv2.getTextSize(text4, font, font_scale, thickness)[0][0]
        ) - margin
        text1_y = margin + cv2.getTextSize(text1, font, font_scale, thickness)[0][1]
        text2_y = text1_y + cv2.getTextSize(text2, font, font_scale, thickness)[0][1] + 10
        text3_y = text2_y + cv2.getTextSize(text3, font, font_scale, thickness)[0][1] + 10
        text4_y = text3_y + cv2.getTextSize(text4, font, font_scale, thickness)[0][1] + 10
        cv2.putText(img, text1, (text_x, text1_y), font, font_scale, color, thickness)
        cv2.putText(img, text2, (text_x, text2_y), font, font_scale, color, thickness)
        cv2.putText(img, text3, (text_x, text3_y), font, font_scale, color, thickness)
        cv2.putText(img, text4, (text_x, text4_y), font, font_scale, color, thickness)
        cv2.namedWindow(f"{self.size}x{self.size} Cube", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{self.size}x{self.size} Cube", img)
        cv2.resizeWindow(f"{self.size}x{self.size} Cube", img_width, img_height)
        key = cv2.waitKey(delay)
        if key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Interrupted by ESC")

    def print_cube(self):
        # Print the current state of each face of the cube with labels
        face_names = ['U', 'L', 'F', 'R', 'B', 'D']  # Face order: Up, Left, Front, Right, Back, Down
        n = self.size
        for idx, face in enumerate(self.state):
            print(f"{face_names[idx]}: {np.array(face).reshape((n,n))}\n")

    def run_solver(self):
        try:
            pdb = {}  # Initialize empty Pattern Database dictionary
            comp = comparison()  # Create an object to store heuristic performance comparisons

            self.scramble()  # Scramble the cube before solving
            print("Scrambled cube:")
            self.print_cube()
            time.sleep(0.1)
            self.display(heuristic_name="Initial")  # Display scrambled state with initial tag

            # Create a copy of the scrambled cube for solving using hybrid heuristic
            cube1 = self.copy()
            start1 = time.time()

            solver = rubiks_asearch()  # Create A* solver instance
            #print("here")
            heuristic_fn = solver.custom_hybrid_heuristic  # Get reference to hybrid heuristic function
            #print("here1")
            heuristic_hybrid = heuristic_fn(cube1)  # Evaluate heuristic value before solving
            #print("here2")
            print("\nSolving Cube Using Hybrid Heuristic!")
            # Solve using A* and the hybrid heuristic
            solution1, nodes_hybrid = solver.astar(
                cube1, heuristic_fn, visualizer=self, heuristic_name="Hybrid Heuristic"
            )

            # If a solution was found, apply moves step-by-step and display state
            if solution1:
                for move in solution1:
                    cube1.move(move)
                    print(f"\nAfter Move {move}:\n")
                    cube1.print_cube()
                    cube1.display(heuristic_name="Hybrid Heuristic", move=move)
                print("Solution Found Using Hybrid Heuristic:", solution1)
                print("Cube solved Using Hybrid Heuristic!")
            else:
                print("No solution found.")

            # Record performance metrics for hybrid heuristic
            step_hybrid = len(solution1) if solution1 else 0
            time1 = time.time() - start1
            comp.add_comparison({
                "heuristic_name": "Hybrid Heuristic",
                "time_taken": time1,
                "solution": solution1,
                "steps": step_hybrid,
                "heuristic_value": heuristic_hybrid,
                "nodes_evaluated": nodes_hybrid
            })

            # Create another copy for solving using Pattern Database heuristic
            cube2 = self.copy()
            start2 = time.time()

            heuristic_fn_pdb = solver.pattern_database_heuristic  # Get PDB heuristic function
            heuristic_pdb = heuristic_fn_pdb(cube2, pdb)  # Compute heuristic value using PDB

            print("\nSolving Cube Using Pattern Database Heuristic!")
            # Solve using A* and the pattern database heuristic
            solution2, nodes_pdb = solver.astar(
                cube2, heuristic_fn_pdb, pdb, visualizer=self, heuristic_name="Pattern Db Heuristic"
            )

            # If a solution is found, apply and show each move
            if solution2:
                for move1 in solution2:
                    cube2.move(move1)
                    print(f"\nAfter Move {move1}:\n")
                    cube2.print_cube()
                    cube2.display(heuristic_name="Pattern Db Heuristic", move=move1)
                print("Solution Found Using Pattern Database Heuristic:", solution2)
                print("Cube solved Using Pattern Database Heuristic!")
            else:
                print("No solution found.")

            # Record performance metrics for PDB heuristic
            step_pdb = len(solution2) if solution2 else 0
            time2 = time.time() - start2
            comp.add_comparison({
                "heuristic_name": "Pattern Db Heuristic",
                "time_taken": time2,
                "solution": solution2,
                "steps": step_pdb,
                "heuristic_value": heuristic_pdb,
                "nodes_evaluated": nodes_pdb
            })

            # Show side-by-side performance comparison of both heuristics
            comp.display_comparisons()

        except KeyboardInterrupt:
            # Gracefully handle manual interruption
            print("Interrupted by user.")
        except Exception as e:
            # Catch and print any runtime errors
            print("Error during solving:", e)
        finally:
            # Ensure OpenCV windows are properly closed
            cv2.waitKey(0)
            cv2.destroyAllWindows()
