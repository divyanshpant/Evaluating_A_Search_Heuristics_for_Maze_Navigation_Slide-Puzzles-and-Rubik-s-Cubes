import numpy as np
import heapq

# Implements A* search algorithm for solving mazes using Manhattan or custom dead-end avoidance heuristic
class MazeAStar:
    def __init__(self, maze, visualizer=None):
        self.maze = maze
        self.visualizer = visualizer

    # Standard Manhattan distance used as heuristic for grid-based pathfinding
    def manhattan(self, cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2
        return abs(x1 - x2) + abs(y1 - y2)

    # Custom heuristic to penalize paths that approach dead ends
    def dead_end_avoidance_heuristic(self, cell, goal):
        x, y = cell
        dead_ends = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                # Check if neighboring cell is open
                if (0 <= nx < self.maze.shape[0] and 0 <= ny < self.maze.shape[1] and 
                    self.maze[nx, ny] == 1):
                    neighbors = 0
                    # Count accessible neighbors of that neighbor
                    for ddx, ddy in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                        nnx, nny = nx + ddx, ny + ddy
                        if (0 <= nnx < self.maze.shape[0] and 0 <= nny < self.maze.shape[1] and 
                            self.maze[nnx, nny] == 1):
                            neighbors += 1
                    if neighbors == 1:
                        dead_ends += 1
        # Slight penalty for every potential dead end detected
        return 0.5 * dead_ends

    # Main A* algorithm for maze solving
    def a_star(self, start, goal, heuristic_name="manhattan"):
        heuristic_fn = self.manhattan if heuristic_name == "manhattan" else self.dead_end_avoidance_heuristic
        open_set = []
        g_score = {start: 0}
        f_score = {start: heuristic_fn(start, goal)}
        heapq.heappush(open_set, (f_score[start], heuristic_fn(start, goal), start))
        came_from = {}
        nodes_evaluated = 0

        while open_set:
            _, _, curr_cell = heapq.heappop(open_set)

            if curr_cell == goal:
                # Reconstruct path from start to goal
                path = {}
                cell = goal
                while cell != start:
                    path[came_from[cell]] = cell
                    cell = came_from[cell]
                return path, g_score[goal], heuristic_fn(goal, goal), nodes_evaluated

            directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # 4-connected grid
            for dx, dy in directions:
                child_cell = (curr_cell[0] + dx, curr_cell[1] + dy)
                if (0 <= child_cell[0] < self.maze.shape[0] and 
                    0 <= child_cell[1] < self.maze.shape[1] and 
                    self.maze[child_cell] == 1):

                    temp_g_score = g_score[curr_cell] + 1
                    nodes_evaluated += 1

                    # Show visualization every 10 steps if available
                    if self.visualizer and nodes_evaluated % 10 == 0:
                        self.visualizer.display_search_state(
                            child_cell,
                            move=None,
                            heuristic_name=heuristic_name,
                            depth=temp_g_score,
                            heuristic_value=heuristic_fn(child_cell, goal),
                            delay=100
                        )

                    # Update scores if a better path is found
                    if child_cell not in g_score or temp_g_score < g_score[child_cell]:
                        g_score[child_cell] = temp_g_score
                        f_score[child_cell] = temp_g_score + heuristic_fn(child_cell, goal)
                        heapq.heappush(open_set, (f_score[child_cell], heuristic_fn(child_cell, goal), child_cell))
                        came_from[child_cell] = curr_cell

        return {}, nodes_evaluated


# Solves N-puzzle problem (like 8-puzzle) using A* search with different heuristics
class PuzzleAStar:
    def __init__(self, size, visualizer=None):
        self.size = size
        self.moves = {'U': -size, 'D': size, 'L': -1, 'R': 1}
        self.visualizer = visualizer

    # Represents a state in the search tree
    class PuzzleState:
        def __init__(self, board, parent, move, depth, cost):
            self.board = board
            self.parent = parent
            self.move = move
            self.depth = depth
            self.cost = cost

        def __lt__(self, other):
            return self.cost < other.cost  # Required for heap comparison

    # Simple heuristic: count of tiles out of place
    def misplaced_tile_heuristic(self, board, goal_state):
        return sum(1 for i in range(len(board)) if board[i] != 0 and board[i] != goal_state[i])

    # A weighted combination heuristic that factors in the blank tile's distance too
    def dynamic_weighted_heuristic(self, board, goal_state):
        misplaced = self.misplaced_tile_heuristic(board, goal_state)
        blank_pos = board.index(0)
        goal_blank_pos = goal_state.index(0)
        blank_distance = abs(blank_pos // self.size - goal_blank_pos // self.size) + abs(blank_pos % self.size - goal_blank_pos % self.size)
        return 1.0 * misplaced + 0.5 * blank_distance

    # Returns a new board after applying a move
    def move_tile(self, board, move, blank_pos):
        new_board = board[:]
        new_blank_pos = blank_pos + self.moves[move]
        new_board[blank_pos], new_board[new_blank_pos] = new_board[new_blank_pos], new_board[blank_pos]
        return new_board

    # A* algorithm for solving the puzzle
    def a_star(self, start_state, goal_state, heuristic_name):
        heuristic_fn = self.misplaced_tile_heuristic if heuristic_name == "Misplaced Tile" else self.dynamic_weighted_heuristic
        open_list = []
        closed_list = set()
        initial_state = self.PuzzleState(start_state, None, None, 0, heuristic_fn(start_state, goal_state))
        heapq.heappush(open_list, initial_state)
        nodes_evaluated = 0

        while open_list:
            current_state = heapq.heappop(open_list)

            if current_state.board == goal_state:
                return current_state, current_state.depth, heuristic_fn(current_state.board, goal_state), nodes_evaluated

            closed_list.add(tuple(current_state.board))
            blank_pos = current_state.board.index(0)

            # Try all valid moves from current blank position
            for move in self.moves:
                new_blank_pos = blank_pos + self.moves[move]

                # Prevent illegal moves that wrap around rows/columns
                if move == 'U' and blank_pos < self.size: continue
                if move == 'D' and blank_pos >= len(start_state) - self.size: continue
                if move == 'L' and blank_pos % self.size == 0: continue
                if move == 'R' and blank_pos % self.size == self.size - 1: continue

                new_board = self.move_tile(current_state.board, move, blank_pos)
                if tuple(new_board) in closed_list:
                    continue

                nodes_evaluated += 1
                heuristic_value = heuristic_fn(new_board, goal_state)
                new_state = self.PuzzleState(new_board, current_state, move, current_state.depth + 1, 
                                             current_state.depth + 1 + heuristic_value)

                # Optionally visualize search state
                if self.visualizer and nodes_evaluated % 100 == 0:
                    self.visualizer.display_search_state(
                        new_state.board,
                        move=new_state.move,
                        heuristic_name=heuristic_name,
                        depth=new_state.depth,
                        heuristic_value=heuristic_value,
                        delay=10
                    )

                heapq.heappush(open_list, new_state)
        return None, nodes_evaluated


# Solves a Rubik's Cube using A* with custom heuristics
class rubiks_asearch:
    # Uses precomputed pattern database heuristic (only center pieces considered for simplicity)
    def pattern_database_heuristic(self, cube, pdb):
        key = str([face[(cube.size*cube.size)//2] for face in cube.state])
        return pdb.get(key, 0)

    # Combines multiple heuristic strategies: misplaced tiles and edge/corner mismatch
    def custom_hybrid_heuristic(self, cube, _=None):
        def misplaced_tile_heuristic(cube, _=None):
            count = 0
            n = cube.size
            for face in cube.state:
                center = face[(n*n)//2]
                count += sum(1 for sticker in face if sticker != center)
            return count

        def edge_corner_pairing_heuristic(cube, _=None):
            s = cube.state
            n = cube.size
            misplaced = 0
            if n == 3:
                edges = [
                    (0, 1, 2, 1), (0, 5, 3, 1), (0, 7, 4, 1), (0, 3, 1, 1),
                    (5, 1, 2, 7), (5, 5, 3, 7), (5, 7, 4, 7), (5, 3, 1, 7),
                    (1, 5, 2, 3), (2, 5, 3, 3), (3, 5, 4, 3), (4, 5, 1, 3)
                ]
                corners = [
                    (0, 0, 1, 0, 2, 0), (0, 2, 2, 2, 3, 0), (0, 8, 3, 2, 4, 0), (0, 6, 4, 2, 1, 2),
                    (5, 6, 1, 8, 4, 8), (5, 0, 1, 6, 2, 6), (5, 2, 2, 8, 3, 6), (5, 8, 3, 8, 4, 6)
                ]
            else:
                return misplaced_tile_heuristic(cube)
            for f1, i1, f2, i2 in edges:
                if s[f1][i1] != s[f1][(n*n)//2] or s[f2][i2] != s[f2][(n*n)//2]:
                    misplaced += 1
            for f1, i1, f2, i2, f3, i3 in corners:
                if s[f1][i1] != s[f1][(n*n)//2] or s[f2][i2] != s[f2][(n*n)//2] or s[f3][i3] != s[f3][(n*n)//2]:
                    misplaced += 1
            return misplaced

        misplaced = edge_corner_pairing_heuristic(cube)
        return misplaced

    # A* search to solve Rubik's Cube with selected heuristic
    def astar(self, start_cube, heuristic_fn, pdb=None, visualizer=None, heuristic_name=None):
        frontier = [(heuristic_fn(start_cube, pdb), 0, start_cube, [])]
        explored = {}
        nodes_evaluated = 0

        if visualizer:
            initial_h = heuristic_fn(start_cube, pdb)
            visualizer.display_search_state(
                start_cube,
                move=None,
                heuristic_name=heuristic_name,
                depth=0,
                heuristic_value=initial_h,
                delay=1000
            )
        while frontier:
            est_total, cost, current, path = heapq.heappop(frontier)
            current_hash = hash(current)
            if current.is_solved():
                return path, nodes_evaluated
            if current_hash in explored and explored[current_hash] <= cost:
                continue
            explored[current_hash] = cost
            
            # Apply each possible move unless it's a redundant inverse
            for move in ['U', "U'", 'D', "D'", 'L', "L'", 'R', "R'", 'F', "F'", 'B', "B'"]:
                if path and move[0] == path[-1][0] and move != path[-1]:
                    continue

                new_cube = current.copy()
                new_cube.move(move)
                new_cost = cost + 1
                h = heuristic_fn(new_cube, pdb)
                nodes_evaluated += 1

                if visualizer and nodes_evaluated % 100 == 0:
                    visualizer.display_search_state(
                        new_cube,
                        move=move,
                        heuristic_name=heuristic_name,
                        depth=new_cost,
                        heuristic_value=h,
                        delay=10
                    )

                heapq.heappush(frontier, (new_cost + h, new_cost, new_cube, path + [move]))

        return None, nodes_evaluated
