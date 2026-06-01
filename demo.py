import cv2
import numpy as np
from termcolor import colored
from problems import Maze, Puzzle, Cube

WINDOW_NAME = "Game Selector"
WIDTH, HEIGHT = 600, 400

# Button areas
BUTTON_AREAS = {
    0: ((80, 90), (520, 130)),    # Maze Solver
    1: ((80, 150), (520, 190)),   # Puzzle Solver
    2: ((80, 210), (520, 250)),   # Rubik's Cube Solver
}

def draw_menu(selected_game, input_text, stage="menu"):
    img = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, "Select a game by clicking:", (50, 50), font, 0.8, (0, 0, 0), 2)

    games = ["Maze Solver", "Puzzle Solver", "Rubik's Cube Solver"]
    for i, game in enumerate(games):
        top_left, bottom_right = BUTTON_AREAS[i]
        color = (0, 128, 255) if selected_game == i else (200, 200, 200)
        cv2.rectangle(img, top_left, bottom_right, color, -1)
        cv2.putText(img, game, (top_left[0] + 10, top_left[1] + 30), font, 0.7, (0, 0, 0), 2)

    if stage == "size_input":
        cv2.putText(img, "Enter size (press Enter to confirm):", (50, 300), font, 0.6, (0, 0, 0), 1)
        cv2.putText(img, input_text, (50, 340), font, 1, (0, 0, 255), 2)

    return img

def get_clicked_button(x, y):
    for idx, ((x1, y1), (x2, y2)) in BUTTON_AREAS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return idx
    return None

def main():
    selected_game = None
    input_text = ""
    stage = "menu"
    confirmed = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_game, stage
        if event == cv2.EVENT_LBUTTONDOWN and stage == "menu":
            btn = get_clicked_button(x, y)
            if btn is not None:
                selected_game = btn
                stage = "size_input"

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        img = draw_menu(selected_game, input_text, stage)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(50)

        if key == 27:  # ESC to quit
            break

        if stage == "size_input":
            if key in [8, 255]:  # Backspace (255 for OpenCV, 8 for standard)
                input_text = input_text[:-1]
            elif key == 13 or key == 10:  # Enter
                confirmed = True
                break
            elif 48 <= key <= 57:  # Digits only
                input_text += chr(key)

    cv2.destroyAllWindows()

    if confirmed:
        launch_game(selected_game, input_text)

def launch_game(selected_game, size_text):
    print(colored(f"You selected: {['Maze Solver', 'Puzzle Solver', 'Rubik\'s Cube Solver'][selected_game]}", "green"))
    try:
        size = int(size_text)
        if selected_game == 0:
            maze = Maze(size)
            maze.solve()
        elif selected_game == 1:
            puzzle = Puzzle(size)
            puzzle.solve()
        elif selected_game == 2:
            cube = Cube(size)
            cube.run_solver()
    except ValueError:
        print(colored("Invalid input! Please enter a valid size number.", "red"))

if __name__ == "__main__":
    main()
