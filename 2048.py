"""
Detects hand gestures and uses them to control the keyboard to play 2048.
Only works if the user has a game of 2048 open. Example: https://2048game.com.

Valid gestures:
- Nothing
    Open hand (all five fingers visible):
- Arrow keys or confirm action (right for yes, left for no)
    Closed hand (no fingers protruding) moving up, down, left, or right
- Ask to press "R" (new game)
    Loop with protruding fingers (thumb and index fingers together) held for 3 seconds
- Ask to quit
    Loop without protruding fingers (all fingers together) held for 3 seconds
"""
import cv2
import numpy as np
import pyautogui # Control keyboard.
import tracker
from collections import deque
from enum import Enum

class HandInfo:
    def __init__(self, x, y, num_fingers, has_hole):
        self.x = x
        self.y = y
        self.num_fingers = num_fingers
        self.has_hole = has_hole

    @classmethod
    def no_hand(cls):
        return cls(0, 0, 1, False)

    @property
    def is_hand(self):
        return self.x != 0 or self.y != 0

    def __str__(self):
        return f"Hand({self.x}, {self.y}, {self.num_fingers}, {self.has_hole})" if self.is_hand else "Empty"

class HandInfoQueue:
    def __init__(self, length):
        self.queue = deque([HandInfo.no_hand()] * length, length)
        self.num_hands = 0
        self.num_holes = 0
        self.num_fingers = length # Each frame has 1.

    def add(self, new_info):
        oldest = self.queue.popleft()
        self.queue.append(new_info)
        self.num_hands += new_info.is_hand - oldest.is_hand
        self.num_holes += new_info.has_hole - oldest.has_hole
        self.num_fingers += new_info.num_fingers - oldest.num_fingers

    @property
    def avg_fingers(self):
        return self.num_fingers / len(self.queue)

    def is_hand(self):
        return self.num_hands > 0.5 * len(self.queue)

    def has_hole(self):
        return self.num_holes > PERCENT_HOLE_THRESHOLD * len(self.queue)

    def get_movement(self):
        # -1 is most recent, 0 is oldest.
        return self.queue[-1].x - self.queue[0].x, self.queue[-1].y - self.queue[0].y

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str([str(info) for info in self.queue])

class Command(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    ASK_RESET = 4
    ASK_QUIT = 5
    RESET = 6
    QUIT = 7
    NONE = 8

    def is_direction(self):
        return self == Command.LEFT or self == Command.RIGHT or self == Command.UP or self == Command.DOWN

    def is_menu(self):
        return self == Command.ASK_RESET or self == Command.ASK_QUIT

    @property
    def text(self):
        return commands_text[self]

def apply_hand_mask(frame):
    """ Returns a binary image that tries to color the hand as white. """
    skin_mask = tracker.hsv_and_ycrcb_mask(frame)
    frame = tracker.blur_noise(frame, skin_mask)
    return tracker.threshold_binarize(frame, invert=False)[1]

def detect_fingers(binary_frame, annotate=False):
    """ Returns the number of fingers and center of the hand. """
    return tracker.detect_fingers(binary_frame, annotate=annotate)

def detect_hole(binary_frame, annotate=False):
    """ Returns the ellipse center, major and minor axis lengths, and angle. """
    binary_frame = cv2.bitwise_not(binary_frame)
    return tracker.detect_hand_ellipse_hole(binary_frame, annotate=annotate)

def get_hand_info(frame):
    binary_frame = apply_hand_mask(frame)
    num_fingers, hand_area, hand_x, hand_y, _ = detect_fingers(binary_frame)
    if hand_area < MIN_HAND_AREA:
        return HandInfo.no_hand()
    hole_x, hole_y, MA, ma, angle, _ = detect_hole(binary_frame)
    has_hole = MIN_HOLE_AREA <= MA * ma <= MAX_HOLE_AREA and ma / MA <= MAX_HOLE_AXIS_RATIO
    return HandInfo(hand_x, hand_y, num_fingers, has_hole)

def is_arrow(move_queue):
    """
    The movement is an arrow key as long as the move queue detected a hand at least PERCENT_HAND_THRESHOLD
    of the time, detected a hole less than PERCENT_HOLE_THRESHOLD of the time, and the average number of
    fingers is below NUM_FINGERS_ARROW_THRESHOLD.
    """
    return move_queue.is_hand() and not move_queue.has_hole() and move_queue.avg_fingers < NUM_FINGERS_ARROW_THRESHOLD

def get_direction(move_queue):
    """
    To determine the arrow direction, first check whether there is more movement in the x or y direction,
    then make sure it is at least MIN_MOVE_DISTANCE either positive or negative. Otherwise it's NONE.
    """
    dx, dy = move_queue.get_movement()
    if abs(dx) >= abs(dy):
        if dx >= MIN_MOVE_DISTANCE:
            return Command.RIGHT
        if dx <= -MIN_MOVE_DISTANCE:
            return Command.LEFT
    else:
        if dy >= MIN_MOVE_DISTANCE:
            return Command.DOWN
        if dy <= -MIN_MOVE_DISTANCE:
            return Command.UP
    return Command.NONE

def is_hold(hold_queue):
    """
    The movement is a hold as long as the move queue detected a hand at least PERCENT_HAND_THRESHOLD
    of the time, and detected a hole at least PERCENT_HOLE_THRESHOLD of the time.
    """
    return hold_queue.is_hand() and hold_queue.has_hole()

def get_temp_command(move_queue, hold_queue):
    """
    Returns the short-term command, which needs to be then verified against past commands.
    Should only be used if the state is not currently a menu.
    """
    if is_arrow(move_queue):
        return get_direction(move_queue)
    if is_hold(hold_queue):
        return Command.ASK_RESET if hold_queue.avg_fingers >= NUM_FINGERS_HOLD_THRESHOLD else Command.ASK_QUIT
    return Command.NONE

def get_confirmation(move_queue, last_command):
    direction = get_direction(move_queue)
    if direction == Command.LEFT:
        return Command.NONE
    if direction == Command.RIGHT:
        if last_command == Command.ASK_RESET:
            return Command.RESET
        return Command.QUIT
    return last_command # Didn't get confirmation, continue asking.

def get_command(move_queue, hold_queue, last_temp_command):
    if last_temp_command.is_menu(): # If we're on a menu.
        command = get_confirmation(move_queue, last_temp_command)
        return command, command
    temp_command = get_temp_command(move_queue, hold_queue)
    if temp_command.is_direction() and temp_command == last_temp_command:
        return Command.NONE, temp_command # Don't accidentally press the same direction twice.
    return temp_command, temp_command

def execute_command(command):
    """ Returns True when the program should exit. """
    if command == Command.LEFT:
        pyautogui.press("left")
    elif command == Command.RIGHT:
        pyautogui.press("right")
    elif command == Command.UP:
        pyautogui.press("up")
    elif command == Command.DOWN:
        pyautogui.press("down")
    elif command == Command.RESET:
        pyautogui.press("r")
    elif command == Command.QUIT:
        return True
    return False


# Constants, some of which should be configurable by the user.

MIRRORED = False # If mirrored, hand moving towards smaller x values of the image counts as right.
FPS = 16 # Number of frames per second. I'm just hardcoding this for convenience.
NUM_SECONDS_MOVE = 0.5 # Checks whether user moved their hand every half second.
NUM_SECONDS_HOLD = 3 # User needs to hold hand for 3 seconds to trigger menu.

MIN_HAND_AREA = 50_000 # Min size of contour to be counted as the hand.
MIN_HOLE_AREA = 5_000 # Min area of the circumscribing rectangle to be counted as a hole.
MAX_HOLE_AREA = 30_000 # Max area of the circumscribing rectangle to be counted as a hole.
MAX_HOLE_AXIS_RATIO = 3 # Maximum ratio of major axis to minor axis to be counted as a hole.
MIN_MOVE_DISTANCE = 100 # Minimum change in coordinate to be considered a move.
NUM_FINGERS_ARROW_THRESHOLD = 2.5 # Number of fingers to distinguish between open and closed hands (arrows).
NUM_FINGERS_HOLD_THRESHOLD = 1.5 # Number of fingers to distinguish between reset and quit.
PERCENT_HAND_THRESHOLD = 0.5 # Proportion of frames that detect hand to count as being a hand.
PERCENT_HOLE_THRESHOLD = 0.3 # Proportion of frames that detect hole to count as having a hole.

MOVE_QUEUE_FRAMES = round(FPS * NUM_SECONDS_MOVE) # Length of queue for all movement info.
HOLD_QUEUE_FRAMES = round(FPS * NUM_SECONDS_HOLD) # Length of queue for all hold info.
window_name = "2048 Menu"

commands_text = {
    Command.LEFT: "Left",
    Command.RIGHT: "Right",
    Command.UP: "Up",
    Command.DOWN: "Down",
    Command.ASK_RESET: "New game?",
    Command.ASK_QUIT: "Quit?",
    Command.RESET: "Starting new game...",
    Command.QUIT: "Quitting...",
    Command.NONE: "",
}

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)
    move_queue = HandInfoQueue(MOVE_QUEUE_FRAMES)
    hold_queue = HandInfoQueue(HOLD_QUEUE_FRAMES)
    last_command = Command.NONE
    command, temp_command = Command.NONE, Command.NONE

    print("Starting gesture detection")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Could not read from camera")
            break

        if MIRRORED:
            frame = cv2.flip(frame, 1) # Flip horizontally about y axis.
        hand_info = get_hand_info(frame)
        move_queue.add(hand_info)
        hold_queue.add(hand_info)
        command, temp_command = get_command(move_queue, hold_queue, temp_command)
        if command != Command.NONE:
            last_command = command

        tracker.add_text(frame, last_command.text, (450, 50), scale=2)
        if command.is_menu():
            tracker.add_text(frame, "No", (50, 50), scale=2)
            tracker.add_text(frame, "Yes", (1100, 50), scale=2)
        tracker.imshow_smaller(window_name, frame, scale=0.5)
        if execute_command(command):
            break

        k = cv2.waitKey(1) # k is the key pressed.
        if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively.
            print("Received user input for exit")
            break

    print("Exiting")
    cv2.destroyAllWindows()
    cam.release()
