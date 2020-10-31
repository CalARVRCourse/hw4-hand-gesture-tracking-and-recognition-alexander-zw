"""
Detects hand gestures and uses them to control the keyboard to play 2048.
Only works if the user has a game of 2048 open. Example: https://2048game.com.

Valid gestures:
- Nothing
    Open hand (all five fingers visible):
- Arrow keys or confirm action (right for yes, left for no)
    Closed hand (no fingers protruding) moving up, down, left, or right
- Ask to press "R" (new game)
    Loop with protruding fingers (thumb and index fingers together) held for 2 seconds
- Ask to quit
    Loop without protruding fingers (all fingers together) held for 2 seconds
"""
import cv2
import numpy as np
import pyautogui
import tracker

def apply_hand_mask(frame):
    """ Returns a binary image that tries to color the hand as white. """
    skin_mask = tracker.hsv_and_ycrcb_mask(frame)
    frame = tracker.blur_noise(frame, skin_mask)
    return tracker.threshold_binarize(frame, invert=False)[1]

def detect_fingers(frame):
    """ Returns the number of fingers and center of the hand. """
    return tracker.detect_fingers(frame)[:3]

def detect_hole(frame):
    """ Returns the ellipse center, major and minor axis lengths, and angle. """
    frame = cv2.bitwise_not(frame)
    return detect_hand_ellipse_hole[:5]


# If mirrored, hand moving towards smaller x values of the image counts as right.
MIRRORED = False

window_name = "2048 Menu"

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)

    print("Starting gesture detection")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Could not read from camera")
            break

        if MIRRORED:
            frame = cv2.flip(frame, 1) # Flip horizontally about y axis.
        binary_frame = apply_hand_mask(frame)

        tracker.imshow_smaller(window_name, frame)

    print("Exiting")
    cv2.destroyAllWindows()
    cam.release()
