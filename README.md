# HW4: Hand Gesture Tracking and Recognition 

For this homework you will be implementing hand gesture recognition and tracking using OpenCV. You will then use this to create custom gestures, and control keyboard and mouse actions. 

There are two basic ways of doing doing hand tracking: the bottom up feature-engineering and heuristic-based approach we'll do in this homework; and a "learning from data" approach. The former has been used extensively during earlier years of AR/VR systems; the latter approach is common today, but doesn't lend itself to this homework assignment. The "learning from data" approach however, can be built on top of the data that we work with these assignment. Machine Learning algorithems may replace, extend or augment the heuristics that you would design in this HW. 

This homework is intended to give you a hands on experience to steps needed to extract reliable input information from real-world video feeds. This includes concepts such as human hand extraction, dealing with contours and hulls, implementing heuristics to design recofnition system for your own gestures, and using those gesture information, to control an action (which in this case is your keyboard and mouse input). 


## Logistics

After you have accepted the assignment, a seperate repo "hw4-gesture-recognition-YourGitID" should have been created. You will push your assignment code to this, and this will be used for grading.

### Deadline

HW4 is due Tuesday 10/30/2020, 11:59PM. Both your code and video need to be turned in for your submission to be complete; HWs which are turned in after 11:59pm will use one of your slip days -- there are no slip minutes or slip hours.

### Academic honesty
Please do not post code to a public GitHub repository, even after the class is finished, since these HWs will be reused both  in the future.

This HW is to be completed as a group containing not more than 3 members. The members should be from your own project group. You are welcome to discuss the various parts of the HWs with your classmates, but you must implement the HWs yourself -- you should never look at anyone else's code.

## Deliverables:

### 1. Video

You will make a 2 minute video showing off your implementation. You should verbally describe, at a very high level, the concepts used to implement the image pose tracking and 3d reconstruction. You must also include captions corresponding to the audio. This will be an important component of all your homework assignments and your final project so it is best you get this set up early. 

### 2. Report

You will need to submit a pdf report. For each part, explaining what you did in that part, with an emphaisis on what you had to do beyond the snippet codes to get that corresponding part, working reliably. 

### 3. Code
You will also need to push your project folder to your Github Classroom assignment's repo.


## Before You Start:
For this homework you will need Python and OpenCV. Refer the previous assignment for setup of these.

### Installing PyAutoGUI

Besides the libraries we used in the previous HW, in this HW we will also be using pyAutoGUI. To install pyAutoGUI, open your conda prompt, and type the following commands
```python
conda activate hw3env
pip install pyautogui
```

## Instructions
Technical Instructions for the assignment can be found at https://docs.google.com/document/d/11zTezw9TxPfq2eKa_JUZjzYNFpwolm3Qtf1syXR_eJk
