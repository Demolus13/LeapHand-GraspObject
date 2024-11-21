#!/usr/bin/env python3
import numpy as np
import random
import cv2
import time

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu

#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second. Each of position, velocity, and current costs one sample, so you can sample all three at 30 Hz or one at 90 Hz.

# Allegro hand conventions:
# 0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more
# http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Zeros_and_Directions_Setup_Guide I believe the black and white figure (not blue motors) is the zero position, and the + is the correct way around. LEAP Hand in my videos start at zero position and that looks like that figure.

# LEAP hand conventions:
# 180 is flat out for the index, middle, ring fingers, and positive is closing more and more.
"""
########################################################

class LeapNode:
    def __init__(self):
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
           
        self.motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(self.motors, 'COM13', 4000000)
                self.dxl_client.connect()
                
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def read_pos(self):
        return self.dxl_client.read_pos()

    def read_vel(self):
        return self.dxl_client.read_vel()

    def read_cur(self):
        return self.dxl_client.read_cur()

def detect_gesture(frame):
    # Placeholder for gesture detection logic
    return random.choice(['rock', 'paper', 'scissors'])

def determine_winner(leap_gesture, human_gesture):
    if leap_gesture == human_gesture:
        return "It's a tie!"
    elif (leap_gesture == 'rock' and human_gesture == 'scissors') or \
         (leap_gesture == 'paper' and human_gesture == 'rock') or \
         (leap_gesture == 'scissors' and human_gesture == 'paper'):
        return "LEAP Hand wins!"
    else:
        return "Human wins!"

def main(**kwargs):
    leap_hand = LeapNode()

    # Define poses in radians
    poses = {
        "rock": np.array([np.radians(180), np.radians(240), np.radians(261), np.radians(253),
                          np.radians(180), np.radians(236), np.radians(295), np.radians(243),
                          np.radians(180), np.radians(242), np.radians(270), np.radians(255),
                          np.radians(149), np.radians(87), np.radians(268), np.radians(253)]),
        
        "paper": np.array([np.radians(180)] * 16),  # All joints flat for open fingers

        "scissors": np.array([np.radians(180), np.radians(180), np.radians(180),
                              np.radians(180), np.radians(180), np.radians(180),
                              np.radians(180), np.radians(180), np.radians(180),
                              np.radians(242), np.radians(270), np.radians(255),
                              np.radians(149), np.radians(87), np.radians(268),
                              np.radians(253)]),
    }

    cap = cv2.VideoCapture(0)

    while True:
        # Choose a random pose for the LEAP hand
        pose_name = random.choice(list(poses.keys()))
        pose = poses[pose_name]
        leap_hand.set_leap(pose)

        # Read camera frame and detect human gesture
        ret, frame = cap.read()
        if not ret:
            break

        human_gesture = detect_gesture(frame)
        print(f"Detected Gesture: {human_gesture}")

        # Determine the winner
        winner = determine_winner(pose_name, human_gesture)
        print(winner)

        # Display the frame
        cv2.imshow("Hand Gesture Detection", frame)

        time.sleep(10)  # Hold each pose for 5 seconds

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
