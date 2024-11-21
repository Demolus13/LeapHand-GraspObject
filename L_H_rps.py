#!/usr/bin/env python3
import numpy as np
import random
import cv2
import time
import mediapipe as mp

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu

#######################################################
"""This can control and query the LEAP Hand."""
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

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to classify the gesture based on landmarks
def classify_gesture(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].x
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].x
   
    if index_tip > thumb_tip and middle_tip > ring_tip:
        return "paper"
    elif index_tip < thumb_tip and middle_tip < ring_tip:
        return "rock"
    else:
        return "scissors"

# Function to detect hand gesture from video frame
def detect_gesture(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
   
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            gesture = classify_gesture(landmarks)
            return gesture, frame
   
    return None, frame

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

        human_gesture, frame = detect_gesture(frame)
        if human_gesture:
            print(f"Detected Gesture: {human_gesture}")

            # Determine the winner
            winner = determine_winner(pose_name, human_gesture)
            print(winner)

            # Display the result and frame
            cv2.putText(frame, f"LEAP Hand: {pose_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Human: {human_gesture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, winner, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Hand Gesture Detection", frame)

        time.sleep(10)  # Hold each pose for 10 seconds

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()