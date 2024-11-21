import cv2
import mediapipe as mp
import numpy as np
import random
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

# Leap Hand control class
def read_pos(self):
        return self.dxl_client.read_pos()

def is_finger_closed(finger_tip, wrist):
    return finger_tip.y > wrist.y  # Assuming y increases downward in the image

# Detect gesture based on hand landmarks
def detect_gesture(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
   
    if (is_finger_closed(thumb_tip, wrist) and
        is_finger_closed(index_tip, wrist) and
        is_finger_closed(middle_tip, wrist)):
        return "rock"
    elif (is_finger_closed(thumb_tip, wrist) and
          not is_finger_closed(index_tip, wrist) and
          not is_finger_closed(middle_tip, wrist)):
        return "scissors"
    else:
        return "paper"

# Determine winner between Leap Hand and human
def determine_winner(leap_gesture, human_gesture):
    if leap_gesture == human_gesture:
        return "It's a tie!"
    elif (leap_gesture == 'rock' and human_gesture == 'scissors') or \
         (leap_gesture == 'paper' and human_gesture == 'rock') or \
         (leap_gesture == 'scissors' and human_gesture == 'paper'):
        return "LEAP Hand wins!"
    else:
        return "Human wins!"

# Main function to detect gestures and control Leap Hand
def main():
    leap_hand = LeapNode()

    # Define poses for Leap Hand (radians)
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
    frame_counter = 0
    frame_limit = 10  # Process every 10th frame to speed up detection

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
           
            # Flip the frame for a mirrored view
            frame = cv2.flip(frame, 1)

            # Process the frame for hand landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Detect human gesture and control Leap Hand
            if results.multi_hand_landmarks and frame_counter % frame_limit == 0:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                   
                    # Detect human gesture
                    human_gesture = detect_gesture(hand_landmarks.landmark)
                    print(f"Detected Gesture: {human_gesture}")
                    cv2.putText(frame, f"Gesture: {human_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Choose a random pose for Leap Hand to play against human
                    leap_gesture = random.choice(list(poses.keys()))
                    leap_hand.set_leap(poses[leap_gesture])
                    print(f"LEAP Hand Gesture: {leap_gesture}")

                    # Determine the winner
                    winner = determine_winner(leap_gesture, human_gesture)
                    print(f"Result: {winner}")

            frame_counter += 1
           
            # Display the frame
            cv2.imshow("Hand Gesture Detection", frame)
            time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()