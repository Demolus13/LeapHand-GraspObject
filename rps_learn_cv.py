import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def detect_gesture(hand_landmarks):
    if hand_landmarks:
        finger_tip_indices = [8, 12, 16, 20]  # Tip indices for index, middle, ring, pinky
        finger_tips = [hand_landmarks.landmark[i] for i in finger_tip_indices]

        if all(tip.y < hand_landmarks.landmark[0].y for tip in finger_tips):
            return "rock"
        elif all(tip.y > hand_landmarks.landmark[0].y for tip in finger_tips) and \
             hand_landmarks.landmark[8].y < hand_landmarks.landmark[12].y:
            return "paper"
        elif (finger_tips[1].y < finger_tips[2].y and
              finger_tips[0].y < finger_tips[1].y):
            return "scissors"

    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            if gesture:
                cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
