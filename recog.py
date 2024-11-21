import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_finger_closed(finger_tip, wrist):
    # Check if the finger tip is close to the wrist (a simplified logic for gesture recognition)
    return finger_tip.y > wrist.y  # Assuming y increases downward in the image

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

# Main function for video capture and gesture detection
def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
           
            # Flip the frame horizontally for a mirrored view
            frame = cv2.flip(frame, 1)

            # Process the frame for hand landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Detect gestures and draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                   
                    # Detect the gesture
                    gesture = detect_gesture(hand_landmarks.landmark)
                    print(f"Detected Gesture: {gesture}")
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
           
            # Display the frame
            cv2.imshow("Hand Gesture Detection", frame)
           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
