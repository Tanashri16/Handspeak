import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, c = img.shape
            landmark_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            
            # Print all landmark coordinates (for debugging)
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"Landmark {id}: x={cx}, y={cy}")

            gesture_text = ""

            # Step 1: Detect "Hello" gesture → Only index finger up
            if (landmark_list[8][1] < landmark_list[6][1] and
                landmark_list[12][1] > landmark_list[10][1] and
                landmark_list[16][1] > landmark_list[14][1] and
                landmark_list[20][1] > landmark_list[18][1]):
                gesture_text = "Attention"

            # Step 2: Detect "Peace" gesture → Index + Middle finger up
            elif (landmark_list[8][1] < landmark_list[6][1] and
                  landmark_list[12][1] < landmark_list[10][1] and
                  landmark_list[16][1] > landmark_list[14][1] and
                  landmark_list[20][1] > landmark_list[18][1]):
                gesture_text = "Peace"

            # Step 3: Detect "Hi" gesture → All fingers up
            elif (landmark_list[8][1] < landmark_list[6][1] and
                  landmark_list[12][1] < landmark_list[10][1] and
                  landmark_list[16][1] < landmark_list[14][1] and
                  landmark_list[20][1] < landmark_list[18][1]):
                gesture_text = "Hi"

            # Step 4: Detect "Stop" gesture → Fist (no fingers up)
            elif (landmark_list[8][1] > landmark_list[6][1] and
                  landmark_list[12][1] > landmark_list[10][1] and
                  landmark_list[16][1] > landmark_list[14][1] and
                  landmark_list[20][1] > landmark_list[18][1]):
                gesture_text = "Stop"

            # Step 5: Detect "Good" gesture → Thumb Up
            elif (landmark_list[4][1] < landmark_list[3][1] and
                  landmark_list[8][1] > landmark_list[6][1]):
                gesture_text = "Good"

           

            # Step 6: Detect "RockOn" gesture → Index + Pinky up
            elif (landmark_list[8][1] < landmark_list[6][1] and
                  landmark_list[20][1] < landmark_list[18][1] and
                  landmark_list[12][1] > landmark_list[10][1] and
                  landmark_list[16][1] > landmark_list[14][1]):
                gesture_text = "RockOn"

            # Show detected gesture on the image
            if gesture_text != "":
                cv2.putText(img, gesture_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
