import math
import keyinput
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX

# 0 For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        imageHeight, imageWidth, _ = image.shape

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        co = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                for point in mp_hands.HandLandmark:
                    if str(point) == "HandLandmark.WRIST":
                        normalizedLandmark = hand_landmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(
                            normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

                        try:
                            co.append(list(pixelCoordinatesLandmark))
                        except:
                            continue

        if len(co) == 2:
            xm, ym = (co[0][0] + co[1][0]) / 2, (co[0][1] + co[1][1]) / 2
            distance = math.sqrt((co[0][0] - co[1][0]) ** 2 + (co[0][1] - co[1][1]) ** 2)

            # Adjust sensitivity based on your preference
            distance_sensitivity = 150

            # Acceleration (move forward)
            if distance < distance_sensitivity:
                print("Accelerate")
                keyinput.release_key('s')
                keyinput.release_key('a')
                keyinput.release_key('d')
                keyinput.press_key('w')

            # Brake (move backward)
            elif distance > distance_sensitivity + 50:  # Adjust sensitivity for braking
                print("Brake")
                keyinput.release_key('w')
                keyinput.release_key('a')
                keyinput.release_key('d')
                keyinput.press_key('s')

            # Steering (left or right)
            else:
                angle = math.atan2(co[1][1] - co[0][1], co[1][0] - co[0][0]) * (180 / math.pi)
                
                # Adjust steering sensitivity
                steering_sensitivity = 1.5
                
                # Calculate steering based on the angle
                if -45 < angle < 45:
                    print("Steer right")
                    keyinput.release_key('s')
                    keyinput.release_key('a')
                    keyinput.press_key('d')
                elif -135 < angle < -45:
                    print("Steer left")
                    keyinput.release_key('s')
                    keyinput.release_key('d')
                    keyinput.press_key('a')
                else:
                    print("Keep straight")
                    keyinput.release_key('a')
                    keyinput.release_key('d')

        elif len(co) == 1:
            print("Keeping back")
            keyinput.release_key('a')
            keyinput.release_key('d')
            keyinput.release_key('w')
            keyinput.press_key('s')

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
