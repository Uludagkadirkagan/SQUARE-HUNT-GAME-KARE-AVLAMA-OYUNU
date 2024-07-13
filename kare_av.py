import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Drawing utility
mp_drawing = mp.solutions.drawing_utils
# Hands model
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

thumb_up = False
bullets = []
score = 0
shape_life = 3  # Shape life in seconds

# Define the positions, sizes, and last update times of the shapes
square_pos = [50, 0]
square_size = 100
square_last_update = time.time()
triangle_pos = [100, 50]
triangle_size = 100
triangle_last_update = time.time()

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the square
        cv2.rectangle(image, tuple(square_pos), (square_pos[0] + square_size, square_pos[1] + square_size), (255, 0, 0),
                      -1)

        # Draw the triangle
        triangle_cnt = np.array([triangle_pos, (triangle_pos[0] + triangle_size, triangle_pos[1]),
                                 (triangle_pos[0] + triangle_size // 2, triangle_pos[1] + triangle_size)]).reshape(
            (-1, 1, 2))
        cv2.drawContours(image, [triangle_cnt], 0, (0, 255, 0), -1)

        # Detections
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

                # Define landmarks for the thumb and index finger
                thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand.landmark[mp_hands.HandLandmark.THUMB_IP]
                index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Check if the thumb is raised
                if thumb_tip.y < thumb_ip.y:
                    thumb_up = True
                else:
                    if thumb_up:
                        bullets.append([int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])])
                    thumb_up = False

                # Move bullets and check for collisions
                new_bullets = []
                for bullet in bullets:
                    bullet[0] -= 50
                    if 0 <= bullet[0] < image.shape[1]:
                        new_bullets.append(bullet)
                        cv2.circle(image, tuple(bullet), radius=10, color=(0, 255, 0), thickness=-1)
                        # Check for collision with square
                        if square_pos[0] <= bullet[0] <= square_pos[0] + square_size and square_pos[1] <= bullet[1] <= \
                                square_pos[1] + square_size:
                            score += 10
                            new_bullets.pop()
                            # Move square to new position
                            square_pos = [random.randint(0, image.shape[1] // 4),
                                          random.randint(0, image.shape[0] - square_size)]
                            square_last_update = time.time()
                        # Check for collision with triangle
                        elif triangle_pos[0] <= bullet[0] <= triangle_pos[0] + triangle_size and triangle_pos[1] <= \
                                bullet[1] <= triangle_pos[1] + triangle_size:
                            score -= 10
                            new_bullets.pop()
                            # Move triangle to new position
                            triangle_pos = [random.randint(0, image.shape[1] // 4),
                                            random.randint(0, image.shape[0] - triangle_size)]
                            triangle_last_update = time.time()
                bullets = new_bullets

                # Move shapes if they haven't been hit for a while
                current_time = time.time()
                if current_time - square_last_update >= shape_life:
                    square_pos = [random.randint(0, image.shape[1] // 4),
                                  random.randint(0, image.shape[0] - square_size)]
                    square_last_update = current_time
                if current_time - triangle_last_update >= shape_life:
                    triangle_pos = [random.randint(0, image.shape[1] // 4),
                                    random.randint(0, image.shape[0] - triangle_size)]
                    triangle_last_update = current_time

        # Show the score
        cv2.putText(image, 'Skor: ' + str(score), (image.shape[1] // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2, cv2.LINE_AA)

        # Show the image
        cv2.imshow('Kare Avlama', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()