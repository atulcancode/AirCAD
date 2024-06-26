import cv2
import mediapipe as mp

# Initialize hand detection model
mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Initialize the drawing utility

# Open webcam connection
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Allow some time for the webcam to initialize
cv2.waitKey(1000)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # RGB conversion (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand detection
    results = mp_hands.process(frame_rgb)

    # Draw detected landmarks on the frame (visualization enabled)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Show the frame with camera input and potentially landmarks
    cv2.imshow('Hand Detection', frame)  # Set a descriptive window title

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
