import threading
import cv2
from deepface import DeepFace
import os

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load reference image and check if it's valid
reference_img_path = 'Reference_img.jpg'
if not os.path.exists(reference_img_path):
    print(f"Error: Reference image '{reference_img_path}' not found!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

Reference_img = cv2.imread("Reference_img.jpg")

if Reference_img is None:
    print(f"Error: Unable to load '{reference_img_path}'. Check the file format and path.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

def check_face(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, Reference_img.copy())
        face_match = result.get('verified', False)
    except Exception as e:
        print(f"Error in face verification: {e}")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame from webcam.")
        break

    if counter % 30 == 0:  # Check face every 30 frames
        threading.Thread(target=check_face, args=(frame.copy(),), daemon=True).start()
    
    counter += 1

    # Display result
    text = "Face Matched" if face_match else "Face Not Matched"
    color = (0, 255, 0) if face_match else (0, 0, 255)
    cv2.putText(frame, text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Ensure frame is valid before displaying
    if frame is not None:
        cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()