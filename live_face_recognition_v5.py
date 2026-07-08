import cv2
import time
import pickle
import csv
import os
from datetime import datetime
from collections import Counter

import pyttsx3
from deepface import DeepFace
from scipy.spatial.distance import cosine

# ==========================
# LOAD FACE DATABASE
# ==========================
with open("models/encodings_v4.pkl", "rb") as file:
    database = pickle.load(file)

# ==========================
# LOAD HAAR CASCADE
# ==========================
face_cascade = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

# ==========================
# START WEBCAM
# ==========================
cap = cv2.VideoCapture(0)

# ==========================
# ATTENDANCE FILE
# ==========================
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])


# ==========================
# VOICE FUNCTION
# ==========================
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except:
        pass


# ==========================
# SAVE ATTENDANCE
# ==========================
def save_attendance(name):

    now = datetime.now()

    date_string = now.strftime("%d-%m-%Y")
    time_string = now.strftime("%H:%M:%S")

    with open(attendance_file, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            name,
            date_string,
            time_string
        ])


# ==========================
# STATES
# ==========================
WAITING = "WAITING"
PROCESSING = "PROCESSING"
SUCCESS = "SUCCESS"
FAILED = "FAILED"

state = WAITING

# ==========================
# VARIABLES
# ==========================
processing_start_time = None
success_start_time = None
failed_start_time = None

predictions = []
final_name = "Unknown"

last_recognition_time = 0

voice_played = False
attendance_saved = False


# ==========================
# MAIN LOOP
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # ---------------- WAITING ----------------
    if state == WAITING:

        cv2.putText(
            frame,
            "SHOW YOUR FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        if len(faces) > 0:

            state = PROCESSING
            processing_start_time = time.time()

            predictions.clear()
            final_name = "Unknown"

            voice_played = False
            attendance_saved = False

    # ---------------- PROCESSING ----------------
    elif state == PROCESSING:

        elapsed = time.time() - processing_start_time

        countdown = max(0, 5 - int(elapsed))

        cv2.putText(
            frame,
            "PROCESSING...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            str(countdown),
            (250, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 255),
            4
        )

        # Recognition every 0.55 sec
        if time.time() - last_recognition_time > 0.55:

            last_recognition_time = time.time()

            for (x, y, w, h) in faces:

                face_crop = frame[y:y + h, x:x + w]

                try:

                    embedding = DeepFace.represent(
                        img_path=face_crop,
                        model_name="Facenet512",
                        enforce_detection=False
                    )[0]["embedding"]

                    best_name = "Unknown"
                    best_distance = 999

                    for item in database:

                        distance = cosine(
                            embedding,
                            item["embedding"]
                        )

                        if distance < best_distance:
                            best_distance = distance
                            best_name = item["name"]

                    if best_distance > 0.60:
                        best_name = "Unknown"

                    predictions.append(best_name)

                    print(
                        f"{len(predictions)} -> {best_name}"
                    )

                except:
                    pass

        # After 9 predictions
        if len(predictions) >= 9:

            final_name = Counter(
                predictions
            ).most_common(1)[0][0]

            if final_name == "Unknown":

                state = FAILED
                failed_start_time = time.time()

            else:

                state = SUCCESS
                success_start_time = time.time()

    # ---------------- SUCCESS ----------------
    elif state == SUCCESS:

        cv2.putText(
            frame,
            "FACE VERIFIED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"WELCOME {final_name.upper()}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if not voice_played:

            speak(f"Welcome {final_name}")

            voice_played = True

        if not attendance_saved:

            save_attendance(final_name)

            attendance_saved = True

        if time.time() - success_start_time > 8:

            state = WAITING

            processing_start_time = None
            success_start_time = None
            failed_start_time = None

            predictions.clear()

            final_name = "Unknown"

            voice_played = False
            attendance_saved = False

    # ---------------- FAILED ----------------
    elif state == FAILED:

        cv2.putText(
            frame,
            "FACE NOT RECOGNIZED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if time.time() - failed_start_time > 2:

            state = WAITING

            processing_start_time = None
            success_start_time = None
            failed_start_time = None

            predictions.clear()

            final_name = "Unknown"

            voice_played = False
            attendance_saved = False

    # Draw rectangles
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Family Face Recognition V5",
        frame
    )

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()