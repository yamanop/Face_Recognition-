import face_recognition
import cv2
import csv
import numpy as np
from datetime import datetime

video = cv2.VideoCapture(0)

# known faces
ronaldo_image = face_recognition.load_image_file("faces/ronaldo.jpg")
ronaldo_encoding = face_recognition.face_encodings(ronaldo_image)[0]

known_face_encodings = [ronaldo_encoding]
known_face_names = ["ronaldo"]

students = known_face_names.copy()

face_locations = []
face_encodings = []

now = datetime.now()
current = now.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{current}.csv"
f = open(filename, "w+", newline="")
writer = csv.writer(f)
writer.writerow(["Name", "Time"])

attendance = True
unknown_logged = False 
while attendance:
    _, frame = video.read()
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for encoding in face_encodings:
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        distance = face_recognition.face_distance(known_face_encodings, encoding)
        best_match = np.argmin(distance)

        if matches[best_match]:
            name = known_face_names[best_match]

        if name in students:
            students.remove(name)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([name, current_time])
        elif name == "Unknown" and not unknown_logged:
            unknown_logged= True
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(["Unknown", current_time])


        # Show name on screen
        cv2.putText(frame, name + " present", (10, 100), cv2.FONT_HERSHEY_DUPLEX,
                    2.5, (255, 0, 0), 5, 3)

    
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        attendance = False  # exit loop

video.release()
cv2.destroyAllWindows()
f.close()
