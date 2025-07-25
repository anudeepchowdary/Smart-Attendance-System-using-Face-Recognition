import cv2, pickle, csv, os
import face_recognition
from datetime import datetime

with open("encodings/encodings.pickle", "rb") as f:
    data = pickle.load(f)

attendance_path = "attendance/attendance.csv"
os.makedirs("attendance", exist_ok=True)
if not os.path.exists(attendance_path):
    with open(attendance_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    now = datetime.now()
    date, time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
    with open(attendance_path, "a", newline="") as f:
        csv.writer(f).writerow([name, date, time])

vs = cv2.VideoCapture(0)
recognized_today = set()

while True:
    ret, frame = vs.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.45)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for i, b in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                counts[data["names"][i]] = counts.get(data["names"][i], 0) + 1
            name = max(counts, key=counts.get)
            if name not in recognized_today:
                mark_attendance(name)
                recognized_today.add(name)
        names.append(name)
    for (top, right, bottom, left), name in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Attendance - Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
