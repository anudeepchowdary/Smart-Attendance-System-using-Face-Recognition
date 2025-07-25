import os, pickle
from imutils import paths
import face_recognition, cv2

imagePaths = list(paths.list_images("dataset"))
knownEncodings, knownNames = [], []

for imagePath in imagePaths:
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    for enc in encodings:
        knownEncodings.append(enc)
        knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}
os.makedirs("encodings", exist_ok=True)
with open("encodings/encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encodings saved to encodings/encodings.pickle")
