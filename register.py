import cv2, os, time

name = input("Enter person name: ").strip()
save_dir = f"dataset/{name}"
os.makedirs(save_dir, exist_ok=True)

cam = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = cam.read()
    if not ret: break
    cv2.imshow("Capture - Press 's' to save, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filepath = os.path.join(save_dir, f"{name}_{count}.jpg")
        cv2.imwrite(filepath, frame)
        print("Saved:", filepath)
        count += 1
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
