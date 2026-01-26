from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

img_path = "nya.png"
img = cv2.imread(img_path)

results = model(img)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("result.jpg", img)
