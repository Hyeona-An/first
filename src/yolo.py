from ultralytics import YOLO
import cv2

# 1. 모델 로드
model = YOLO("yolov8n.pt")

# 2. 이미지 읽기
img_path = "test2.jpg" 
img = cv2.imread(img_path)

# 3. YOLO 추론
results = model(img)
person_count = 0

# 4. 탐지 결과 처리
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
       
        if cls != 0:
            continue

        person_count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"person {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# 5. 혼잡도 판단
if person_count <= 3:
    level = "LOW"
elif person_count <= 7:
    level = "MID"
else:
    level = "HIGH"

# 6. 상단 정보 패널 (반투명)
overlay = img.copy()
cv2.rectangle(overlay, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

# 7. 텍스트 표시
cv2.putText(
    img,
    f"People: {person_count}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)
cv2.putText(
    img,
    f"Level: {level}",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (255, 255, 255),
    2
)

# 8. 결과 출력
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("result.jpg", img)
print(f"People={person_count}, Level={level}")
