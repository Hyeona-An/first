# src/image_processing.py
import cv2

def draw_overlay(img, boxes, n_people: int, level: str):
   
    # 박스
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f"person {conf:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1, cv2.LINE_AA
        )

    # 상단 패널
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    # 텍스트
    cv2.putText(img, f"People: {n_people}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Level: {level}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return img
