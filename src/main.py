# src/main.py
import cv2

from yolo import PersonDetector # 사람 탐지 클래스 
from util import classify_crowd #혼잡도 레벨 분류 
from processing import draw_overlay

def main():
    model_path = "src/yolo8n.pt"   
    input_path = "src/test2.jpg"    
    output_path = "src/result.jpg"

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없음: {input_path}")

#person 탐지 모델 생성 
    detector = PersonDetector(model_path=model_path, conf=0.25)
    boxes = detector.detect_people(img)

    n_people = len(boxes) #사람 수 계산 
    level = classify_crowd(n_people)

    out = draw_overlay(img, boxes, n_people, level)
    cv2.imwrite(output_path, out)

    print(f"[OK] saved: {output_path} | people={n_people} | level={level}")

if __name__ == "__main__":
    main()
