import cv2
import numpy as np

# 이미지 로드 
image = cv2.imread('194test.jpg')

# HSV 색공간 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색 범위 
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 마스크 생성
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# 원본 이미지에서 빨간색 영역만 필터링
result = cv2.bitwise_and(image, image, mask=mask)

# 6. 결과 출력
cv2.imshow("Original Image", image)
cv2.imshow("Red Filtered", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
