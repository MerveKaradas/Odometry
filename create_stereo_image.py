"""YAPAY STEREO GÖRÜNTÜSÜ OLUŞTURMA"""

import cv2
import numpy as np

# Görüntüyü yükleyin
image = cv2.imread('images/left_stereo.jpeg')

# Görüntü boyutları
height, width, _ = image.shape

# Yatay kaydırma miktarı (bu, stereo etkisini ayarlayan parametredir)
shift = 20

# Sol göz için görüntüyü kaydır
left_image = np.zeros_like(image)
left_image[:, :-shift] = image[:, shift:]

# Sağ göz için görüntüyü kaydır
right_image = np.zeros_like(image)
right_image[:, shift:] = image[:, :-shift]

# Sonuçları göster
cv2.imshow('Left Eye Image', left_image)
cv2.imshow('Right Eye Image', right_image)

# Görüntüleri kaydet
cv2.imwrite('images/left_eye_image.jpg', left_image)
cv2.imwrite('images/right_eye_image.jpg', right_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
