#!usr/bin/python3 

""" 
    Feature detection - görüntüde önemli noktaları bulmak
    Feature description - bu noktalardan "vektör" oluşturur
    Feature matching - iki görüntüdeki vektörleri karşılaştırır
"""

import cv2
import matplotlib.pyplot as plt

#görüntüyü al
img = cv2.imread("SIFT/images/masa.jpeg", cv2.IMREAD_GRAYSCALE)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# SIFT olustur
sift = cv2.SIFT_create()

#anahtar noktaları ve tanımlayıcıları bul
keypoints, descriptors = sift.detectAndCompute(img,None)

#anahtar noktaları ciz
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#göster
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("SIFT Anahtar noktaları")
plt.axis("off")
plt.show()