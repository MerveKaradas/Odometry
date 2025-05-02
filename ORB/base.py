"""
ORB, İki temel algoritmanın birleşiminden oluşur : 
    FAST : Hizli şekilde köşe(keypoint) bulur.
    BRIEF : Basit ve hızlı sekilde descriptor cikarir.

Sonra ekstra iyilestirmeler yapıp sunlari saglar:
    Rotation Invariance (Donmeye dayanıklı)
    Scale Invariance (Ölcek degisimlerine kismi dayaniklilik)

En önemli avantaji:
    Cok hizlidir.
    Ucretsizdir.(SIFT/SURF eskiden patentliydi, ORB tamamen acik kaynak)

Descriptor tipi binarydir.
SIFT, SURF 'de float 
"""

import cv2
import matplotlib.pyplot as plt

#goruntu yukleme
img1 = cv2.imread("sokak.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("sokak2.jpeg", cv2.IMREAD_GRAYSCALE)

#ORB olusturma
orb = cv2.ORB_create()

#anahtar noktaları ve tanımlayıcıları bul 
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

#eslestirici olustur (BFMatcher - Brute Force Matcher- hamming mesafesi kullanılacak)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#crossCheck=True eslesmelerin karsilikli doğru olmasini saglar

#Descriptorlar arası eslesitrme yap 
matches = bf.match(des1, des2)

#eslesmeleri mesafelerine gore sirala
matches = sorted(matches, key=lambda x:x.distance)

#eslesmeleri cizz
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30],None, flags= cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#goster
plt.figure(figsize=(20,10))
plt.imshow(img_matches)
plt.title("ORB özellik eşleştirme")
plt.axis("off")
plt.show()