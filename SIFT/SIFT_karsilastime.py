import cv2
import matplotlib.pyplot as plt

#goruntu yukleme
img1 = cv2.imread("sokak.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("sokak2.jpeg", cv2.IMREAD_GRAYSCALE)

#SIFT olusturma
sift = cv2.SIFT_create()

#anahtar noktaları ve tanımlayıcıları bul 
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#eslestirici olustur (BFMatcher - Brute Force Matcher)
bf = cv2.BFMatcher()

#Descriptorlar arası eslesitrme yap
# her desc için en iyi 2 eşleşmeyi buluyoruz    
matches = bf.knnMatch(des1, des2, k=2)

#Lowe's ratio(oran) testi uygula (güvenilir eslesmeleri secmek icin)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

#eslesmeleri ciz
img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#goster
plt.figure(figsize=(20,10))
plt.imshow(img_matches)
plt.title("SIFT özellik eşleştirme")
plt.axis("off")
plt.show()