"""
Fundamental Matrix :  İki görüntü arasında epipolar ilişkileri tanımlar. YAni:
    Birinci görüntüdeki bir nokta,
    İkinci görüntüde bir doğru(epipolar line) üzerinde bulunmak zorundadir.
Yani noktanın eşleşmesi doğru bir çizgi üzerinde olmak zorunda. Kameranın kalibrasyon bilgileri bilnmiyor. Ama bilinirse(odak uzaklığı gibi) fundamental matrixten
essential matrixe geçiş yapabiliriz.

Özetle fundamental matrix : iki kamera arasındaki temel ilişki, iç parametre bilinmeden
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    """ img1 üzerinde epipolar çizgileri ve nokta eşleşmelerini çizer"""
    r, c = img1.shape
    # r: Görüntünün yüksekliği (row = satır), c: Görüntünün genişliği (column = sütun)

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r_line[2] / r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0] * c) / r_line[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(np.int32(pt1)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(np.int32(pt2)), 5, color, -1)
    
    return img1_color, img2_color  # return is moved outside the loop

img1 = cv2.imread("ORB/images/sokak.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("ORB/images/sokak2.jpeg", cv2.IMREAD_GRAYSCALE)

# ORB oluştur ve kp,des bul
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# bf, Brute Force Matcher'dır ve Hamming mesafesi kullanarak deskriptörleri eşleştirir. Bu işlem, görüntülerdeki benzer noktaları bulmak için kullanılır.
# crossCheck=True: Bu, sadece iki yönde de eşleşen noktaların kabul edilmesini sağlar (yani, her eşleşme yalnızca hem birinci hem de ikinci görüntüde doğrulanmışsa kabul edilir).
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# en iyi eşleşmeleri sırala
matches = sorted(matches, key=lambda x: x.distance)  # Corrected variable name from 'mathces' to 'matches'

# eslesen noktalari çikar
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# fundamental matrix hesapla
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

print(f"Fundamental Matrix : \n {F}")

# epipolar çizgilerinin hesabı (ikinci görüntüye göre 1. görüntü çizilecek)
# Bu fonksiyon, ikinci görüntüdeki eşleşen noktalar (pts2) kullanılarak, birinci görüntüdeki epipolar çizgileri hesaplar. Fundamental matrix kullanılarak çizgiler belirlenir.
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)

img5, img6 = draw_epipolar_lines(img1, img2, lines1, pts1, pts2)

# Görüntüleri kaydet
cv2.imwrite("ORB/images/epipolar_lines_image1.jpg", img5)
cv2.imwrite("ORB/images/epipolar_lines_image2.jpg", img6)

#görüntüleri göster
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(img5)
plt.title("Epipolar Çizgiler - Görüntü 1")
plt.axis("off")

plt.subplot(122)
plt.imshow(img6)
plt.title("Eşleşen Noktalar - Görüntü 2")
plt.axis("off")
plt.show()
