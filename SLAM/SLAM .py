"""ardışık 2 görüntü ile SLAM
KITTI dataset kullanilacak
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----Ardışık görüntü seç-----

img1 = cv2.imread("KITTI4.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("KITTI5.png", cv2.IMREAD_GRAYSCALE)


# ------Özellik çikarimi ve eşleştirme-----
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# BF eşleştirici
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Eşleşmeleri mesefeye göre sirala
matches = sorted(matches, key=lambda x: x.distance)

#eşleşen noktalari çikar
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])


# -----Essential Matrix Hesaplama ve Poz Çikarma -----

# KITTI için örnek intrinsic kamera parametreleri(örnek değerler)
K = np.array([[718.8560, 0, 607.1928],
              [0, 718.8560, 185.2157],
              [0, 0, 1]])


#Essential matrix hesabi
E, mask = cv2.findEssentialMat(pts1, pts2, K,method=cv2.RANSAC, prob=0.999, threshold=1.0)
"""
prob: İyi bir model bulma ihtimali. Büyük olursa daha fazla deneyerek daha iyi sonuç bulur.(Daha yavaş olabilir)
threshold : Nokta uyumunda hata toleransıdır. Küçük olursa daha sıkı eşleşme olur(daha az outlier kabul edilir.)
"""

#Kameralar arasindaki hareket
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

print(f"Rotation matrix : \n {R}")
print(f"Translation matrix : \n{t}")

# -----Basit visual odometry çizimi-----

#kameranin baslangic noktasi
trajectory = np.zeros((600, 600,3), dtype=np.uint8)

#kamera hareketini güncelle
x, y = 300, 300 #baslangic noktasi ortada

t_movement = t.flatten()
x += int(t_movement[0]*100)
y += int(t_movement[2]*100) # z ekseni ileri yön

cv2.circle(trajectory, (x,y), 1, (0,255,0), 2)

#çizimi göster
plt.imshow(trajectory)
plt.title("Kamera trajektörisi (Visual Odometry)")
plt.axis("off")
plt.show()
