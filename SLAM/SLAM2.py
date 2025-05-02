"""
KITTI dataset kullanilacak
/home/mana/Downloads/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/

10 frame boyunca kameranin nasil ilerlediğini çizen bir trajektöri oluşturduk

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----KITTI dataset üzerinden resimleri okuma-----

path = '/home/mana/Downloads/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/'
num_frames = 10
images = []

for i in range(num_frames):
    filename = path + f'{i:010}.png'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    images.append(img)


# ------ Kameranın ilk pozisyonunu ve trajektori alanını ayarla-------------


#kameranin baslangic noktasi
trajectory = np.zeros((800, 800,3), dtype=np.uint8)

#kamera koordinatları
cur_R = np.eye(3) # baslangıcta yonu düzgün
cur_t = np.zeros((3,1)) #basalngic merkezde 


# KITTI için örnek intrinsic kamera parametreleri(örnek değerler)
K = np.array([[718.8560, 0, 607.1928],
              [0, 718.8560, 185.2157],
              [0, 0, 1]])


# Frame -frame işleyerek kameranın yönünü çizmek

orb = cv2.ORB_create()

for i in range(num_frames-1):
    img1 = images[i]
    img2 = images[i+1]

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

    #Essential matrix hesabi
    E, mask = cv2.findEssentialMat(pts1, pts2, K,method=cv2.RANSAC, prob=0.999, threshold=1.0)
    """
    prob: İyi bir model bulma ihtimali. Büyük olursa daha fazla deneyerek daha iyi sonuç bulur.(Daha yavaş olabilir)
    threshold : Nokta uyumunda hata toleransıdır. Küçük olursa daha sıkı eşleşme olur(daha az outlier kabul edilir.)
    """
    #Kameralar arasindaki hareket
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    #Pozisyonu güncelle
    cur_t += cur_R @ t
    cur_R += R @ cur_R

    #kameranın X-Z düzlemindeki yerini çiz
    x = int(cur_t[0]) + 400
    y = int(cur_t[2]) + 100

    cv2.circle(trajectory, (x,y), 1, (0,255,0), 2)


#Sonuç görselleştir
plt.figure(figsize=(10,10))
plt.imshow(trajectory)
plt.title("Kamera trajektörisi (Visual Odometry)")
plt.axis("off")
plt.show()










