"""
SLAM Nedir ? (Simultaneous Localization and Mapping): 
    Bir robotun ya da  kameranin :
    - Nerede olduğunu(localization)
    - Etrafinin haritasini çikarmasini(Mapping) ayni anda yapabilmesidir. Yani hem "ben neredeyim" hem "etraf nasil görünüyor" sorularina cevap verir.

    Çalişma yapisi :
        1. Görüntüden feature'lar çikarilir(SIFT, ORB gibi) 
        2. Bu featurelar takip edilerek hareket tahmini yapilir.
        3. Aynı anda, görülen yerler ile bir harita oluşturulur.
        4. Hareket ve harite birlikte optimize edilir.


SfM Nedir ? (Structure from Motion) : 
    Sadece bir video veya videodan :
    - Kameranin hareketini bulur
    - 3D dünya modelini çikarir.

    SLAM ile benzerdir ama SfM'de :
    -Gerçek zamanda çalişma zorunluluğu yoktur (önceden çekilmiş video veya fotoğraf seti ile yapilir)
    -Daha yoğun ve kaliteli 3D Model çikarilir

Yapilacak adimlar (Mini SfM plani): 
1. İki ardişik görüntü oku
2. Feature çikarimi ve eşleştirme (SIFT veya ORB ile)
3. Fundamental ve Essential Matrix hesapla
4. Kamera pozisyonu bul (R, t)
5. Triangulation ile 3D noktalar üret
6. 3D noktalari çizdir (scatter plot)


BU KOD :
Kod, iki ardışık görüntü kullanarak kameranın hareketini ve sahnedeki 3D nesnelerin konumlarını belirler.
Sonuç olarak elde edilen 3D nokta bulutu, gerçek dünyadaki bir sahnenin üç boyutlu temsilidir.
    
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---- 1. GÖRÜNTÜ OKUMA -----
img1 = cv2.imread("images/sokak.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/sokak2.jpeg", cv2.IMREAD_GRAYSCALE)

# ---- 2. Feature Extraction ve Eşleştirme ----

# orb olusturma
orb = cv2.ORB_create()

#keypoint ve descriptor çikar
kp1,des1 = orb.detectAndCompute(img1, None)
kp2,des2 = orb.detectAndCompute(img2, None) # fonksiyonu ile görüntülerdeki önemli noktalar (keypoints) ve bunlara karşılık gelen öznitelikler (descriptors) elde ediliyor.

#Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #Bu özniteliklerin birbirine ne kadar benzer olduğunu ölçmek için Brute Force Matcher kullanılıyor
matches = bf.match(des1,des2)

#mesafeye göre sirala
matches = sorted(matches, key= lambda x:x.distance)

#iyi eşleşmeleri seç
good_matches = matches[:100] # en iyi 100 eşleşme

#noktalari ayikla
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# ---- 3.Fundamental ve Essential Matrix Hesaplama ----

# fundamental matrix hesabi (RANSAC ile)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# K matrisi(iç parametre matrisi farz ediyoruz)
h,w = img1.shape
K = np.array([[1000, 0 , w/2],
              [0, 1000, h/2],
              [0, 0, 1]])

# Essential matrix (matris çarpiminde matris sirasi önemli)
E = K.T @ F @ K


# ---- 4.Kamera Pozisyonlarini Bulma ----

#mask ile sadece doğru eşleşmeleri al
pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]


#pose (Rotation R ve Translation t ) çikar
retval, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

# ---- 5.Triangulation ile 3D Nokta üretimi ----

#projeksiyon matrisleri
P1 = np.dot(K, np.hstack((np.eye(3),np.zeros((3,1)))))
P2 = np.dot(K, np.hstack((R,t)))

#Triangulation Points
pts1_inliers = pts1_inliers.T
pts2_inliers = pts2_inliers.T


"""
Triangulation, iki görüntüdeki eşleşen noktaların 3D uzaydaki konumlarını hesaplamaya yarar.
Bu, her iki görüntüdeki eşleşen noktaların projeksiyonunu kullanarak, her iki görüntüden de 3D nokta bilgisini türetir.
Kameraların projeksiyon matrisleri (P1 ve P2) kullanılarak 3D noktalar hesaplanır
"""
points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_inliers, pts2_inliers)

#Homojen koordinatları normalleştir
"""
Triangulation ile elde edilen 3D noktalar homojen koordinatlar şeklindedir ve bunları normalleştirerek 
(yani 4. koordinat olan w'yi çıkararak) 3D uzaydaki noktalar elde edilir
"""
points_3d = points_4d_hom/ points_4d_hom[3]
points_3d = points_3d[:3].T


# ---- 6. 3D Nokta Bulutu ----

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], s=5)
ax.set_title('SfM ile Çikarilan 3D Noktalar')
ax.set_xlabel("X ekseni")
ax.set_ylabel("Y ekseni")
ax.set_zlabel("Z ekseni")
plt.show()