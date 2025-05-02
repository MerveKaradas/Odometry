"""
Gerçek video akışıyla mini SLAM demosu
İzlenecek yol :
1.Gerçek KITTI odometry görüntülerini kullancağız.
2.Anlik feature tracking + pose recovery + harita güncelleme yapacağiz
3.Video akışı ilerlerdikçe kameranin yolunu ve haritayi çizeceğiz.

1.KITTI dizinden art arda frameleri oku
2.Optical flow ile featurelari takip et
3.Essential matrix ile pozisyon çikar
4.Kameranin trajektörisini çiz ve harita üret
5.Anlik olarak ekranda göster (canli çizim)


Neler Yaptik ? :
-Gerçek zamanli KITTI framelerini okuduk,
-Featurlari Optical FLow ile takip ettik
-Essential Matrix ve Pose ile kamera konumunu güncelledik
-Anlik kamera yolu ve featurlar ekranda çizildi!
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------ Başlangiç Ayarlari -------------------------------

# KITTI veri yolu
path = '/home/mana/Downloads/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/'

# Optical Flow ayarları
lk_params = dict(winSize=(21,21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Kamera iç parametreleri (KITTI calibration)
K = np.array([[718.8560, 0, 607.1928],
              [0, 718.8560, 185.2157],
              [0, 0, 1]])

# Başlangıç pozisyonu
cur_R = np.eye(3)
cur_t = np.zeros((3,1))

# İlk frame
idx = 0
old_img = cv2.imread(path + f'{idx:010}.png', cv2.IMREAD_GRAYSCALE)

# İlk feature'ları bul
old_pts = cv2.goodFeaturesToTrack(old_img,
                                  maxCorners=2000,
                                  qualityLevel=0.01,
                                  minDistance=7,
                                  blockSize=7)

# Trajektori haritası
traj = np.zeros((800, 800, 3), dtype=np.uint8)

# -------- Sürekli Frame İşleme DÖngüsü ----------------

# Kaç frame işlenecek
max_frames = 100  # 100 frame işleyelim (daha fazla da olur)

for idx in range(1, max_frames):
    new_img = cv2.imread(path + f'{idx:010}.png', cv2.IMREAD_GRAYSCALE)

    if new_img is None:
        break

    # Optical Flow ile noktaları takip et
    new_pts, status, error = cv2.calcOpticalFlowPyrLK(old_img, new_img, old_pts, None, **lk_params)

    # Başarılı eşleşmeleri seç
    good_old = old_pts[status == 1]
    good_new = new_pts[status == 1]

    # Essential Matrix ve Pose çıkar
    E, mask = cv2.findEssentialMat(good_new, good_old, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_old, K)

    # Pozisyonu güncelle
    cur_t += cur_R @ t 
    cur_R = R @ cur_R

    # X-Z düzleminde trajektori çiz
    x, y = int(cur_t[0]) + 400, int(cur_t[2]) + 100
    cv2.circle(traj, (x, y), 1, (0,255,0), 2)

    # Güncel görüntü + feature noktalarını göster
    display = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
    for p in good_new:
        cv2.circle(display, tuple(p.astype(int)), 2, (0,255,0), -1)

    cv2.imshow('Current Frame', display)
    cv2.imshow('Trajectory', traj)

    # Çıkış için 'q' tuşu
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Bir sonraki frame için güncelle
    old_img = new_img.copy()
    old_pts = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()

