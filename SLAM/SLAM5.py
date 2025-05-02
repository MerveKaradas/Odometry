"""
Loop Closure ve Global Optimization (Giriş)
Şu anda temel sistemi kurduk. Loop closure aşaması için normalde :
-Harita üzerinde tekrar ayni yere gelince,
-"Bu noktalari daha önce gördüm" deyip,
-Sistemi optimize ederiz.


Loop Closure Nedir?
    Loop closure şu anlama gelir:
    -Bir kamera uzun bir süre ilerler,
    -Sonra başlangiç noktasina veya daha önce ziyaret ettiği bir yere döner,
    -Eğer bu algilanirsa ("Ben buraya daha önce gelmiştim")
    -Sistem geçmişteki hatalari düzeltmek için haritayi optimize eder
Yani : Geçmişteki pozisyonlarimi ve harita noktalarini daha doğru hale getirebilirim.

Global optimization Nedir?
    Loop closure yapinca,  
        TÜm kamera pozisyonlaarini,
        Tüm 3D noktalari beraber optimize etmek gerekir.
    Buna Global Optimization(Global Bundle Adjustment) denir.
    Amaç : 
        Kamera rotalarini düzelterek haritayi toparlamak,
        Hatalari azaltip her şeyi birbirine uyumlu hale getirmek.



"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----- Baslangic kod yapisi -----

path = '/home/mana/Downloads/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/'
num_frames = 30
images = []

for i in range(num_frames):
    filename = path + f'{i:010}.png'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    images.append(img)

# KITTI için örnek intrinsic kamera parametreleri(örnek değerler)
K = np.array([[718.8560, 0, 607.1928],
              [0, 718.8560, 185.2157],
              [0, 0, 1]])

#kamera koordinatları
cur_R = np.eye(3) # baslangıcta yonu düzgün
cur_t = np.zeros((3,1)) #basalngic merkezde 

#kameranin baslangic noktasi
trajectory = np.zeros((800, 800,3), dtype=np.uint8)


#optical flow parametreleri
lk_params = dict(winSize =(21,21),
                 maxLevel=3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#ilk keyframe
old_img = images[0]

#Shi-Tomasi corner detection
old_pts = cv2.goodFeaturesToTrack(old_img,
                                  maxCorners=2000,
                                  qualityLevel=0.01,
                                  minDistance=7,
                                  blockSize=7)

keyframes = [0] #keyframe indexlerini tutuyoruz
all_map_points = [] # 3D harita noktalarini burada biriktireceğiz


# ----- Frame-Frame İşleyip Keyframe Seçelim -------------- 

for i in range(1, num_frames):

    new_img = images[i]

    #optical flow ile noktaları takip et
    new_pts, status, error = cv2.calcOpticalFlowPyrLK(old_img, new_img, old_pts, None, **lk_params)

    #başarılı eşleştirmeleri se.
    good_old = old_pts[status==1]
    good_new = new_pts[status==1]

    #essential matrix
    E, mask = cv2.findEssentialMat(good_new, good_old, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    #pose hesabı
    _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_old, K)

    # Kameranın global pozisyonunu güncelle
    cur_t += cur_R @ t
    cur_R += R @ cur_R

    #kameranın X-Z düzlemindeki yerini çiz
    x = int(cur_t[0]) + 400
    y = int(cur_t[2]) + 100

    cv2.circle(trajectory, (x,y), 1, (0,255,0), 2)


    #Eğer keyframe zamanı geldiyse
    if i % 5 == 0:
        keyframes.append(i)

        #iki keyframe arasında triangulation yapıp yeni 3D map points ekle
        P0 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
        P1 = K @ np.hstack((R,t))

        pts1 = good_old.reshape(-1,1,2)
        pts2 = good_new.reshape(-1,1,2)

        #mask ile sadece doğru eşleşmeleri al
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        pts1 = pts1.reshape(-1,2)
        pts2 = pts2.reshape(-1,2)

        pts1 = pts1.T
        pts2 = pts2.T

        points_4d = cv2.triangulatePoints(P0, P1, pts1, pts1)
        points_4d /= points_4d[3]

        all_map_points.append(points_4d[:3].T)

    # bir sonraki iterasyon için hazırla
    old_img = new_img.copy()

    old_pts = good_new.reshape(-1,1,2)


# İki keyframe arasindaki map_points karşilaştirilir

#ilk ve son keyframe'deki noktalardan örnekler
start_points = all_map_points[0]
end_points = all_map_points[-1]


#basitleştir : sadece ilk 100 noktayi karşilaştir
start_points = start_points[:100]
end_points = end_points[:100]


#iki set arasindaki mesafeyi hesapla
diff = np.linalg.norm(start_points - end_points,axis=1)

#ortalama hata 
mean_error = np.mean(diff)

print(f"Baslangic ve bitis arasindaki ortalama hata : {mean_error:.4f}")

"""
Eğer hata küçükse :
    Kamera gerçekten bir döngü yaptı ve başlangiç noktasina yaklaşti demektir.
    Yani loop closure başarili olur.
Eğer hata büyükse : 
    Hafif drift olmuş olabilir(normaldir çünkü henüz tam bundle adjustment yapmadık)

"""

