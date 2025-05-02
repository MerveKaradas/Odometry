"""
Feature matching yapamyıp, optical flow ile featurları framden frame takip edersek : 
-Çok daha hızlı çalışır
-Daha akıcı ve stabil olur
-SLAM sistemine geçiş çok daha rahat olur!

PLAN :
1. Feature noktalarini (örn : Shi-Tomasi corner) belirle
2. Optical flow ile featuraları takip et(Lucas Kanade Optical flow)  Optical flow (KLT Tracker) ?
3. Her framede yeni kamera hareketini bul
4. Trajektoriyi optical flow ile daha düzgün çiz


Önceki yöntemimizde : 
    ORB feature bul -> eşleştir -> pose hesapla idi
Şİmdi : 
    - Güzel noktalar (corner) seçeceğiz. (Shi-Tomasi gibi),
    - Optical Flow(Lucas- Kanade methodu) ile noktaları frameden frame takip edeceğiz
    - Sonra yine pose çıkarıcaz
    - Daha yumuşak ve stabil bir kamera yolu çizeceğiz

1. İlk framede iyi feature noktaları seç (goodFeaturesToTrack)
2. Optical FLow ile noktaları takip et
3. Essential Matrix ve Pose çikar
4. Kameranin pozisyonunu güncelle
5. Kamera yolunu çizdir

Bu kodlar sonucunda elimizde olanlar : 
-Kamera hareketleri var
-Harita oluşturacak noktalar var (featurelar)
-Optimizasyon altyapisini biliyoruz (Bundle Adjustment)
-Gerçek zamanli görsel takip alt yapimiz var (Optical Flow)

"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----- 1. İlk framede feature noktalarını seçelim -----

path = '/home/mana/Downloads/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/'
num_frames = 10
images = []

for i in range(num_frames):
    filename = path + f'{i:010}.png'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    images.append(img)


#kameranin baslangic noktasi
trajectory = np.zeros((800, 800,3), dtype=np.uint8)

#kamera koordinatları
cur_R = np.eye(3) # baslangıcta yonu düzgün
cur_t = np.zeros((3,1)) #basalngic merkezde 


# KITTI için örnek intrinsic kamera parametreleri(örnek değerler)
K = np.array([[718.8560, 0, 607.1928],
              [0, 718.8560, 185.2157],
              [0, 0, 1]])

#optical flow parametreleri
lk_params = dict(winSize =(21,21),
                 maxLevel=3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
"""
Lucas-Kanade yöntemiyle iki ardışık kare arasındaki noktaları takip etmek (track etmek) için cv2.calcOpticalFlowPyrLK() fonksiyonu kullanılır.

winSize=(21, 21)

    Bu, her noktayı izlemek için kullanılan arama penceresinin (window) boyutudur.
    Yani algoritma, bu büyüklükteki bir alanda hareketi arar.
    Büyük değerler daha kararlı ama daha yavaş olabilir.
    🔹 21x21 piksel'lik bir alan içerisinde noktaların nereye gittiği hesaplanıyor.

maxLevel=3

    Bu, piramid seviyesidir. Görüntü, farklı çözünürlüklerde image pyramid yapısına bölünür.
    3 dediğimizde, 4 seviyeli bir piramit (0, 1, 2, 3) oluşturuluyor.
    Bu sayede, küçükten büyüğe çözünürlüklerde takip yapılabiliyor ve büyük hareketler bile tespit edilebiliyor.
    🔹 Küçük çözünürlükte büyük hareketleri yakalayıp, yüksek çözünürlükte detaylı düzeltme yapılır.

criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    Bu, algoritmanın ne zaman duracağını belirler.
    TERM_CRITERIA_EPS → Belirli bir hassasiyet (0.01) sağlandığında dur.
    TERM_CRITERIA_COUNT → Maksimum 30 iterasyona ulaştığında dur.
    İkisini | (bitwise OR) ile birleştiriyoruz: herhangi biri gerçekleşirse dur.
    🔹 30 iterasyona kadar çalışır, ama eğer hata 0.01 altına düşerse daha erken durur.

"""


#-------- 2.İlk görüntüde feature noktalarını seçelim ------------------

#ilk frame
old_img = images[0]

#Shi-Tomasi corner detection
old_pts = cv2.goodFeaturesToTrack(old_img,
                                  maxCorners=2000, #Algılanacak en fazla köşe sayısı
                                  qualityLevel=0.01, #Bu, iyi feature’ların ne kadar “iyi” olması gerektiğini belirler. "0.01" demek: en iyi köşenin “kalitesinin” %1’i kadar kalitedeki köşeleri de kabul et.Yani 1.0 → sadece en iyi köşeyi, 0.01 → çok daha fazla köşeyi alır.
                                  minDistance=7, #Tespit edilen noktalar arasında olması gereken minimum piksel mesafesi.Bu sayede noktalar birbirine çok yakın olmaz; daha dengeli dağılır.
                                  blockSize=7) #Her köşe noktasının kalitesini hesaplarken kullanılan çevresel pencere boyutu.7x7’lik bir pencere etrafında değerlendirme yapılır.Büyük blockSize, daha yumuşak ama daha kararlı sonuç verir.


# ----- 3. Frame FRame Optical Flow ile Takip ve Pose Hesaplama -------------- 

for i in range(1, num_frames):

    new_img = images[i]

    #optical flow ile noktaları takip et
    new_pts, status, error = cv2.calcOpticalFlowPyrLK(old_img, new_img, old_pts, None, **lk_params)

    #başarılı eşleştirmeleri se.
    good_old = old_pts[status==1]
    good_new = new_pts[status==1]

    #essential matrix
    E, mask = cv2.findEssentialMat(good_new, good_old, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    #pose
    _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_old, K)

    #Pozisyonu güncelle
    cur_t += cur_R @ t
    cur_R += R @ cur_R

    #kameranın X-Z düzlemindeki yerini çiz
    x = int(cur_t[0]) + 400
    y = int(cur_t[2]) + 100

    cv2.circle(trajectory, (x,y), 1, (0,255,0), 2)
    """
    Parametre	    Açiklama
    -------------------------------------------------------------------------------------------------
    trajectory	    Üzerine çember çizilecek görüntü. Genelde boş (siyah) bir imaj.
    (x, y)	        Çemberin merkez koordinatları. Burada kameranın bulunduğu yeni konumu temsil ediyor.
    1	            Çemberin yarıçapı. Yani 1 piksel büyüklüğünde bir nokta.
    (0, 255, 0) 	Çemberin rengi: Bu RGB değil, BGR formatında. Yani bu yeşil renktir.
    2              	Çizgi kalınlığı. 2 piksel kalınlıkta çizilecek.
    """

    # bir sonraki iterasyon için hazırla
    old_img = new_img.copy()
    old_pts = good_new.reshape(-1,1,2)

#Sonuç görselleştir
plt.figure(figsize=(10,10))
plt.imshow(trajectory)
plt.title("Optical Flow ile Kamera trajektörisi (Visual Odometry)")
plt.axis("off")
plt.show()
