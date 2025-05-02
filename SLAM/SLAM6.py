"""
PLAN : Mini Pose Graph Optimization (Simlified)
1. Her keyframe için pozisyonlari(poz ve rotasyon) kaydedelim
2. Pozisyonlar arasında edge(kenar) kuralim
3. Basitleştirilmiş bir hata fonksiyonu tanimlayalim(aralarindaki fark)
4. scipy.optimize.least_squares ile tüm pozlari optimize edelim!

Böylece : 
    SLAM haritamizdaki pozisyonlar daha uyumlu hale gelecek,
    Loop closure olduğunda sistem kendi kendini düzeltmiş olacak!



Pose Graph Nedir?
    - Her düğüm (node) -> Bir kamera pozu(konum + yönelim, yani [R|t])
    - Her kenar (edge) -> İki poz arasindaki ilişki/bağlanti (mesafe/dönüşüm)

Şu şekilde :
    [Pose1]---[Pose2]---[Pose3]---[PoseN]

Her poz birbirine bağli
Aralarindaki kenarlar, ölçümlerden geliyor (motion estimation, visual odometry vs.)

Loop closure olduğunda şöyle bir yeni kenar eklenir : 

    [Pose1]---[Pose2]---[Pose3]---[PoseN]
       |_____________________________|

    Yani Pose1 ile PoseN arasinda yeni bir kenar eklenir,
    çünkü biliyoruz ki aslinda ayni yere geldik.

TÜm pozlar ve noktalar birleştirilir.
Artık:
-Sadece ardişik pozlarin bilgisi yok.
-Döngü kapanişi(loop closure) bilgisi de geldi.
-Bir sürü bağlanti ve constraint var.
Bu bilgileri kullanarak, tüm pozlari ve 3D noktalari global uyumlu hale getirmek istiyoruz.

Global Bundle Adjustment yapilir:
Şİmdi sahneye Global Bundle Adjustment çikiyor.
Bundle Adjustment şunu yapar:
- Tüm kameralarin (pozlar) ve 3D noktalarin
- Tüm gözlem (bağlanti) verisine göre
- Hatalari en küçük hale getirecek şekilde optimize edilmesi
Yani :
    Hem kameralarin konumunu, hem de 3D Noktlari küçük küçük düzelterek,
    her gözlemdeki toplam hatayi minimize ederiz.
Fayda :
    Drift(birikmiş hatalar) temizlenir,
    Harita/dünya modeli daha doğru hale gelir.

Trajectory - geçilen yol 

Akiş şu şekilde : 
1. Odometry ile pozlar birikir.
2. Loop closure algilanir.
3. Yeni constraint (kenar) eklenir.
4. Graph optimization başlar:
    Global pozlar ve noktalar düzeltilir.
    Hatalar minimize edilir.
5. Düzgün bir harita + doğru bir yol elde edilir.


AŞAMA                          NE YAPİLİR?                               NEDEN?
------------------------------------------------------------------------------------------------------
Graph kur                      Pozlar ve bağlantilar düğümlenir          Yapiyi oluşturmak için
Constraint ekle                Loop closure baği eklenir                 Global bilgi gelsin diye
Global Bundle Adjustment       Pozlar ve noktalar optimize edilir        Harita düzeltilsin diye
Sonuç                          Harita doğru olur, sistem stabil olur


"""

import cv2
import numpy as np
import matplotlib.pyplot as plt # 2D çizimler
from mpl_toolkits.mplot3d import Axes3D # matplotlib kütüphanesine ait olan 3D çizim aracını yüklüyor.
from scipy.optimize import least_squares # bilimsel optimizasyon amaçlı bir fonksiyonu yüklüyor.
#least_squares(), bir hata fonksiyonunu minimize etmek için kullanılır.


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


#optical flow parametreleri : iki ardışık görüntü arasinda piksel seviyesinde hareketin bulunmasidir.
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
"""
Lukas-Kanade(Optical Flow) Algoritmasi Adimlari : 
1.Görüntüde iyi feature'lar (köşe noktlari gibi) seçilir (Shi-Tomasi corner detector kullanlır)
2.Bu noktalarin çevresinde küçük bir pencere(ör 5x5) alinir.
3.Optical flow denklemi bu pencere için çözülür.
4.Her köşe noktasi için yeni pozisyon tahmin edilir.
Ve sonuç : Her feature noktasinin nereye gittiğini(ve ne kadar kaydiğini) bulmuş olursun
"""

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
    """
    Kodda sadece cur_t[0] ve cur_t[2] kullanılıyor. Yani:

    cur_t[0] → Kamera pozisyonunun X ekseni (sağ-sol)
    cur_t[2] → Kamera pozisyonunun Z ekseni (ilerleme yönü)

    Bu ikili, hareket düzlemini oluşturuyor (genelde araba gibi sistemler için kamera çoğunlukla yere paralel ilerler, 
    yani Y eksenindeki değişim çok azdır veya ihmal edilebilir). Bu nedenle X-Z düzleminde yol çiziliyor.

    Peki neden +400 ve +100?

        Bunlar offset değerleridir. Şu amaçla kullanılırlar:
            Görsel çizim koordinat sisteminde:
                Ekranlarda ya da OpenCV resimlerinde (örneğin trajectory değişkeni), koordinatlar şöyle çalışır:
                    (0, 0) en sol üst köşedir.
                    x → sağa doğru artar
                    y → aşağıya doğru artar

    Ama SLAM ya da VO sisteminde, t[0], t[2] negatif olabilir (yani kamera sola veya geriye gidebilir). Bu da ekranda görüntü dışında 
    kalmalarına neden olur. Bu yüzden:

    Amaç                                                   	Kodda yapılan
    ---------------------------------------------------------------------------------------------------------------------
    Görüntü merkezine yakın bir başlangıç noktası yapmak	+400, +100 eklenmiş
    Sol üst köşeye çarpmasını engellemek                	Pozitif offset verilmiş
    Görselin orta kısmına çizmeye başlamak	                trajectory boyutu (800, 800) olduğu için merkez (400, 400)

    Neden x için +400 ama y için +100 verilmiş? İkisi neden eşit değil?
    Bu farkın sebebi, kameranın Z ekseni boyunca genellikle daha fazla hareket etmesi, yani sistemin ileri yönde (Z yönünde) 
    daha büyük pozitif değerler alması. Eğer +400 gibi büyük bir offset verilirse, kamera kısa sürede görüntünün dışına taşabilir.

    """

    cv2.circle(trajectory, (x,y), 1, (0,255,0), 2)
    """
    trajectory adli görüntü üzerine:

       - (x, y) konumuna
       - yeşil renkte (0, 255, 0)
       - yarıçapı 1 piksel olan
       - 2 piksel kalınlığında çerçeveli bir nokta (daire) çizer.
    """


    #Eğer keyframe zamanı geldiyse
    if i % 5 == 0:
        keyframes.append(i)

        """
        İki farklı görüntü (frame) arasında aynı noktaya karşılık gelen 2D eşleşmeleri kullanarak, 
        o noktanın 3D koordinatını tahmin etmek → buna triangulation (üçgenleme) denir.
        """
        #iki keyframe arasında triangulation yapıp yeni 3D map points ekle
        P0 = K @ np.hstack((np.eye(3), np.zeros((3,1)))) # ilk kameranın pose'u (R=I, t=0)
        P1 = K @ np.hstack((R,t)) # ikinci kameranın pozisyonu

        #İki frame’deki eşleşen 2D noktaları alıyoruz.
        pts1 = good_old.reshape(-1,1,2) 
        pts2 = good_new.reshape(-1,1,2)

        #mask ile sadece doğru eşleşmeleri al
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        pts1 = pts1.reshape(-1,2)
        pts2 = pts2.reshape(-1,2)

        # triangulatePoints fonksiyonu 2xN format ister. Bu yüzden .T yapılıyor.
        pts1 = pts1.T
        pts2 = pts2.T

        points_4d = cv2.triangulatePoints(P0, P1, pts1, pts2)
        points_4d /= points_4d[3] # 4xN matrisin her sütunu [X, Y, Z, W] formatında. 3D'ye çevirmek için her noktayı W’ye bölüyoruz → [X/W, Y/W, Z/W].
        """
        Kavram                    	Anlamı
        ------------------------------------------------------------------------------------------------------
        points_4d                 	Homojen koordinatlarda 4xN matris (X, Y, Z, W)
        W                       	Ölçekleme faktörü, 3D noktanın normalize edilmesi için kullanılır
        points_4d /= points_4d[3]	Tüm noktaları W'ye bölerek gerçek (x, y, z) konumlarına geçiş yapılır
        """

        all_map_points.append(points_4d[:3].T) #  points_4d[:3]:Bu, sadece ilk 3 satırı alır: [X, Y, Z] .T ile (3, N) → (N, 3) olur. Her satır bir 3D nokta olur

    # bir sonraki iterasyon için hazırla
    old_img = new_img.copy()

    old_pts = good_new.reshape(-1,1,2)
    """
    reshape(-1, 1, 2):

    -1 → Nokta sayısı otomatik hesaplansın
    1 → Her nokta tek bir satır olarak tutulacak
    2 → Her nokta 2 boyutlu (x, y)
    """


# ------------ Pozisyonlari ve Edge'leri Kurmak -------------------------
#her keyframe'in pozisyonunu ve rotasyonunu toplayalim : 


#keyframelerin pozisyonlarini topla
poses = [] # Her bir pose : (R,t)

cur_R = np.eye(3)
cur_t = np.zeros((3,1))

orb = cv2.ORB_create()

for i in keyframes:
    img = images[i]

    if i == 0:
        # ilk frame bir şey yapma
        old_img = img
        old_pts = cv2.goodFeaturesToTrack(old_img,
                                          maxCorners=2000, #Algılanacak en fazla köşe sayısı
                                          qualityLevel=0.01,  #Bu, iyi feature’ların ne kadar “iyi” olması gerektiğini belirler. "0.01" demek: en iyi köşenin “kalitesinin” %1’i kadar kalitedeki köşeleri de kabul et.Yani 1.0 → sadece en iyi köşeyi, 0.01 → çok daha fazla köşeyi alır.
                                          minDistance=7, #Tespit edilen noktalar arasında olması gereken minimum piksel mesafesi.Bu sayede noktalar birbirine çok yakın olmaz; daha dengeli dağılır.
                                          blockSize=7) #Her köşe noktasının kalitesini hesaplarken kullanılan çevresel pencere boyutu.7x7’lik bir pencere etrafında değerlendirme yapılır.Büyük blockSize, daha yumuşak ama daha kararlı sonuç verir.

        poses.append((cur_R.copy(),cur_t.copy()))
        continue

    #Optical flow ile noktalari takip et
    new_pts, status, error = cv2.calcOpticalFlowPyrLK(old_img, img, old_pts, None, **lk_params)

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

    poses.append((cur_R.copy(),cur_t.copy()))

    old_img = img.copy()
    old_pts = good_new.reshape(-1,1,2)


    # -------- Basit bir edge kurmak --------------------

    #iki poz arasindaki hareket(relative transformation) buluyoruz:
    edges = []

    for i in range(len(poses) -1):

        #i ve i+1 frame'leri arasinda bir bağlanti(edge) tanimliyoruz
        R1, t1 = poses[i]
        R2, t2 = poses[i+1]
       
        # Relative hareket : ilkten ikinciye
        rel_R = R2 @ R1.T
        rel_t = t2 - rel_R @ t1

        edges.append((i, i+1, rel_R, rel_t))
        #Edge bilgisi : nasil döndü, nasil yer değiştirdi

# ------------ Optimizasyon Fonksiyonu Tanimlamak -----------------

#bütün pozlari optimize edecek bir hata fonksiyonu
def pose_graph_error(params, edges):
    num_nodes = len(poses)
    errors = []

    for(i, j, rel_R, rel_t) in edges:
        # i ve j node'larinin tahmini pozisyonlari

        ti = params[6*i : 6*i + 3]
        ri = params[6*i + 3 : 6*i + 6]
        tj = params[6*j : 6*j + 3]
        rj = params[6*j +3: 6*j + 6]

        Ri, _ = cv2.Rodrigues(ri)
        Rj, _ = cv2.Rodrigues(rj) #3D uzaydaki bir vektörün rotasyonunu hesaplamak için kullanılan bir matematiksel yöntemdir. 


        # beklenen pozisyon farki
        pred_t = Ri.T @ (tj -ti)
        pred_R = Ri.T @ Rj

        #Translation hatasi
        t_error = pred_t - rel_t.flatten()

        # rotation hatasi (küçük açilar için yaklaşik)
        r_error = cv2.Rodrigues(pred_R @ rel_R.T)[0].flatten()

        errors.append(t_error)
        errors.append(r_error)
    return np.hstack(errors)

# ---------- Optimization Çaliştirmak -----------------
#başlangiç parametrelerimizi hazirlayip optimize ediyoruz:

# başlangiç pozisyonlar(rotation vector + translation)
params_init = []

for R,t in poses:
    rvec, _ = cv2.Rodrigues(R)
    params_init.extend(t.flatten())
    params_init.extend(rvec.flatten())

params_init = np.array(params_init)

#optimize et
res = least_squares(pose_graph_error, params_init, verbose=2, args=(edges,))

# --------------- Optmize Edilmiş Pozlari Çikarmak ------------------

optimized_poses = []

for i in range(len(poses)):
    t_ = res.x[6*i : 6*i + 3]
    r_ = res.x[6*i + 3 : 6*i + 6] # Bu, pozisyonun rotasyon vektörüdür.
    R_, _ = cv2.Rodrigues(r_)
    optimized_poses.append((R_, t_.reshape(3,1)))
""""
1. optimized_poses = []
    optimized_poses: Bu liste, optimize edilmiş tüm pozisyonları (her biri bir rotasyon matrisi ve bir translasyon vektörü içerir) saklamak için kullanılacaktır.

2. for i in range(len(poses)):
    Bu döngü, tüm pozları (pozlar genellikle kameranın farklı zaman dilimlerinde aldığı pozisyonları temsil eder) sırasıyla işlemek için kullanılır.
    poses listesi, her bir kameranın pozisyonunu ve yönelimini (rotasını) içeren (R, t) çiftlerinden oluşur.
    len(poses): Pozların sayısını döndürür, yani kaç tane poz var.

3. t_ = res.x[6*i : 6*i + 3]
    res.x: Bu, optimizasyon sonucu elde edilen parametrelerin bulunduğu vektördür. Bu vektör, her bir poz için translasyon vektörü ve rotasyon vektörünü içerir.
    res.x[6*i : 6*i + 3]: Bu, res.x vektöründeki, i. pozisyon için translasyon vektörünü alır.
        6*i ile başlar, çünkü her bir poz için 6 değer vardır: 3 değer translasyon vektörü (x, y, z) ve 3 değer rotasyon vektörü (r_x, r_y, r_z).
        6*i + 3: Bu, rotasyon vektörünün başlama indeksidir, çünkü ilk 3 değer translasyon vektörüdür.
    Bu satırda, t_ değişkeni, i. pozisyonun translasyon vektörünü alır.

4. r_ = res.x[6*i + 3 : 6*i + 6]
    r_: Bu satırda, aynı şekilde i. pozisyonun rotasyon vektörü alınır. Yani res.x vektörünün 4. 5. ve 6. elemanları (indeksler 6*i + 3 ile 6*i + 6 arasındaki değerler) rotasyon vektörüdür.
    Rotasyon vektörü, kameranın dönme hareketini ifade eder ve genellikle üç bileşenden oluşur: (r_x, r_y, r_z).

5. R_, _ = cv2.Rodrigues(r_)
    cv2.Rodrigues(r_): r_ rotasyon vektörünü bir rotasyon matrisine dönüştürür. Bu dönüşüm, Rodrigues rotasyon formülü kullanılarak yapılır.
        r_: Bu, pozisyonun rotasyon vektörüdür.
        R_: Bu, rotasyon matrisidir. Yani, rotasyon vektöründen hesaplanan 3x3'lük bir matris.
        _: Buradaki ikinci dönüşüm sonucu (derivatif veya Jacobian matrisi) kullanılmadığı için _ olarak atandı.

6. optimized_poses.append((R_, t_.reshape(3,1)))
    Bu satırda, optimize edilmiş rotasyon matrisi (R_) ve optimize edilmiş translasyon vektörü (t_) bir tuple olarak optimized_poses listesine ekleniyor.
    t_ vektörü reshape edilmiştir: t_ başlangıçta 1x3 bir vektör olabilir. reshape(3, 1) komutu, bunu 3x1 boyutunda bir vektöre dönüştürür. Bu, kameranın pozisyonunu temsil eder.

Sonuç olarak, her bir poz için optimize edilmiş rotasyon matrisi ve translasyon vektörü elde edilmiş olur ve bunlar optimized_poses listesinde saklanır.
Özet:

Bu kod, optimizasyon sonucu elde edilen rotasyon ve translasyon parametreleriyle optimize edilmiş pozisyonları hesaplar ve her bir poz için rotasyon matrisini ve translasyon vektörünü optimized_poses listesine ekler. Bu, özellikle pose graph optimization (poz grafik optimizasyonu) yöntemleri kullanarak kameranın daha doğru bir trajektoriye oturtulmasını sağlar.
"""




# ------------ SONUÇ : Optimize edilmiş kamera trajektörisi -----------------
traj_opt = np.zeros((800, 800,3), dtype=np.uint8)

for R, t in optimized_poses:
    x = int(t[0]) + 400
    y = int(t[2]) + 100
    cv2.circle(traj_opt, (x,y), 2 , (0,0,255), -1)


#Sonuç görselleştir
plt.figure(figsize=(10,10))
plt.imshow(traj_opt)
plt.title("Pose Graph Optimization Sonrasi Kamera Yolu")
plt.axis("off")
plt.show()


# --------------- Optimize Edilmiş 3D Harita ------------------------
"""
Şu anda : 
    Optimize edilmiş kamera pozlari(optimized_poses) elimizde,
    Her keyframe'de triangulate ettiğimiz 3D noktalar(all_map_points) elimizde

Şimdi bu ikisini bu ikisini birleştirip düzgün bir 3D harita çiziyoruz.
"""

#Tüm 3D harita noktalarini birlestir
map_points = np.vstack(all_map_points)


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

#3D Harita noktalarini çiz
ax.scatter(map_points[:,0], map_points[:,1], map_points[:,2], s=1, label="Harita Noktalari")

#Optimize kamera pozlarini çiz
for R, t in optimized_poses:
    ax.scatter(t[0], t[1], t[2], c="red",s=10, label="Kamera Pozisyonu")


#Etiketler
ax.set_title("Optimize Edilmiş 3D Harita ve Kamera Yolu")
ax.set_xlabel("X Ekseni")
ax.set_ylabel("Y Ekseni")
ax.set_zlabel("Z Ekseni")

#Kamera ikonlari üst üste binmesin diye sadece bir kere label gösterelim
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()