"""
PLAN : Bundle Adjustment(SLAM içi tam optimizasyon)
1. Kameralarin pozisyonlarini ve 3D map noktalarini topla
2. Reprojection Error fonksiyonu yaz
3. scipy.optimize.least_squares ile tüm sistemi optimize et
4. Sonuçları çiz ve hata farklarini göster

Böylece:
Kameralarin yolu(trajectory) daha düzgün olacak,
3D harita noktalari daha doğru yerlerde olacak

Bundle Adjustment nedir?
    Şunlari yapar : 
    - Tüm kameralarin(pozlar) ve 3D noktalarin
    - Tüm gözlem(bağlanti) verisine göre
    - Hatalari en küçük hale getirecek şekilde optimize edilmesi
    Yani kameralarin konumlarini, hem de 3D noktalari küçük küçük düzelterek, 
    her gözlemdeki toplam hatayi minimize eder

    Fayda:
    -Drift(birikmiş hatalar) temizlenir,    
    -Harita/dünya modeli daha doğru hale gelir.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ----- Reprojection Error Fonksiyonu -------- 

def project_points(points_3d, K, R, t):
    """ 3D noktayı 2D'ye projekte et (kamera üzerinden) """
    proj_points = (R @ points_3d.T) + t
    proj_points = proj_points.T
    proj_points = proj_points / proj_points[:, 2:]  # Normalize z
    proj_points = (K @ proj_points.T).T
    return proj_points[:, :2]  # (x, y)

#asıl hata fonksiyonumuz
def bundle_adjustment_error(params, n_cams, n_points, camera_indices, point_indices, points_2d, K):
    """ Tüm kameralar ve tüm 3D noktalar için reprojection error """
    cam_params = params[:n_cams * 6].reshape((n_cams, 6))
    points_3d = params[n_cams * 6:].reshape((n_points, 3))
    
    errors = []

    for i in range(len(points_2d)):
        cam_idx = camera_indices[i]
        point_idx = point_indices[i]

        t = cam_params[cam_idx, :3]
        rvec = cam_params[cam_idx, 3:]

        R, _ = cv2.Rodrigues(rvec)

        projected = project_points(points_3d[point_idx].reshape(1, 3), K, R, t.reshape(3, 1))

        error = (projected.flatten() - points_2d[i])
        errors.append(error)

    return np.array(errors).ravel()

# Başlangıç Ayarları
path = '/home/mana/Downloads/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/'

lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Kamera iç parametreleri (KITTI calibration)
K = np.array([[718.8560, 0, 607.1928],
              [0, 718.8560, 185.2157],
              [0, 0, 1]])

# Başlangıç pozisyonu
cur_R = np.eye(3)
cur_t = np.zeros((3, 1))

# İlk frame
idx = 0
old_img = cv2.imread(path + f'{idx:010}.png', cv2.IMREAD_GRAYSCALE)

if old_img is None:
    print("İlk resim yüklenemedi, dosya yolunu kontrol edin!")
    exit()

# İlk feature'ları bul
old_pts = cv2.goodFeaturesToTrack(old_img,
                                  maxCorners=2000,
                                  qualityLevel=0.01,
                                  minDistance=7,
                                  blockSize=7)

if old_pts is None:
    print("İlk feature'lar bulunamadı!")
    exit()

# Trajektori haritası
traj = np.zeros((800, 800, 3), dtype=np.uint8)

# Sürekli Frame İşleme DÖngüsü
max_frames = 10
poses = []
poses.append((cur_R.copy(), cur_t.copy()))

# Harita noktalarını toplama
all_map_points = [] 

for idx in range(1, max_frames):
    new_img = cv2.imread(path + f'{idx:010}.png', cv2.IMREAD_GRAYSCALE)

    if new_img is None:
        print(f"{idx} numaralı resim yüklenemedi.")
        break

    # Optical Flow ile noktaları takip et
    new_pts, status, error = cv2.calcOpticalFlowPyrLK(old_img, new_img, old_pts, None, **lk_params)

    if new_pts is None or status is None:
        print(f"Optical flow hatası - {idx} numaralı frame.")
        continue

    # Başarılı eşleşmeleri seç
    good_old = old_pts[status == 1]
    good_new = new_pts[status == 1]

    # Essential Matrix ve Pose çıkar
    E, mask = cv2.findEssentialMat(good_new, good_old, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_old, K)

    # Kameranın global pozisyonunu güncelle
    cur_t += cur_R @ t
    cur_R = R @ cur_R

    # Keyframe seçimi ve triangülasyon
    if idx % 5 == 0:
        P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = K @ np.hstack((R, t))

        pts1 = good_old.reshape(-1, 1, 2)
        pts2 = good_new.reshape(-1, 1, 2)

        # Mask ile doğru eşleşmeleri seç
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        pts1 = pts1.reshape(-1, 2)
        pts2 = pts2.reshape(-1, 2)

        pts1 = pts1.T
        pts2 = pts2.T

        points_4d = cv2.triangulatePoints(P0, P1, pts1, pts2)
        points_4d /= points_4d[3]

        if points_4d.shape[1] > 0:  # Eğer harita noktası oluşturulabiliyorsa
            all_map_points.append(points_4d[:3].T)
        else:
            print(f"Triangülasyon başarısız - {idx} numaralı keyframe.")

    # Bir sonraki frame için hazırla
    old_img = new_img.copy()
    old_pts = good_new.reshape(-1, 1, 2)

    poses.append((cur_R.copy(), cur_t.copy()))

# 3D Harita Noktalarını Kontrol Et
if len(all_map_points) == 0:
    print("Harita noktaları oluşturulamadı. Bu sebeple Bundle Adjustment yapılmaz.")
    exit()

map_points = np.vstack(all_map_points)

# ---------- Optimization için input hazırlık ---------------

# Pozlar (R, t)
cam_params = []
for R, t in poses:
    rvec, _ = cv2.Rodrigues(R)
    cam_params.append(np.hstack((t.flatten(), rvec.flatten())))
cam_params = np.array(cam_params)

# 3D noktalar
points_3d = map_points.copy()

# Gözlemler (eşleşmeler) - Şu anda tüm kameralarla ilişkilendiriyoruz
n_cams = cam_params.shape[0]
n_points = points_3d.shape[0]


"""
Bu satır her çalıştırmada rastgele eşleştirme yapıyor. Bu yüzden her çalıştırmada camera_indices dizisi farklı oluyor.

Yani:

    Aynı 3D noktalar,
    Farklı kameralara "gözlem yapıyor gibi" bağlanıyor,
    Dolayısıyla points_2d tekrar farklı şekilde projekte ediliyor,
    Sonuç olarak bundle adjustment’ın çözmesi gereken optimizasyon problemi her seferinde farklı bir senaryo oluyor.
    Bu sebeple sonuçlardaki cost farklı olur
"""
# Basit eşleştirme: Her kamera tüm noktaları görüyor gibi varsayıyoruz (örnek olsun diye)
camera_indices = np.random.choice(n_cams, n_points)
point_indices = np.arange(n_points)

# Gerçek 2D noktalar tahmini (proje edilerek)
points_2d = []
for idx in range(n_points):
    R, t = poses[camera_indices[idx]]
    p2d = project_points(points_3d[idx].reshape(1, 3), K, R, t)
    points_2d.append(p2d.flatten())
points_2d = np.array(points_2d)

# --------- Bundle Adjustment Optimizasyonu ------------------

# Başlangıç parametre vektörü
params_init = np.hstack((cam_params.ravel(), points_3d.ravel()))

# Optimize et
res = least_squares(bundle_adjustment_error, params_init, verbose=2,
                    args=(n_cams, n_points, camera_indices, point_indices, points_2d, K),
                    max_nfev=100)

print("Optimization bitti.")

# -------- Optimize Sonuçları Çıkarmak ----------------

# Optimize edilmiş parametreleri ayrıştır
opt_cam_params = res.x[:n_cams * 6].reshape((n_cams, 6))
opt_points_3d = res.x[n_cams * 6:].reshape((n_points, 3))

# Yeni optimize pozlar
optimized_poses = []
for cam in opt_cam_params:
    t = cam[:3]
    rvec = cam[3:]
    R, _ = cv2.Rodrigues(rvec)
    optimized_poses.append((R, t.reshape(3, 1)))

# 3D Görselleştirme
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Optimize harita noktaları
ax.scatter(opt_points_3d[:, 0], opt_points_3d[:, 1], opt_points_3d[:, 2], s=1, label="Optimize Harita Noktaları")

# Optimize kamera pozları
for R, t in optimized_poses:
    ax.scatter(t[0], t[1], t[2], c='red', s=10)

ax.set_title('Bundle Adjustment Sonrası 3D Harita ve Kamera Yolu')
ax.set_xlabel('X Ekseni')
ax.set_ylabel('Y Ekseni')
ax.set_zlabel('Z Ekseni')
plt.legend()
plt.show()

"""
Çıkan Sonucun Anlamı:

Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   
    0              1         4.8057e-17                                    6.47e-02    
    1             16         4.8057e-17      0.00e+00       0.00e+00       6.47e-02    
`xtol` termination condition is satisfied.
Function evaluations 16, initial cost 4.8057e-17, final cost 4.8057e-17, first-order optimality 6.47e-02.

Detaylı Açıklama:
-------------------------------------------------------------------------------------------------------------------
Alan	        Açıklama
Cost	        BA’nın minimize etmeye çalıştığı reprojection error’un karesi toplamı. Burada 4.8057e-17, yani neredeyse sıfır. Bu, projeksiyon hatasının çok küçük olduğunu gösterir.
Cost reduction	Önceki adıma göre maliyet (error) ne kadar azalmış. 0 görünüyor çünkü zaten başta hata çok azdı.
Step norm	    Parametrelerin ne kadar değiştiğini gösteriyor. 0 ise, optimizasyon artık bir ilerleme yapmıyor demektir.
Optimality	    Gradient’in büyüklüğü. 6.47e-02 küçük bir değer, bu da sistemin bir optimuma çok yakın olduğunu gösterir.

xtol termination	“X-tolerance” sınırına ulaşıldı demektir; yani artık değişiklik yapılacak kadar büyük bir iyileşme yok.
Ne Anlama Geliyor?
"""