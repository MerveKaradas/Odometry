"""
Bundle Adjusyment Nedir ? :
    Şu ana kadar 3D noktalar ve kamera pozlarini bulduk, ama biraz hatali olabilir. Çünkü : 
        -Eşleşmeler tam kusursuz değil 
        -Ölçüm hatalari var
        -Triangulation mükemmel değil
    işte Bundle Adjustment, bu hatalari minimize etmek için:
        -Hem 3D noktalari
        -hem de hamera pozisyonlarini beraber optimize eder.
    
        Bunu yaparken bir hedefi vardir : "Tüm reprojection hatalarini en aza indir!"
        Kisaca: Bundle adj., çoklu kameradan veya farkli bakıiş açilarindan alinan görüntülerde, hem kamera 
        pozisyonlarini(konum + yönelim) hem de 3D nokta konumalrini ayni anda optimize etmek için kullanilan bir yöntemdir.
        
        Baska bir deyişle : 
        - Kamerlarin nerede olduğunu daha goğru bulmak.
        - 3D noktalarin gerçek yerlerini daha doğru hesaplamak

        Neden "Bundle" Adjustment deniliyor ? : 
        - "Bundle" terimi, kamera ile sahnedeki 3D noktalar arasinda çizilen ışın demetlerinden (ray bundle) geliyor.
        - "Adjustment" ise bu ışın demetlerini en iyi şekilde hizalamak, yani hatayi minimize etmek anlaminda.

Reprojection Error Nedir?
    Bir 3D noktayi 2D görüntüye geri projekte ediyorsun,
    Gerçek 2D gözlemle(feature noktasıyla) arasindaki mesafeyi ölçüyorsun 

    Rep. error =  Gözlenen 2D Nokta - Projeksiyonda hesaplanan 2D nokta

    -> Kisaca : "3D noktayi çevirince, gerçek noktaya ne kadar yakin düştü?"


"""

import numpy as np
import cv2
from scipy.optimize import least_squares

# ---- Reprojection Error fonk ----

# 3D noktalari kameraya projekte et
def project_points(points_3d, K, R, t):
    points_proj = ( R @ points_3d.T) + t # dünya koordinatlarindaki 3D noktalari kamera koordinat sistemine çevirdik.
    points_proj = points_proj.T # tekrar (Nx3) çevirdik satırlarla çalişmak daha kolay
    points_proj = points_proj / points_proj[:,2:] # Normalize z'ye (iki boyutta z'yi kaybediyoruz z'yi x ve y'ye bölerek scale uygulamiş olduk)
    points_proj = (K @ points_proj.T).T
    """
    K --> Kamera iç parametre matrisi
        Bu matris : 
            -Focal length(Odak uzakliği)
            -Principal point(Görüntü merkez koordinatlari)
            -Skew(çoğunlukla sifir) gibi bilgileri içerir
        Burada normalize edilmiş(x',y') noktalarini piksel koordinatlarina çeviriyoruz.
        
        Normalize edilmiş değerler :
        (x',y' = x/z, y/z)
        
        Not : Bunu yapmazsan, bir "kamera düzlemi" oluşmaz, çünkü gerçek dünyada da derinlik (z) etkisi vardir. 
    """
    
    return points_proj[:, :2] #sadece x,y

def reprojection_error(params, points_3d, points_2d, K):
    """
    Reprojection error hesaplama. Bu fonksiyonun amaci:
    - Tahmin edilen pozisyon ve yönelim(params) kullanarak 3D noktalari 2D'ye projekteetmek,
    - Sonra gözlenen(gerçek) 2D noktalarla karşılaştırıp aradaki hatayi(reprojection error) hesaplamak.

    
    Parametreler:
    - points_3d: 3D noktaların (Nx3) numpy dizisi
    - points_2d: 2D görüntüdeki eşleşen noktaların (Nx2) numpy dizisi
    - K: Kamera iç parametre matrisi (3x3)
    - params : [rvec_x, rvec_y, rvec_z, t_x, t_y, t_z] şeklinde 6 tane parametre içeriyor.
        ilk 3'ü dönüş vektörü (rotation vector)
        sonraki 3'ü taşıma (translation vector)
    """

    rvec = params[:3] # rotation vector
    tvec = params[3:6] # translation vector

    #Rotation vector -> Rotation matrix
    R, _ = cv2.Rodrigues(rvec) # OpenCV Rodrigues fonksiyonu bu vektörü bir 3x3 rotasyon matrisine çeviriyor.
    t = tvec.reshape(3,1) # Translation vektörünü bir (3,1) boyutunda sütun vektörünü çeviriyoruz. Çünkü proje işlemlerinde (3xN) matrislere toplamak için boyut uyumu gerekiyor.
    #Rodrigues dönüşümü kısaca : Küçük bir eksen-çevresinde dönüş vektörünü tam rotasyon matrisine dönüştürür.

    #3D noktalari projekte et
    points_proj = project_points(points_3d, K, R, t) 
    #3D noktalari, tahmini kamera pozisyonuyla projekte ediyoruz. Yani : "Bu tahmine göre 2D'de nerede görünürdü?" diyoruz

    # # points_2d'yi transpoze et (N x 2) şeklinde olması gerektiğinden
    points_2d = points_2d.T  # Şimdi şekli (2, N) olacak

    # Hata : Gözlenen 2D- projekte edilen 2D nokta
    error = (points_proj - points_2d).ravel() 
    # ravel() ile matris tek boyuta indirgeniyor. Böylece optimizasyon algoritamaları (örneğin least squares çözücü kullanilabilir)

    return error


# ---- Temel İŞlemler ---- 

img1 = cv2.imread("images/sokak.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/sokak2.jpeg", cv2.IMREAD_GRAYSCALE)

# ORB oluştur ve kp,des bul
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


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

#goruntu boyutu alma
h, w = img1.shape

#iç parametre matrisi K tanimlama
K = np.array([[1000, 0 , w/2],
             [0 ,1000 , h/2],
             [0, 0, 1]])

# Essential matrix hesabi
E = K.T @ F @ K

print(f"Essential matrix : \n{E}")


# Kamera pozisyonlarini bulma(Rotation R, Translation t)

#sadece mask içindeki doğru eşleşmeleri alıyoruz
pts1_inliers = pts1[mask.ravel() ==1 ]
pts2_inliers = pts2[mask.ravel() ==1 ]

#kamera hareketlerini çikar
retval, R, t, mask_pose =  cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
#recoverPose bizim yerimize R ve t matrixlerini buluyor
# R -> 3x3 dönüşüm(rotation) matrisi
# t -> 3x1 yer değiştirme(translation) matrisi

print("Rotation Matrix R: \n",R)
print("Translation Matrix R: \n",t)

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
pts1_inliers_transpoz = pts1_inliers.T
pts2_inliers_transpoz = pts2_inliers.T


"""
Triangulation, iki görüntüdeki eşleşen noktaların 3D uzaydaki konumlarını hesaplamaya yarar.
Bu, her iki görüntüdeki eşleşen noktaların projeksiyonunu kullanarak, her iki görüntüden de 3D nokta bilgisini türetir.
Kameraların projeksiyon matrisleri (P1 ve P2) kullanılarak 3D noktalar hesaplanır
"""
points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_inliers_transpoz, pts2_inliers_transpoz)

#Homojen koordinatları normalleştir
"""
Triangulation ile elde edilen 3D noktalar homojen koordinatlar şeklindedir ve bunları normalleştirerek 
(yani 4. koordinat olan w'yi çıkararak) 3D uzaydaki noktalar elde edilir. 
"""
points_3d = points_4d_hom/ points_4d_hom[3]
points_3d = points_3d[:3].T



# ---- Optimization Problemi Hazirlik ----

# Başlangiç rotasyon ve translasyon (recoverse sonucu)
rvec, _  = cv2.Rodrigues(R)
rvec = rvec.flatten()
tvec = t.flatten()

# optimizasyon başlangic parametreleri
params_init = np.hstack((rvec, tvec))

# optimize et(reprojection error minimize et)
res = least_squares(reprojection_error, params_init, verbose=2,
                    args=(points_3d, pts1_inliers_transpoz, K))

if res:
    print("Optimiziasyon başarılı")
else:
    print("Optimiziasyon başarısız")

# sonuç parametreleri
rvec_optimized = res.x[:3]
tvec_optimized = res.x[3:6]

# Optimize edilmiş rotasyon matrisi
R_optimized, _ = cv2.Rodrigues(rvec_optimized)
t_optimized = tvec_optimized.reshape(3,1)

# ---- Optimize Edilen Sonçlarla Reprojection ----

#optimize edilmiş noktlarla projeksiyon
points_proj_optimized = project_points(points_3d, K, R_optimized,t_optimized)

#hata hesaplama
reproj_error_before = np.mean(np.linalg.norm(project_points(points_3d,K,R,t) - pts1_inliers, axis=1))
reproj_error_after = np.mean(np.linalg.norm(points_proj_optimized - pts1_inliers,axis=1))

print(f"Önceki Reprojection Error : {reproj_error_before:.4f}")
print(f"Sonraki Reprojection Error : {reproj_error_after:.4f}")

