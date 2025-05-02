"""
Essential matrix, kalibre edilmiş(odak uzaklığı, optik merkez bilinen) kameralar için kullanilir.
Yani kameralarda iç parametreler(K) biliniyorsa: 
    E= K.T @ K @ F
@ : matris çarpimi

Eğer kameralarin iç yapisi(K) biliniyorsa, Fundamental Matrixten Essential Matrix bulunabilir.


K = [[ fx 0 cx ],
     [ 0 fy cy ],
     [ 0  0  1 ]]

fx, fy -> odak uzunluklari(pixeller cinsinden) normal biz oluşturunca 1000 veriyorlar
cx, cy -> optik merkez koordinatlari (genellikle görüntü merkezine çok yakindir) - buna da görüntünün tam ortasini veriyorlar

Essential matrix ile neler yapilir? : 
    E matrixten iki kamera arasindaki Rotation(R) ve Translation(t) matrislerini çikarabiliriz.
Yani : Kameralarin nasıl dondugunu ve birbirine göre nasıl kaydiğini bulabiliyoruz!

Triangulation Nedir?
- İki farklı kamera/görüntüden aynı noktaya bakıyorsun.
- Bu iki bakiş açisindan çikan işinlar 3D uzayda nerede kesişirse, o noktanin gerçek 3D pozisyonu bulunur.

Yani iki görüntüde aynı nesnenin nerede olduğunu biliyorsak -> 3D dünyada nerede olduğunu çikarabiliriz.

BU KOD :
Bu Python kodu, iki farklı görüntüdeki eşleşen noktaları kullanarak temel (Fundamental) ve esas (Essential) matrislerini hesaplar ve 
ardından bu matrisler yardımıyla kameraların konumlarını ve 3D uzaydaki noktaların pozisyonlarını çıkarır.
Bu kod, iki kameranın görüntülerinden 3D dünyadaki nokta koordinatlarını bulmaya yönelik bir süreçte, temel ve esas matrisleri kullanarak çok önemli adımları gerçekleştiren bir örnektir.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


img1 = cv2.imread("images/sokak.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/sokak2.jpeg", cv2.IMREAD_GRAYSCALE)

#  Özellik Çıkartma (ORB)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


#  Özellik Eşleşmesi
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# en iyi eşleşmeleri sırala
matches = sorted(matches, key=lambda x: x.distance)  

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
# Temel matris (F) ile iç parametre matrisi (K) kullanılarak esansiyel matris (E) hesaplanır. 
# Esansiyel matris, iki kameranın ortak uzaydaki pozisyon ilişkilerini tanımlar.
E = K.T @ K @ F

print(f"Essential matrix : \n{E}")


#-------- Kamera pozisyonlarini bulma(Rotation R, Translation t) ---------------

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


# 1. kamera projeksiyon matrisi (ilk kamera orijinde)
P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3,1)))))


#2. kamera projeksiyon matrisi (ikinci kamera R ve t ile)
P2 = np.dot(K, np.hstack((R,t)))
"""
R-> dönüş
t -> yerdeğiştirme

np.eye(3) -> 3x3 birim matris (ilk kameranin dönüşü yok)
np.zeros((3,1)) -> ilk kameranin translasyonu yok (orijinde)
 
np.dot(A, B) ifadesi, A ile B'nin matris çarpimini yapiyor. Yani lineer cebirdeki klasik matris çarpimi.

np.hstack((np.eye(3), np.zeros((3,1)))) kısmı mesela bir 3x4 matris oluşturuyor : [I: O], yani 3x3 
birim matrisin yanina bir 3x1 sifir vektörü eklenmiş hali.
K: iç parametre matrisin, 3x3 boyutunda

np.dot(K, [I | O]) yaptiginda, K matrisi ile [I|O] matrisini normal matris çarpim kurallariyla çarpiyoruz.

Özetle : np.dot(A, B): A ve B matrislerinin standart matris çarpimidir. ( satir x sütün)
Burada yapilan şey: kamer iç parametreleri ile diş parametreleri birleştirip kamera projeksiyon matrisini oluşturmak
"""

# 2D noktalari (inliers) uygun formata getiriyoruz
pts1_inliers_t =  pts1_inliers.T
pts2_inliers_t =  pts2_inliers.T

# 3D noktalari hesapla
points_4d_hom = cv2.triangulatePoints( P1, P2, pts1_inliers_t, pts2_inliers_t)

#Homojen koordinatları normal 3D'ye çevir (x, y, z)
points_3d = points_4d_hom / points_4d_hom[3]
points_3d = points_3d[:3].T # 3xN degil Nx3 yapiyoruz

print("Bulunan 3D Noktalar: \n",points_3d)


# Noktalarin uzaydaki dagilimlari 

#3D görselleştirme
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], s=5)
ax.set_title("3D Noktlar")
ax.set_xlabel("X ekseni")
ax.set_ylabel("Y ekseni")
ax.set_zlabel("Z ekseni")

plt.show()
