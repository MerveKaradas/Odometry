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


"""
import numpy as np
import cv2


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
