import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('SIFT/images/sokak.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("SIFT/images/sokak2.jpeg", cv2.IMREAD_GRAYSCALE)

#SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#Eslestirme - bf tanımlayıcılar arasındaki eşleşmeleri bulmak için kullanılan bir algoritmadır.
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

#Lowe's ratio testi
"""
Lowe's ratio testi, eşleşmelerin doğruluğunu arttırmak için kullanılan bir tekniktir. 
Burada, her eşleşme için en yakın iki eşleşme (m ve n) arasındaki mesafeye bakılır. 
Eğer ilk eşleşmenin mesafesi, ikinci eşleşmenin mesafesinin %75'inden küçükse, o eşleşme "iyi" olarak kabul edilir. 
Bu, yanlış eşleşmeleri filtrelemeye yardımcı olur.
"""
good_matches = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

#eslesen nokta kordinatları
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
"""
Burada, iyi eşleşen noktalarin koordinatlari alinir:

    src_pts: img1 (ilk görüntü) üzerindeki eşleşen noktalarin koordinatlari.

    dst_pts: img2 (ikinci görüntü) üzerindeki eşleşen noktalarin koordinatlari.

kp1[m.queryIdx].pt ve kp2[m.trainIdx].pt ile anahtar noktalarin koordinatlari elde edilir.
"""

#RANSAC ile homography ve mask bul
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
"""
Homografi, bir görüntüyü başka bir görüntüye dönüştürmek için kullanılan bir matristir. Bu matrisi bulmak için RANSAC (Random Sample Consensus) algoritması kullanılır.

    RANSAC: Eşleşmelerdeki hata paylarını göz ardı ederek, doğru eşleşmeleri bulmaya çalışır. Özellikle görüntülerdeki gürültüyü (outliers) azaltmaya yardımcı olur.
    cv2.findHomography() fonksiyonu, eşleşen noktaların koordinatlarını alır ve bunlara karşılık gelen homografi matrisini hesaplar.
    M: Homografi matrisini içerir.
    mask: Doğru eşleşmeleri (1 ile işaretlenmiş) ve yanlış eşleşmeleri (0 ile işaretlenmiş) gösteren bir maskedir.
    5 değeri : 5.0 burada bir piksel cinsinden tolerans değeridir.
        Yani : 
            Eğer bir eşleşme sonucunda bulundan nokta, gerçek beklenen yerden 5 pikselden daha fazla sapıyorsa, o eşleşme yanlış kabul edilir ve dışlanır(outlier)
            Eğer sapma 5 pikselin altındaysa, doğru eşleşme(inlier) kabul edilir.
        Küçük değerler daha katı davranırken(az eşleşme), büyük değerler daha yumuşak(çok eşleşme) davranır. 3-5 arası idealdir.
"""

matchesMask = mask.ravel().tolist()
"""
    mask : Hangi eşleşmelerin doğru(inlier) olduğunu söyleyen bir liste/matrix. 2D(2 boyutlu) bir matris gibi görünür(shape: (n,1))
        içinde sadece 1 veya0'lar vardır. (1-> eşleşme doğru 2->eşleşme yanliş)
    ravel() : çok boyutlu bir diziyi tek boyutlu(flattened) hale getirir. Yani (n,1) şeklinde duran mask'i tek satırda (n,) haline getiriyor
        Bunu yaıyoruz  çünkü drawMatchesKnn() gibi fonksiyonlar tek boyutlu liste bekliyor.

    mask(önce)   : [[1], [0], [1], [1] , [0]]
    mask.ravel() : [1 0 1 1 0]
    mask.ravel().tolist() : [ 1, 0 , 1, 1, 0]
"""


draw_params = dict(matchColor = (0,255,0), #doğru eşleşmeler yeşil 
                   singlePointColor = None, # Tekli noktalar çizilmesin
                   matchesMask = matchesMask, #sadece doğru eşleşmeleri goster
                   flags=2 # Çizim için bayrak parametresi 
                   # Çizim bayrağıdır. Bu bayrak, eşleşen noktaların birbirine bağlanacağı şekilde çizilmesini sağlar. Bayrak değeri 2, yan yana görüntülerde eşleşen noktaların çizilmesini sağlar.
                    )

#Bu fonksiyon, her iki görüntüyü yan yana getirir ve eşleşen noktaları çizerek gösterir.
img_ransac = cv2.drawMatches(img1, kp1, img2, kp2 , good_matches, None, **draw_params)
# **draw_params: Çizim için parametreler (renk, eşleşme maskesi vb.).



#goster
plt.figure(figsize=(20,10))
plt.imshow(img_ransac)
plt.title("SIFT, RANSAC İle- temiz eşleşmeler")
plt.axis("off")
plt.show()

print(f"Toplam eşleşme : {len(good_matches)}")
print(f"RANSAC sonrası doğru eşleşme : {matchesMask.count(1)}")