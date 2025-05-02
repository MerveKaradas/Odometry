"""
Her anahat nokta için yön bilgisi atıyoruz
gradients(eğimler) kullanılıyor. Kenarlarının yönüne bakarak bir açı atanıyor.
Neden? Rotasyon(dönme) dayanıklılığı için .Bir nesne dönse bile, ona göre yönlendirme yapabiliyoruz


SOBEL Nedir?
Sobel operatörü, bir görüntüde kenarları bulmak için kullanılan bir matematiksel tekniktir. Özellikle, görüntüdeki parlaklık değişimlerini(yeni gradientı) hesaplar.

Daha basitçe:
    Bir görüntüde keskin bir renk/parlaklık değişimi varsa(mesela siyah-beyaz sınırı), orada kenar vardır
    Sobel, bu değişimi hem x ekseninde(sağ-sol) hem de y ekseninde (yukarı-aşağı) ayrı ayrı ölcer
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("SIFT/images/masa.jpeg", cv2.IMREAD_GRAYSCALE)

# x ve y eksenlerinde kenar bul
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5) #görüntüdeki yatay parlaklık degisimleri -> dikey kenarlar
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5) #görütüdeki dikey parlaklık degisimleri -> yatay kenarlar
"""
    img : islenecek göruntu
    cv2.CV_64F : cikti tipini ayarliyor(daha hassas degerler icin 64-bit float)
    sonraki iki parametre : hangi yönde turev almak istedigimizi belirtiyor
        (1, 0) -> x ekseninde turev alir -> yatay kenarlar bulunur(yani dikey cizgiler gorunur)
        (0, 1) -> y ekseninde turev alir -> dikey kenarlar bulunur(yani yatay cizgiler gorunur)
    ksize=5 : kullanılan cekirdek(kernel) boyutu, 5x5'lik bir pencere kullanılıyor
"""


"""
SIFT Algoritmasında her anahtar noktaya bir yön atamak istiyoruz.
Bu yüzden o noktanın etrafındaki kenarların:
    ne kadar güclü oldugunu(magnitude)
    hangi yone baktigini(orientation) bilmemiz gerekiyor
Bu bilgiler sayesinde nesne döndürülse bile ayni şekilde anahtar noktalari eslestirilebilir.
"""

#Gradient magnitude ve orientation
magnitude = np.sqrt(sobelx**2 + sobely**2)
orientation = np.arctan2(sobely,sobelx) * (180/np.pi)

plt.imshow(orientation, cmap='hsv')
plt.title("Gradient Orientation")
plt.axis("off")
plt.show()