"""
    Disparity  : İki görüntüde ayni noktanin yatay kaymasi
    Depth : Disparity kullanarak hesaplanan uzaklik
    Rectification : İki görüntüyü yatay hizalama işlemi
    Baseline : İki kamera arasindaki fiziksel mesafe
    Focal lenght : Kameranin odak uzunluğu 

    Disparity Map : Noktalar ne kadar kaymiş?
    Depth Map : Noktalar kameraya ne kadar uzak?

    Burdaki kod için stereo görüntülere ihtiyacimiz var eğer bulamazsaniz create_stereo_image.py dosyasindan kodlar üzerinden örnek bir stereo görüntüsü oluşturabilirisiniz.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_left = cv2.imread("images/left_eye_image.jpg", cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread("images/right_eye_image.jpg", cv2.IMREAD_GRAYSCALE)


"""
Disparity map üretiminde OpenCV'de iki yöntem var:
    StereoBM_create() -> hizli ama biraz düşük kalite
    StereoSGBM_create() -> daha kaliteli(kullandiğim)
"""

# Stereo SGBM (Semi Global Block Matching) oluştur - eşleştirici
stereo = cv2.StereoSGBM_create(minDisparity=0, #eşleşmeler için min derinlik farkı
                                numDisparities=64, 
                                blockSize=5, # blok boyutu
                                P1=8 * 3 * 5**2,  # İç parametreler ve piksel farkları
                                P2=32 * 3 * 5**2,  # İç parametreler ve piksel farkları
                                disp12MaxDiff=1, # disparit mapde her iki yönswki max fark
                                uniquenessRatio=10,# potansiyel eşleşmenin benzersiz olma orani
                                speckleWindowSize=100,
                                speckleRange=32)

"""
StereoSGBM_create(): Bu fonksiyon, stereo eşleştirme (disparity map) üretmek için kullanılan bir algoritma olan "Semi Global Block Matching" (SGBM) metodunu oluşturur.

    minDisparity: Görüntüdeki eşleşmeler için minimum disparity (derinlik farkı) değerini belirtir. Bu, genellikle 0 olarak belirlenir.
    numDisparities: Derinlik haritasındaki disparity (fark) değerinin aralığını belirtir. Burada 64, yani 64 farklı disparity değeri hesaplanacaktır.
    blockSize: Eşleştirme için kullanılan blok boyutunu belirler. Bu, görüntüdeki her bir küçük alanın ne kadar geniş olacağını belirtir.
    P1 ve P2: Bu parametreler, stereo eşleştirmede pikseller arasındaki farkların cezalandırılmasını kontrol eder. P1 daha düşük (küçük farklara) ve P2 daha yüksek (büyük farklara) cezalar uygular.
    disp12MaxDiff: Disparity haritasında, her iki yön arasındaki maksimum farkı belirtir. Bu, eşleştirmenin tutarsızlıklarını engellemek için kullanılır.
    uniquenessRatio: Eşleştirme işlemi için, potansiyel eşleşmenin benzersiz olma oranını ayarlayan bir parametredir.
    speckleWindowSize: Görüntüdeki "gürültü" noktalarını ortadan kaldırmaya yardımcı olan bir parametre.
    speckleRange: Eşleşmeler arasındaki farklılıkları sınırlayan bir parametre.
"""

# Eşleştirme işlemi - disparity map
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
"""
    stereo.compute(): Sol ve sağ göz görüntüleri arasındaki disparity (fark) haritasını hesaplar. Disparity haritası, her pikselin sağ ve sol göz arasındaki yatay farkı temsil eder.
    astype(np.float32): Hesaplanan disparity haritasını float32 türüne dönüştürür. Bu, hassas hesaplamalar yapabilmek için gereklidir.
    / 16.0: OpenCV, disparity değerlerini genellikle 16 ile çarpar. Bu adım, bu çarpanı tersine çevirmeyi sağlar, yani doğru disparity değerlerine ulaşırız.
"""

# Disparity haritasını görselleştir
plt.figure(figsize=(10, 5))
plt.imshow(disparity, cmap='gray')
plt.colorbar()
plt.title("Disparity Map")
plt.axis("off")



""" Depth(Derinlik) Haritasi Üretimi """

#stereo kameranin parametreleri
focal_lenght = 1000 # odak uzakliği (örnek değer)
baseline = 0.54 # Kamera merkezleri arası mesafe (metre)

# Derinlik hesapla (Z = f* B / disparity)
depth = (focal_lenght * baseline) / (disparity + 1e-6) # 0 bölmeye karşı küçük sayi ekledik
"""
    focal_lenght: Bu, stereo kameranın odak uzaklığıdır. 1000 örnek bir değerdir ve kameranın lens özelliklerine göre değişebilir.
    baseline: Kamera merkezleri arasındaki mesafeyi belirtir. Bu mesafe, kameralar arasındaki fiziksel mesafeyi ifade eder ve genellikle santimetre veya metre cinsinden ölçülür.
    depth: Derinlik (Z) haritasını hesaplamak için kullanılan formül, Z = (f * B) / disparity'dir. Burada:

        f odak uzaklığı,
        B kameralar arasındaki mesafe (baseline),
        disparity ise iki görüntü arasındaki farktır.
        1e-6: disparity sıfır olduğunda bölme hatasını önlemek için eklenen küçük bir değerdir.
"""

# derinlik haritasini görselleştir
plt.figure(figsize=(10,5))
plt.imshow(depth, 'plasma')
plt.title('Derinlik haritasi')
plt.colorbar(label = 'Mesafe (metre)')
plt.axis("off")
plt.show()