import cv2
import matplotlib.pyplot as plt

img = cv2.imread("SIFT/images/masa.jpeg", cv2.IMREAD_GRAYSCALE)

#Farklı sigma değerleri ile bulanıklastir
blur1 = cv2.GaussianBlur(img, (5,5),1 )
blur2 = cv2.GaussianBlur(img, (5,5),2 )

cv2.imshow("blur1",blur1)
cv2.imshow("blur2",blur2)


# iki bulanık goruntu arasındaki fark
dog = cv2.subtract(blur1, blur2)

#göster
plt.imshow(dog,cmap="gray")
plt.title("Gaussianın farkı")
plt.axis("off")
plt.show()