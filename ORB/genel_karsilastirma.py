import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# İki görüntüyü oku
img1 = cv2.imread('ORB/images/sokak.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('ORB/images/sokak2.jpeg', cv2.IMREAD_GRAYSCALE)

# Sonuçları kaydetmek için liste
results = []

# 1. SIFT
sift = cv2.SIFT_create()

start_time = time.time()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

end_time = time.time()

results.append(('SIFT', len(good_matches), end_time - start_time))

img_sift = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # 2. SURF
# surf = cv2.xfeatures2d.SURF_create(400)

# start_time = time.time()
# kp1, des1 = surf.detectAndCompute(img1, None)
# kp2, des2 = surf.detectAndCompute(img2, None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append([m])

# end_time = time.time()

# results.append(('SURF', len(good_matches), end_time - start_time))

# img_surf = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 3. ORB
orb = cv2.ORB_create()

start_time = time.time()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

end_time = time.time()

results.append(('ORB', len(matches), end_time - start_time))

img_orb = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Sonuçları yazdır
print("\nSonuçlar:")
for alg, match_count, duration in results:
    print(f"{alg}: {match_count} eşleşme, {duration:.4f} saniye")

# Görselleri göster
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(img_sift)
plt.title('SIFT Matches')
plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(img_surf)
# plt.title('SURF Matches')
# plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_orb)
plt.title('ORB Matches')
plt.axis('off')

plt.show()
