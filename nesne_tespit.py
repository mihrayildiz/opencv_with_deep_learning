import cv2
import matplotlib.pyplot as plt
import numpy as np


#%% Canny
img = cv2.imread("london.jpg",0)
plt.figure(), plt.imshow(img, cmap ="gray"),plt.axis("off")


edges =cv2.Canny(img, threshold1 =0, threshold2 = 255)
plt.figure(), plt.imshow(edges, cmap ="gray"),plt.axis("off")

"""
Edges ile irlikte gereksiz alanlar içinde tespit yaptı örneğin su için 
bu yüzden threshold değerleri ile oynamak tercih edilebir.
"""
med_val = np.median(img)
print(med_val)

low = int(max(0, (1-0.33)*med_val))
high = int(min(255, (1+0.33)*med_val))

edges =cv2.Canny(img, threshold1 =low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap ="gray"),plt.axis("off")

#tam istenilen ünede olmadı bu yüzden blur işlemi uygulandı.

blurred_img = cv2.blur(img, ksize=(5,5))
plt.figure(), plt.imshow(blurred_img, cmap ="gray"),plt.axis("off")

med_val = np.median(blurred_img)
print(med_val)


low = int(max(0, (1-0.33)*med_val))
high = int(min(255, (1+0.33)*med_val))


edges =cv2.Canny(blurred_img, threshold1 =low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap ="gray"),plt.axis("off")


#%%%Köşe Algılama

img =cv2.imread("sudoku.jpg",0)
print(img.shape)
plt.figure(), plt.imshow(img, cmap ="gray"),plt.axis("off")

dst =cv2.cornerHarris(img, blockSize =2, ksize =3, k =0.04)
plt.figure(), plt.imshow(dst, cmap ="gray"),plt.axis("off")

dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap ="gray"),plt.axis("off")

#shi tomsai detection
img =cv2.imread("sudoku.jpg",0)
img = np.float32(img)
edges =cv2.goodFeaturesToTrack(img, 100, 0.01,10)
edges =np.int64(edges)

for i in edges :
    x,y =i.ravel()
    cv2.circle(img,(x,y),3,(125,125,125), cv2.FILLED)


plt.imshow(img)
plt.axis("off")

#100 = kaç köşe olun
#10 minimum distance

#%% Kontur Agılama

img = cv2.imread("contour.jpg",0)
plt.figure(), plt.imshow(img, cmap ="gray"),plt.axis("off")

counturs, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(counturs)
external_counturs = np.zeros(img.shape)
internal_counturs = np.zeros(img.shape)


for i in range(len(counturs)):
    if hierarch[0][i][3] ==-1:
        cv2.drawContours(external_counturs, counturs, i, 255, -1)
    else:
          cv2.drawContours(internal_counturs, counturs, i, 255 ,-1)
          
plt.figure(), plt.imshow(external_counturs, cmap ="gray"),plt.axis("off")
plt.figure(), plt.imshow(internal_counturs, cmap = "gray"),plt.axis("off")

#%% Şablon Eşleme (template matching)

"""
şablon göüntüsünü daha büyük bir görüntü üzerinde gezdierek konumda eşleşme yapılır.
"""
img = cv2.imread("cat.jpg", 0)
print(img.shape)
template =cv2.imread("cat_face.jpg",0)
print(template.shape)
h,w = template.shape


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    method =eval(meth)
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#eval metodu stringi methoda çevirir.

  if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray")
    plt.title("Eşleşen Sonuç"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap = "gray")
    plt.title("Tespit edilen Sonuç"), plt.axis("off")
    plt.suptitle(meth)
    
    plt.show()



#%% özellik Eşleştirme

import cv2
import matplotlib.pyplot as plt

# ana görüntüyü içe aktar
chos = cv2.imread("chocolates.jpg", 0)
plt.figure(), plt.imshow(chos, cmap = "gray"),plt.axis("off")

# aranacak olan görüntü
cho = cv2.imread("nestle.jpg", 0)
plt.figure(), plt.imshow(cho, cmap = "gray"),plt.axis("off")

# orb tanımlayıcı
# köşe-kenar gbi nesneye ait özellikler
orb = cv2.ORB_create()

# anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

# bf matcher, eşleşme yapıldı
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# noktaları eşleştir
matches = bf.match(des1, des2)

# mesafeye göre sırala
matches = sorted(matches, key = lambda x: x.distance)

# eşleşen resimleri görselleştirelim
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 2)
plt.imshow(img_match), plt.axis("off"),plt.title("orb")

# sift. tekrar özelikler çıkarıldı
sift = cv2.xfeatures2d.SIFT_create()

# bf
bf = cv2.BFMatcher()

# anahtar nokta tespiti sift ile
kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)

matches = bf.knnMatch(des1, des2, k = 2)

guzel_eslesme = []

for match1, match2 in matches:
    
    if match1.distance < 0.75*match2.distance:
        guzel_eslesme.append([match1])
    
plt.figure()
sift_matches = cv2.drawMatchesKnn(cho,kp1,chos,kp2,guzel_eslesme,None, flags = 2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("sift")

#%% Kedi Yüzü Tespit Etme
import cv2
import os

files = os.listdir()
#print(files)


img_path_list= [] #imageleri içe aktarma
for f in files:
    if f.endswith(".jpg"):
        img_path_list.append((f))
print(img_path_list)


for j in img_path_list: #görselleştirme
    print(j)
    image =cv2.imread(j)
    gray_image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cat_face = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    face = cat_face.detectMultiScale(gray_image, scaleFactor =1.045, minNeighbors = 2)
    for (i,(x,y,w,h)) in enumerate((face))  :
        cv2.rectangle(image,(x,y), (x+w, y+h), (255,255,0), 3)
        cv2.putText(image, "kedi {}".format(i+1), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0),2)
    
    cv2.imshow(j,image)
    if  cv2.waitKey(0) & 0xFF == ord('q'): continue #q ya bastıkça image değişsin


#%% Yaya Algılama

files =os.listdir()
img_path_list=[]

for f in files :
    if f.endswith(".jpg"):
        img_path_list.append(f)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for i in img_path_list:
    print(i)

    image = cv2.imread(i)
    (rects,weights) = hog.detectMultiScale(image, padding =(8,8), scale= 1.05)
    for (x,y,w,h) in rects:
       cv2.rectangle(image,(x,y), (x+w, y+h), (255,255,0), 3)
   
    cv2.imshow("yaya :",image)
    if  cv2.waitKey(0) & 0xFF == ord('q'): continue






























