import cv2

img = cv2.imread("messi5.jpg",0)
cv2.imshow("ilk resim", img)
k = cv2.waitKey(0) & 0xFF

if k ==27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite("messi_gray.jpg",img)
    cv2.destroyAllWindows()

#%% Video Aktarma
import time

video_name = " MOT17-04-DPM.mp4 "

cap = cv2.VideoCapture(video_name)

print("genişlik", cap.get(3))
print("yükseklik", cap.get(4))

if cap.isOpened() == False: #video aktarıldı mı kontrol edildi.
    print("hata")


while True:
    ret,frame =cap.read()
    if ret == True: #bağlanma oldu ise
        time.sleep(0.001) #video olduğu için seri bir şekilde akar yavaşlatmak için
        cv2.imshow("Video", frame)
        
    else: break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
cap.release()
cv2.destroyAllWindows()     
        
#%% Kamera Açma ve Video Kaydetme

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width,height)


writer = cv2.VideoWriter("video_kaydı.mp4", cv2.VideoWriter_fourcc(*"DIVX"),20, (width,height))       
        
# fourcc çerceveleri sıkıştırmak için kullanılır.   
# 20 =fps
#boyut bilgisi

while True:
     ret,frame = cap.read()
     cv2.imshow("video", frame)
     
     writer.write(frame)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):break
     
     
cap.release()
writer.release()
cv2.destroyAllWindows()
     
#%% Yeniden Boyutlandırma ve Kırpma

img = cv2.imread("lenna.png", 0)
print("Boyut : ",img.shape)
cv2.imshow("orjinal ", img)
     
imgResized = cv2.resize(img, (800,800))
print("resized image",imgResized.shape)
cv2.imshow("imgResized", imgResized)     
     
#crop
imgCropped = img[:200,0:300]
cv2.imshow("croppedimg",imgCropped )
 
#%%Şekiller ve Metin

import numpy as np
img = np.zeros((512,512,3),np.uint8)
cv2.imshow("siyah", img)


cv2.line(img,(100,100), (100,300), (255,0,0),3)
cv2.imshow("line",img)

cv2.rectangle(img, (0,0),(256,256),(0,255,0),3)
cv2.imshow("rect", img)

cv2.circle(img, (300,300), 15,(0,0,255),3)
cv2.imshow("rect", img)


cv2.putText(img, "resim", (350,350), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
cv2.imshow("rect", img)

#%% Görüntülerin Birleştirilmesi

img = cv2.imread("lenna.png")

hor = np.hstack((img,img))
cv2.imshow("horizantol", hor)

ver = np.vstack((img,img))
cv2.imshow("vertically", ver)

#%% Perspektif Çarpıtma

img = cv2.imread("kart.png")
cv2.imshow("orjinal", img)

width = 400
height = 500

point1 = np.float32 ([[203,1],[1,472],[540,150],[338,617]])

point2 = np.float32 ([[0,0],[0,height],[width,0],[width,height]])
matrix = cv2.getPerspectiveTransform(point1,point2)
print(matrix)

imgoutput = cv2.warpPerspective(img,matrix, (width,height))
cv2.imshow("imgoutput",imgoutput)

#%% Görüntü Karıştırma
import matplotlib.pyplot as plt
img1 = cv2.imread("img1.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("img2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


print(img1.shape)
print(img2.shape)

#boyutlar farlıydı aynı olmalı bu yüzden resize işlemi yapıldı.

img1= cv2.resize(img1,(600,600)) 

img2= cv2.resize(img2,(600,600)) 

#karıştırılmış resim = alpha*img1+ beta*img2

blended = cv2.addWeighted(src1 = img1, alpha=1, src2 = img2, beta = 0.5, gamma =0)
plt.figure()
plt.imshow(blended)

#%% Görüntü Eşikleme
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img, cmap ="gray") #cmap = colormap 
plt.axis("off")
plt.show( )


#eşikleme

_,thresh_img = cv2.threshold(img, thresh = 60, maxval= 255,type =cv2.THRESH_BINARY)
#thresh= 60 dedik eşik 60 oldu 60 üzeri beyaz olur.

plt.figure()
plt.imshow(thresh_img, cmap ="gray")
plt.axis("off")
plt.show()

#uyarlamalı eşik değeri

thresh_img2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)

#◘255= maxvalue
# ADAPTIVE_THRESH_MEAN_C kullanılan yöntem
# c sabiti = 8
#11 blok size

"""
adaptiveThreshold piksel topluluğuna göre karar verilen thrshold değeri ile çalışmaktır.
11 piiksel topluluklarının size değeridir.
"""

plt.figure()
plt.imshow(thresh_img2, cmap ="gray")
plt.axis("off")
plt.show()

#%%Blurring
import numpy as np

#gürültüyü gidermek için kullanılır. Detaylar azaltılır böylece gürültü azaltılır.

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img),plt.axis("off"), plt.show()


#ortalama bulanıklşatırma

dst2 = cv2.blur(img, ksize = (3,3))
plt.figure(), plt.imshow(dst2),plt.axis("off"), plt.title("bulanik"),plt.show()

#gausian blur

gb = cv2.GaussianBlur(img, ksize=(3,3), sigmaX =7)
plt.figure(), plt.imshow(gb),plt.axis("off"), plt.title("Gaussian"),plt.show()

#medyan blur

mb=cv2.medianBlur(img, ksize=(3))
plt.figure(), plt.imshow(gb),plt.axis("off"), plt.title("Median"),plt.show()

def gaussianNoise(image): #gürültü fonksiyonu
    row, col, ch = image.shape
    mean = 0 #gürültünün ortalaması
    var = 0.05
    sigma = var**0.5
    
    gauss = np.random.normal(mean, sigma, (row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss

    return noisy


img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255 #image normalize edildi çünkü gürültü 0-1 arasında 
plt.figure(), plt.imshow(img),plt.axis("off"), plt.show()

gaussianNoiseimage = gaussianNoise(img)
plt.figure(), plt.imshow(gaussianNoiseimage),plt.axis("off"), plt.title("GausianNoise"),plt.show()


gb = cv2.GaussianBlur(gaussianNoiseimage, ksize=(3,3), sigmaX =7)
plt.figure(), plt.imshow(gb),plt.axis("off"), plt.title("GaussianFilter"),plt.show()

#tuz-biber gürültüsü : image üzerine siyah-beyaz noktacıkların bulunması.

 
def SaltPepper(image):
    row, col,ch = image.shape
    s_vs_p = 0.7
    
    amount = 0.004
    
    noisy = np.copy(image)
    
    #salt beyaz noktalar
    
    num_salt =np.ceil(amount*image.size*s_vs_p) #görüntü üzerindeki beyaz gürültü sayısı
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape] #beyaz gürültünün koordinatlarını belirledi
    noisy[coords] = 1 #beyaz gürültü
    
    
    num_pepper =np.ceil(amount*image.size*(1 - s_vs_p))#görüntü üzerindeki siyah gürültü sayısı
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape] #siyah gürültünün koordinatlarını belirledi
    noisy[coords] = 0  #siyah gürültü
    
    return noisy

salt_pepper_image =SaltPepper(img)
plt.figure(), plt.imshow(salt_pepper_image),plt.axis("off"), plt.title("SaltPepper"),plt.show()

mb=cv2.medianBlur(salt_pepper_image.astype(np.float32), ksize=(3))
plt.figure(), plt.imshow(mb),plt.axis("off"), plt.title("Median"),plt.show()


#%%   Morfolojik İşlemler
#erozyon : beyaz noktaların azalmasını sağlar. aşınma işlemi gerçekleşir..
#genişleme : byaz noktaları arttırır.
#açma : erozyon + genişleme gürültü arındırmak için faydaladır
#kapama : genişleme + erozyon 

img = cv2.imread("datai_team.jpg")
plt.figure(), plt.imshow(img, cmap ="gray"), plt.axis("off"), plt.show()

#erozyon

kernel = np.ones((5,5), dtype = np.uint8)
result = cv2.erode(img,kernel, iterations =1)
plt.figure(), plt.imshow(result, cmap ="gray"), plt.axis("off"),plt.title("erode"), plt.show()

#genişleme dilation 

result2 = cv2.dilate(img, kernel, iterations = 1)
plt.figure(), plt.imshow(result2, cmap ="gray"), plt.axis("off"),plt.title("dilate"), plt.show() 
  

#noise  
whiteNoise = np.random.randint(0,2,size = img.shape[:2])
whiteNoise=whiteNoise*255  #gürültü normalize olarak oluştu istenilen scalaya çekildi.
plt.figure(), plt.imshow(whiteNoise, cmap ="gray"), plt.axis("off"),plt.title("noise"), plt.show()     


img = cv2.imread("datai_team.jpg",0)
noise_img = whiteNoise + img
plt.figure(), plt.imshow(noise_img, cmap ="gray"), plt.axis("off"),plt.title("noiseimage"), plt.show()     

#açılma
opening= cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(opening, cmap ="gray"), plt.axis("off"),plt.title("opening"), plt.show() 



blackNoise = np.random.randint(0,2,size = img.shape[:2])
blackNoise=blackNoise*-255  #gürültü normalize olarak oluştu istenilen scalaya çekildi.
plt.figure(), plt.imshow(blackNoise, cmap ="gray"), plt.axis("off"),plt.title("blacknoise"), plt.show()  

black_noise_image =   blackNoise + img
black_noise_image[black_noise_image <= -245] = 0
plt.figure(), plt.imshow(black_noise_image, cmap ="gray"), plt.axis("off"),plt.title("blacknoiseimage"), plt.show() 

#kapama
closing = cv2.morphologyEx(black_noise_image.astype(np.float32), cv2.MORPH_CLOSE )
plt.figure(), plt.imshow(closing, cmap ="gray"), plt.axis("off"),plt.title("closing"), plt.show()
    
#gradient 

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  
plt.figure(), plt.imshow(gradient, cmap ="gray"), plt.axis("off"),plt.title("gradient"), plt.show() 

#%% Gradyanlar

img =cv2.imread("sudoku.jpg",0)
plt.figure(), plt.imshow(img, cmap ="gray"), plt.axis("off"),plt.title("orjinal"), plt.show() 
    
# x gradyanı

sobelx = cv2.Sobel(img, cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure(), plt.imshow(sobelx, cmap ="gray"), plt.axis("off"),plt.title("sobelx"), plt.show() 
#y gradyanı
sobely = cv2.Sobel(img, cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure(), plt.imshow(sobely, cmap ="gray"), plt.axis("off"),plt.title("sobely"), plt.show() 


# x ve y gradyanı
laplace = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure(), plt.imshow(laplace, cmap ="gray"), plt.axis("off"),plt.title("laplace"), plt.show() 
#%% Histogram
#ton dağılımını gösterir. 
import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("red_blue.jpg")
img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img),plt.title("img"), plt.show() 

print(img.shape)


img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
print(img_hist.shape)
plt.figure(), plt.plot(img_hist)

color = ("b", "g", "r")
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels = [i], mask = None, histSize = [256], ranges = [0,256])
    plt.plot(hist, color = c)
    
# 
golden_gate = cv2.imread("goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(golden_gate_vis)    
    
print(golden_gate.shape)

#orjinal resmin bir bölgesin odaklanmak istedik. Bunun için önce sıfırlardan ve beyazdan oluşan bir mask oluşturduk.
mask = np.zeros(golden_gate.shape[:2], np.uint8)
plt.figure(), plt.imshow(mask, cmap = "gray")  

mask[1500:2000, 1000:2000] = 255
plt.figure(), plt.imshow(mask, cmap = "gray") 

masked_img_vis = cv2.bitwise_and(golden_gate_vis, golden_gate_vis, mask = mask) #mask ile and işlemi yaparak sadece istiğimiz noktayı aldık.
plt.figure(), plt.imshow(masked_img_vis, cmap = "gray") 


masked_img = cv2.bitwise_and(golden_gate, golden_gate, mask = mask)
masked_img_hist = cv2.calcHist([golden_gate], channels = [0], mask = mask, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(masked_img_hist) 

# histogram eşitleme
# karşıtlık arttırma : piksel değerlinin 0-255 arası dağılımını arttıma.
img = cv2.imread("hist_equ.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray") 

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(img_hist)

eq_hist = cv2.equalizeHist(img)
plt.figure(), plt.imshow(eq_hist, cmap = "gray") 

eq_img_hist = cv2.calcHist([eq_hist], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(eq_img_hist)

#↓histSize = [256] : 0 ve 255' e kadar olmak üzer 256 değer

#♠ranges = [0,256] : 0'dan 255'e










    
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    