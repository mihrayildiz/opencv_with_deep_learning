import cv2
import matplotlib.pyplot as plt

def kayan_pencere(img, step, ws):
    
    #step : pecere görüntü üzerinde kaç piksel kaysın
    #ws : pencerenin size
    for y in range(0, img.shape[0] -ws[1], step):
        for x in range(0, img.shape[1] - ws[0], step):
            yield (x,y, img[y:y+ws[1], x:x+ws[0]])
            
img = cv2.imread("husky.jpg")
img = kayan_pencere(img, 5, (200,150))

for i,mage in enumerate(img):
    print(i)
    if i == 37:
        print(mage[0], mage[1])
        plt.imshow(mage[2])