import cv2
import matplotlib.pyplot as plt

def image_piramit(img, scale = 1.5, minSize = (224,224)):
    #scale : ölçek değeridir.
    #minSize : tuple çünkü imagelerin hem yükseklikleri hemde genişlik bilgisi var. 224 verdik.
    
    yield img
    
    while True:
        w = int (img.shape[1] /scale) #resize işlemi int ğerler üzerinde çalışır.
        img= cv2.resize(img, dsize = (w,w))
        
        if img.shape[0] < minSize[1] or img.shape[1] <minSize[0] : #yükseklik ve genişlik bilgii minSize ile karşılaştırıldı.
            break #görüntüyü daha fazla küçültme çık.
        yield img
        
img = cv2.imread("husky.jpg")
img =image_piramit(img,1.5,(10,10))

for i, image in enumerate(img):
    print(i)

    if i == 8:
        plt.imshow(image)


# minsize olana kadar 1.5' e bölündü ve farklı görüntüler elde edildi.








