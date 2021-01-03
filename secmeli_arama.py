import cv2
import random


image = cv2.imread("pyramid.jpg")
image =cv2.resize(image,dsize = (600,600))
cv2.imshow("img", image)

ss =cv2.ximagproc.segmentation.createSelectiveSearchSegmentation() #selectivesearch algosunu içeri aktardık.
ss.setBaseImage(image)

ss.switchToSelectiveSearchQuality()

print("start")
rescts = ss.process() #SelectiveSearchSegmentation algosu çalışırıldı ve tspit edilen nesnelerin köşe değerleri rescts'te tutuldu.
#len(rescts)
output = image.copy()


for (x,y,w,h) in  rescts[:50]:  #ilk 50si görüntülendi.
    color = [random.randint(0,255) for j in range(0,3)] #rastgele renk oluşturuldu.
    cv2.rectangle(output, (x,y), (x+w, y+h), color,2)
    
cv2.imshow("output", output)

"""
Seçmeli arama kayan pencere e piramit gösterimi yerine kullanılan bir metodtur.
"""