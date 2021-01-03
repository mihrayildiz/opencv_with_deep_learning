import cv2 
import numpy as np

def non_maxi_suppression(boxes, probs = None,overlapThresh = 0.3):
    #boxes = karşılatırılacak kutular
    #probs : dönen olasılık ihtimallleri
    #overlapThresh : seçim için threshold
    
    
    if len(boxes) == 0: #kutu yok ise
        return []
    
    if boxes.type.kind == "i":
        boxes = boxes.astype("float")
    
    x1 = boxes[:,0] #boxes 4 nokta döner başlangıç (x,y) 'si ve bitiş (x,y)'si
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    #alan
    area = (x2-x1+1)*(y2-y1+1)
    
    idx= y2
    
    #olasılık değerleri
    if probs is not None:
        idx =probs
    
    idx = np.argsort(probs)
    
    pick = [] #seçilen kutular
    
    while len(idx) >0 :
        last = len(idx) -1
        i = idx[last]
        pick.append(i)      

    
    #en büyük ve en küçük x ve y
    
       # en buyuk ve en küçük x ve y
       xx1 = np.maximum(x1[i], x1[idxs[:last]])
       yy1 = np.maximum(y1[i], y1[idxs[:last]])
       xx2 = np.minimum(x2[i], x2[idxs[:last]])
       yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # w,h bul
        w = np.maximum(0,xx2 - xx1 + 1)
        h = np.maximum(0,yy2 - yy1 + 1)
        
        # overlap 
        overlap = (w*h)/area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    return boxes[pick].astype("int")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    